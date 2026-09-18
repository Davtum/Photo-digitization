"""Отрезки и точки схода. Спецификация, п. 4.2."""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


def detect_segments(gray: np.ndarray, min_length_px: float = 40.0) -> np.ndarray:
    """Отрезки на изображении, отфильтрованные по длине.

    Требует OpenCV >= 4.8.0: в 4.1–4.7 LSD был изъят из-за патента.
    """
    if gray.ndim != 2:
        raise ValueError("ожидается одноканальное изображение")

    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        return np.empty((0, 4), dtype=float)

    segs = lines.reshape(-1, 4).astype(float)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= min_length_px]


@dataclass(frozen=True)
class VanishingPoint:
    """Точка схода: однородные координаты, число поддержавших отрезков, невязка."""

    point: np.ndarray       # однородные координаты (3,)
    support: int
    residual_px: float


@dataclass(frozen=True)
class PlaneConfidence:
    """Мера доверия к восстановленной плоскости фасада. Спецификация, п. 4.2.

    Не диагностика, а часть контракта: ниже порога система обязана не гадать,
    а передать управление оператору. Список `reasons` называет, что именно
    помешало довериться результату.
    """

    value: float
    support_h: int
    support_v: int
    residual_px: float
    orthogonality_deg: float
    coverage: float
    reasons: list[str] = field(default_factory=list)


def _segment_lines(segs: np.ndarray) -> np.ndarray:
    """Прямые в однородных координатах через векторное произведение концов."""
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    return np.cross(p1, p2)


def _midpoints(segs: np.ndarray) -> np.ndarray:
    """Середины отрезков в однородных координатах."""
    return np.column_stack([(segs[:, 0] + segs[:, 2]) / 2.0,
                            (segs[:, 1] + segs[:, 3]) / 2.0,
                            np.ones(len(segs))])


def endpoint_deviation_px(segs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Отклонение концов отрезков от направления на точку схода, в пикселях.

    Для отрезка длиной L, чьё направление отклонилось от направления «середина —
    точка схода» на угол alpha, величина равна (L/2)·sin(alpha). Это настоящие
    пиксели кадра, и они не зависят от удалённости точки схода: у почти
    параллельного пучка, чья точка схода уходит на 1e5 px и дальше, мера остаётся
    конечной и малой ровно тогда, когда отрезки действительно сонаправлены.

    В прежней редакции критерием была нормированная на ‖v‖ дистанция прямой до
    точки схода: фактический допуск в пикселях равнялся threshold·‖v‖ и рос без
    границ вместе с удалением точки схода, так что на почти параллельном пучке
    инлайерами становились все отрезки подряд.
    """
    mid = _midpoints(segs)
    rays = np.cross(mid, np.asarray(v, dtype=float))   # прямая «середина — точка схода»
    norms = np.linalg.norm(rays[:, :2], axis=1)
    ends = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    dev = np.abs(np.einsum("ij,ij->i", ends, rays)) / np.maximum(norms, 1e-12)
    half_len = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) / 2.0
    # Вырожденный случай: точка схода села на середину отрезка — направление на неё
    # не определено, и отрезок такую гипотезу не поддерживает.
    return np.where(norms > 1e-9, dev, half_len)


def _hypothesis_pairs(n: int, max_hypotheses: int, rng) -> np.ndarray:
    """Пары прямых-гипотез: полный перебор, пока он дешевле выборки.

    При n <= 63 все n(n-1)/2 пар укладываются в бюджет, перебор исчерпывающий, и
    результат перестаёт зависеть от seed. Выборка включается только там, где полный
    перебор действительно дорог. Прежние 200 случайных пар недосэмплировали уже
    27 прямых (351 пара).
    """
    if n * (n - 1) // 2 <= max_hypotheses:
        return np.column_stack(np.triu_indices(n, k=1))
    i = rng.integers(0, n, size=max_hypotheses)
    j = (i + 1 + rng.integers(0, n - 1, size=max_hypotheses)) % n
    return np.column_stack([i, j])


def _fit_vanishing_point(segs: np.ndarray, threshold_px: float, rng,
                         max_hypotheses: int = 2000):
    """RANSAC по парам прямых с пиксельным критерием инлайера.

    Возвращает точку схода, маску поддержки и среднее отклонение инлайеров в пикселях.
    """
    empty = np.zeros(len(segs), dtype=bool)
    if len(segs) < 2:
        return np.array([0.0, 0.0, 1.0]), empty, float("inf")

    lines = _segment_lines(segs)
    best_point, best_mask, best_resid = None, empty, float("inf")
    for i, j in _hypothesis_pairs(len(segs), max_hypotheses, rng):
        v = np.cross(lines[i], lines[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        dev = endpoint_deviation_px(segs, v)
        mask = dev < threshold_px
        resid = float(dev[mask].mean()) if mask.any() else float("inf")
        if mask.sum() > best_mask.sum() or (mask.sum() == best_mask.sum()
                                            and resid < best_resid):
            best_point, best_mask, best_resid = v, mask, resid

    if best_point is None or not best_mask.any():
        return np.array([0.0, 0.0, 1.0]), empty, float("inf")

    # Уточнения по МНК здесь намеренно нет. Оно напрашивается — гипотезу строили две
    # прямые, а поддержал десяток, — но МНК по прямым минимизирует алгебраическую
    # невязку, а не отклонение концов в пикселях, и на деле проигрывает: без охраны
    # на одной сцене улучшает точность втрое, на другой во столько же ухудшает, а под
    # охраной «принимать, только если не хуже по тому же правилу, каким выбиралась
    # гипотеза» не принимается ни на одном из проверенных входов.
    # Это наблюдение на выборке, а не доказательство недостижимости: утверждать, что
    # ветка мертва при любом входе, оснований нет. Решение не вносить уточнение
    # опирается именно на измерение.
    dev = endpoint_deviation_px(segs, best_point)
    return best_point, best_mask, float(dev[best_mask].mean())


def _split_bundles(segments: np.ndarray):
    """Разделение на горизонтальный и вертикальный пучки по ближайшей оси кадра.

    Правило «отрезок принадлежит той оси, на которую он больше проецируется»
    записано прямым сравнением проекций, без порога в градусах: пороговой
    константы здесь нет, а значит её нечего молча сдвинуть.
    """
    dx = np.abs(segments[:, 2] - segments[:, 0])
    dy = np.abs(segments[:, 3] - segments[:, 1])
    is_vert = dy > dx
    return segments[~is_vert], segments[is_vert]


def direction_instability_deg(segs: np.ndarray, K: np.ndarray) -> float:
    """Угловая неустойчивость НАПРАВЛЕНИЯ K⁻¹v, в градусах.

    Потребитель использует направление, а не пиксельное положение точки схода.
    Это разные по устойчивости величины: при фронтальной съёмке точка схода уходит
    в бесконечность и её положение не определено по построению, тогда как
    направление определено превосходно — а фронтальный кадр для задачи наилучший,
    а не худший. Прежняя мера считала устойчивость положения и потому отвергала
    кадр в упор с ошибкой направления в восемь тысячных градуса.

    Направление — правый нуль-вектор системы (K^T·l)^T·d = 0, составленной прямо в
    пространстве направлений. По теории возмущений нуль-вектор поворачивается
    примерно на σ3/σ2 — это консервативная граница сверху, на годных сценах она
    превышает фактическую ошибку в десятки раз. Деление на число поддержавших
    прямых m приводит границу к рабочему предсказателю: множитель подобран
    измерением, а не выведен. На выборке из 144 замеров (48 сцен и 24 почти
    параллельных входа) варианты сравнивались по разделению популяций и связи с
    фактической ошибкой направления:

        σ3/σ2        точность 0.877, корреляция +0.542
        σ3/σ2/√m     точность 0.935, корреляция +0.674
        σ3/σ2/m      точность 0.957, корреляция +0.726   <- принят
        медиана/веер точность 0.913, корреляция +0.641
        разброс по половинам  точность 0.929, корреляция -0.074
    """
    if len(segs) < 2:
        return float("inf")
    rows = _segment_lines(segs) @ np.asarray(K, dtype=float)
    rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    sv = np.linalg.svd(rows, compute_uv=False)
    if sv[-2] < 1e-12:
        return float("inf")
    return float(np.degrees(sv[-1] / sv[-2]) / len(segs))


def _support_spread(segs: np.ndarray, image_size: tuple[int, int]) -> float:
    """Доля кадра, которую охватывают середины поддержавших отрезков.

    Пучок, сгрудившийся в одном углу, не свидетельствует о плоскости всего кадра,
    даже если внутри себя он идеально согласован.
    """
    if len(segs) == 0:
        return 0.0
    mid = _midpoints(segs)[:, :2]
    span = mid.max(axis=0) - mid.min(axis=0)
    rel = span / np.maximum(np.asarray(image_size, dtype=float), 1.0)
    return float(np.hypot(*rel) / math.sqrt(2.0))


MIN_SEGMENTS = 8
INLIER_PX = 2.0
# Доля допуска, выше которой средняя невязка инлайеров сама становится причиной
# отказа. Именно доля, а не абсолютные пиксели: инлайер по определению лежит ближе
# threshold_px, так что абсолютный порог выше допуска был бы недостижим — ровно тот
# мёртвый код, что уже был найден в проверке неопределённой невязки.
RESIDUAL_MAX_FRACTION = 0.75
RESIDUAL_REF_PX = 1.5
ORTHOGONALITY_MAX_DEG = 15.0
MIN_COVERAGE = 0.2
MIN_SUPPORT_TOTAL = 10
MIN_SPREAD = 0.2
# Порог угловой неустойчивости направления. Подобран по той же выборке из 144
# замеров: отделяет замеры с ошибкой ниже 0.5° от замеров с ошибкой выше 1.0° с
# точностью 0.957. Популяции перекрываются, полного разделения нет — метрика
# остаётся мерой доверия, а не предсказателем точности.
MAX_INSTABILITY_DEG = 0.15
# Масштаб, на котором неустойчивость снижает доверие. Мягче порога намеренно:
# оценка неустойчивости консервативна и на годных сценах превышает фактическую
# ошибку направления в десятки раз.
INSTABILITY_REF_DEG = 0.30
# Порог передачи оператору. Ниже него список причин обязан быть непуст:
# доверие ниже порога без названной причины вызывает оператора, не объясняя зачем.
CONFIDENCE_THRESHOLD = 0.5


def estimate_vanishing_points(
    segments: np.ndarray,
    image_size: tuple[int, int],
    K: np.ndarray,
    threshold_px: float = INLIER_PX,
    seed: int = 0,
) -> tuple[VanishingPoint, VanishingPoint, PlaneConfidence]:
    """Две ортогональные точки схода плюс мера доверия к плоскости.

    Доверие — часть контракта, а не диагностика: ниже порога система обязана не
    гадать, а передать управление оператору (спецификация, п. 4.2). Поэтому `value`
    зависит не только от того, какая доля отрезков поддержала гипотезу, но и от
    невязки инлайеров в пикселях, и от отклонения восстановленных направлений от
    прямого угла.
    """
    reasons: list[str] = []
    if len(segments) < MIN_SEGMENTS:
        reasons.append("слишком мало отрезков")
        empty = VanishingPoint(np.array([1.0, 0.0, 0.0]), 0, float("inf"))
        return empty, empty, PlaneConfidence(0.0, 0, 0, float("inf"), 0.0, 0.0, reasons)

    horiz, vert = _split_bundles(segments)

    rng = np.random.default_rng(seed)
    vh_pt, vh_mask, vh_res = _fit_vanishing_point(horiz, threshold_px, rng)
    vv_pt, vv_mask, vv_res = _fit_vanishing_point(vert, threshold_px, rng)

    Kinv = np.linalg.inv(K)
    dh, dv = Kinv @ vh_pt, Kinv @ vv_pt
    dh, dv = dh / (np.linalg.norm(dh) + 1e-12), dv / (np.linalg.norm(dv) + 1e-12)
    orthogonality = math.degrees(math.acos(min(1.0, abs(float(dh @ dv)))))

    support_h, support_v = int(vh_mask.sum()), int(vv_mask.sum())
    coverage = (support_h + support_v) / max(len(segments), 1)
    # Невязка в настоящих пикселях кадра: среднее отклонение концов инлайеров от
    # направления на точку схода. В прежней редакции здесь стояла безразмерная
    # величина, умноженная на max(image_size), — произвольный множитель, а не
    # перевод в пиксели; расхождение с истинной дистанцией достигало 34 раз.
    finite = [r for r in (vh_res, vv_res) if np.isfinite(r)]
    residual = float(np.mean(finite)) if finite else float("inf")
    supporting = (np.vstack([horiz[vh_mask], vert[vv_mask]])
                  if support_h + support_v else np.empty((0, 4)))
    spread = _support_spread(supporting, image_size)
    instability = max(direction_instability_deg(horiz[vh_mask], K),
                      direction_instability_deg(vert[vv_mask], K))

    if support_h < 2 or support_v < 2:
        reasons.append("недостаточная поддержка одной из точек схода")
    if abs(orthogonality - 90.0) > ORTHOGONALITY_MAX_DEG:
        reasons.append(f"направления не ортогональны: {orthogonality:.1f}°")
    if coverage < MIN_COVERAGE:
        reasons.append("малая доля отрезков поддержала точки схода")
    # Проверки на неопределённую невязку здесь нет намеренно: она недостижима.
    # residual бесконечна только когда бесконечны обе невязки пучков, а каждая из
    # них бесконечна только при нулевой поддержке своего пучка — то есть этот
    # случай уже перехвачен проверкой support_h < 2 or support_v < 2 выше.
    if residual > RESIDUAL_MAX_FRACTION * threshold_px:
        reasons.append(f"велика невязка инлайеров: {residual:.2f} px")
    if spread < MIN_SPREAD:
        reasons.append("поддержавшие отрезки занимают малую долю кадра")
    if instability > MAX_INSTABILITY_DEG:
        reasons.append(
            f"направление точки схода определено неустойчиво: "
            f"±{instability:.2f}°")
    if support_h + support_v < MIN_SUPPORT_TOTAL:
        reasons.append(f"мало поддерживающих отрезков: {support_h + support_v}")

    # Доверие падает от каждой из трёх причин порознь: от нехватки поддержки, от
    # роста невязки и от ухода угла между направлениями от прямого. Прежняя редакция
    # была аффинной функцией одного лишь покрытия, из-за чего пучок почти
    # параллельных прямых с дрожанием в 1° получал доверие до 1.00 при пустых
    # причинах и разбросе направления в 6° по seed.
    q_coverage = min(1.0, 0.3 + 0.7 * coverage)
    q_residual = 1.0 / (1.0 + residual / RESIDUAL_REF_PX)
    q_orthogonality = max(0.0, 1.0 - abs(orthogonality - 90.0) / ORTHOGONALITY_MAX_DEG)
    q_stability = 1.0 / (1.0 + instability / INSTABILITY_REF_DEG)
    value = (0.0 if reasons
             else q_coverage * q_residual * q_orthogonality * q_stability)

    # Структурный инвариант контракта (спецификация, п. 4.2): доверие ниже порога
    # обязано сопровождаться названной причиной. Дискретные проверки покрывают не
    # всё — доверие есть произведение непрерывных множителей и может уйти под порог,
    # когда каждая проверка порознь пройдена. Тогда причиной объявляется тот
    # множитель, который просел сильнее прочих.
    if value < CONFIDENCE_THRESHOLD and not reasons:
        factors = [
            (f"малая доля отрезков поддержала точки схода: {coverage:.2f}", q_coverage),
            (f"велика невязка инлайеров: {residual:.2f} px", q_residual),
            (f"направления не ортогональны: {orthogonality:.1f}°", q_orthogonality),
            (f"направление точки схода определено неустойчиво: ±{instability:.2f}°",
             q_stability),
        ]
        worst = min(factors, key=lambda item: item[1])[0]
        reasons.append(f"доверие ниже порога {CONFIDENCE_THRESHOLD}: {worst}")

    return (
        VanishingPoint(vh_pt, support_h, residual),
        VanishingPoint(vv_pt, support_v, residual),
        PlaneConfidence(value, support_h, support_v, residual, orthogonality, coverage, reasons),
    )
