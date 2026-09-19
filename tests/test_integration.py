# -*- coding: utf-8 -*-
"""Сквозная проверка геометрической цепочки через РЕАЛЬНЫЙ детектор отрезков.

Единственное место, где восстановленная поза сравнивается с истиной сцены по всей
цепи: рендер -> `detect_segments` -> `estimate_vanishing_points` -> гомография ->
`camera_pose`. Точные точки схода здесь НЕ подставляются — ими проверены отдельные
звенья (`tests/test_homography.py`), и на них невязка опорной точки составляет
1.6e-14 % расстояния, то есть двенадцать порядков ниже рабочей. Тест, подставивший
их сюда, измерял бы арифметику, а не конвейер; охрана от такой подмены —
`test_chain_really_runs_through_the_detector`.

**Что здесь считается масштабом.** `camera_pose` требует числа миллиметров на
ректифицированную единицу, а оно появляется только после того, как оператор задал
опорный размер (см. докстринг `pipeline/plane.py`). Роль оператора играют две
величины, взятые из истины сцены: точка начала отсчёта (левый нижний угол фасада) и
одна опорная длина (низ фасада, 20000 мм). Больше из истины не берётся НИЧЕГО —
гомография, точки схода и поза приходят из кадра. Следствие, которое нельзя
затушёвывать: нижняя сторона фасада в ректифицированных миллиметрах выходит равной
20000 мм тождественно, по построению масштаба, и проверкой не является. Проверяются
стороны и углы, в калибровку не входившие.

**Стоимость.** Файл прогоняет 72 полных цепочки на кадрах 5280x3956; детекция
отрезков занимает около 1.4 с на кадр, весь файл — порядка двух минут. Результат
каждого ракурса считается один раз и переиспользуется всеми проверками.
"""
import math
from dataclasses import dataclass

import numpy as np
import pytest

from facade_digitizer.geometry.homography import (
    camera_pose,
    homography_from_vanishing_points,
    rectified_to_facade_mm,
)
from facade_digitizer.pipeline.plane import estimate_plane
from facade_digitizer.pipeline.rectify import attainable_mm_per_px, rectify
from tests.test_homography import CORNERS, apply, exact_vps
from tests.test_synth import K, SIZE, make_rich_scene, make_scene

FACADE_W_MM = 20000.0
FACADE_H_MM = 15000.0

# Сетка ракурсов. Вертикальное смещение камеры ОБЯЗАТЕЛЬНО ненулевое: замер задачи 8
# показал, что худшие оценки даёт именно оно, а сетка с dy = 0 занижает невязку.
# Перемерено здесь на расширенной сетке (dy из {-2000, 0, +2000}, 54 ракурса на
# сцену, вне этого файла): на разреженной сцене P95 невязки опорной точки при dy = 0
# равен 0.133 %, при dy = +-2000 — 0.601 %, то есть в 4.5 раза больше. Число из
# сетки без вертикального смещения описывало бы не конвейер, а удобный срез.
DX_MM = (-3000.0, -1500.0, 0.0, 1500.0, 3000.0, 4500.0)
DY_MM = (-2000.0, 2000.0)
DIST_MM = (9000.0, 12000.0, 15000.0)
GRID = tuple((dx, dy, dist) for dx in DX_MM for dy in DY_MM for dist in DIST_MM)

SPARSE = "разреженная"
RICH = "обогащённая"
SCENES = {SPARSE: make_scene, RICH: make_rich_scene}

# Масштаб ректифицированного растра. Значение не произвольно: тест сверяет его с
# границами `attainable_mm_per_px`, то есть с тем, что растр получается возможного
# размера, а не подбирает масштаб вслепую.
RECT_MM_PER_PX = 10.0

# Сетка контрольных точек фасада для невязки АБСОЛЮТНОГО положения (п. 2.2):
# 5x5 по габариту плюс четыре угла проёма.
_gx, _gy = np.meshgrid(np.linspace(0.0, FACADE_W_MM, 5), np.linspace(0.0, FACADE_H_MM, 5))
PROBE_MM = np.vstack([
    np.column_stack([_gx.ravel(), _gy.ravel()]),
    np.array([[4000.0, 3000.0], [5460.0, 3000.0], [5460.0, 4900.0], [4000.0, 4900.0]]),
])


@dataclass(frozen=True)
class ChainResult:
    """Что цепочка выдала на одном ракурсе.

    **Знаменателей у относительных невязок ДВА, и они разные.** Прежний докстринг
    называл все невязки «процентами расстояния»; это неверно, и неверно опасно:
    величины с разными знаменателями нельзя ни складывать, ни мерить одним допуском.
    Сами знаменатели выбраны правильно и не меняются — меняются имена, чтобы их
    нельзя было перепутать.

    * `foot_pct_of_distance`, `distance_pct_of_distance` — доли РАССТОЯНИЯ ОТ КАМЕРЫ
      ДО ПЛОСКОСТИ (`truth.cz`, здесь 9000…15000 мм). Так нормируется поза: обе её
      величины измеряются в той же единице, в какой определена сама поза.
    * `position_pct_of_lever` — доля ПЛЕЧА контрольной точки от начала координат
      фасада (здесь 3750…25000 мм). Так п. 2.2 нормирует невязку абсолютного
      положения, и так п. 6.4 выводит допуск: «0.2-0.5 % на фасаде 25 м это
      50-125 мм на дальнем краю».

    Одно и то же число процентов означает в этих двух случаях разные миллиметры: на
    ракурсе с дистанцией 9000 мм один процент позы есть 90 мм, а один процент
    положения дальнего угла габарита — 250 мм.
    """

    needs_operator: bool
    reasons: tuple
    confidence: float
    half_turn_resolved: bool = False
    foot_pct_of_distance: float = float("nan")
    distance_pct_of_distance: float = float("nan")
    angles_deg: tuple = ()
    top_mm: float = float("nan")
    left_mm: float = float("nan")
    right_mm: float = float("nan")
    raster_angles_deg: tuple = ()
    raster_top_mm: float = float("nan")
    raster_left_mm: float = float("nan")
    position_pct_of_lever: float = float("nan")
    position_abs_max_mm: float = float("nan")
    attainable_lo_mm_per_px: float = float("nan")
    attainable_hi_mm_per_px: float = float("nan")


def _corner_angles_deg(quad):
    """Углы четырёхугольника при каждой из четырёх вершин, в градусах."""
    out = []
    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        out.append(float(math.degrees(math.acos(min(1.0, cos)))))
    return tuple(out)


_CACHE = {}


def chain(kind, dx, dy, dist):
    """Полная цепочка на одном ракурсе. Считается один раз на ракурс."""
    key = (kind, dx, dy, dist)
    if key in _CACHE:
        return _CACHE[key]

    scene = SCENES[kind](dx=dx, dy=dy, dist=dist)
    image = scene.render()
    plane = estimate_plane(image, K)

    if plane.needs_operator:
        _CACHE[key] = ChainResult(needs_operator=True,
                                  reasons=tuple(plane.confidence.reasons),
                                  confidence=float(plane.confidence.value))
        return _CACHE[key]

    # Роль оператора: начало отсчёта и одна опорная длина.
    rect_corners = apply(plane.H, scene.project(CORNERS))
    origin_rect = tuple(rect_corners[0])
    mm_per_unit = FACADE_W_MM / float(np.linalg.norm(rect_corners[1] - rect_corners[0]))

    # Восстановление позы возвращает ПАРУ; признак неразрешённости не отбрасывается.
    camera, half_turn = camera_pose(plane.H, plane.vh, plane.vv, K,
                                    mm_per_unit, origin_rect)
    truth = scene.camera_on_plane()

    probe_rect = apply(plane.H, scene.project(PROBE_MM))
    probe_mm = rectified_to_facade_mm(probe_rect, origin_rect, mm_per_unit)
    deviation = np.linalg.norm(probe_mm - PROBE_MM, axis=1)
    lever = np.linalg.norm(PROBE_MM, axis=1)
    far = lever > 1.0                    # у самого начала координат доля не определена
    assert int(far.sum()) == len(PROBE_MM) - 1, (
        "плечо не определено более чем у одной контрольной точки: "
        f"{len(PROBE_MM) - int(far.sum())} из {len(PROBE_MM)}")

    # Границы достижимого масштаба сохраняются, а утверждение о них вынесено в
    # отдельную проверку: `lo <= 10 <= hi` выполняется с запасом 11x снизу и 1300x
    # сверху, то есть ошибку масштаба меньше семикратной не различает вовсе. Сверх
    # того, `rectify` ниже сверяет `mm_per_px` с теми же границами сам и отказывает
    # громко, так что охраной этот assert не был и здесь.
    lo, hi = attainable_mm_per_px(image, plane.H, mm_per_unit)
    rectified = rectify(image, plane.H, mm_per_unit, RECT_MM_PER_PX,
                        origin_rect_units=origin_rect)
    px_corners = apply(rectified.H, scene.project(CORNERS))
    ox, oy = rectified.origin_rect_px
    raster_mm = np.column_stack([(px_corners[:, 0] - ox) * rectified.mm_per_px,
                                 -(px_corners[:, 1] - oy) * rectified.mm_per_px])

    _CACHE[key] = ChainResult(
        needs_operator=False,
        reasons=tuple(plane.confidence.reasons),
        confidence=float(plane.confidence.value),
        half_turn_resolved=bool(half_turn.resolved),
        foot_pct_of_distance=(
            100.0 * float(np.hypot(camera.cx - truth.cx, camera.cy - truth.cy)) / truth.cz),
        distance_pct_of_distance=100.0 * abs(camera.cz - truth.cz) / truth.cz,
        angles_deg=_corner_angles_deg(rect_corners),
        top_mm=float(np.linalg.norm(rect_corners[2] - rect_corners[3])) * mm_per_unit,
        left_mm=float(np.linalg.norm(rect_corners[3] - rect_corners[0])) * mm_per_unit,
        right_mm=float(np.linalg.norm(rect_corners[2] - rect_corners[1])) * mm_per_unit,
        raster_angles_deg=_corner_angles_deg(raster_mm),
        raster_top_mm=float(np.linalg.norm(raster_mm[2] - raster_mm[3])),
        raster_left_mm=float(np.linalg.norm(raster_mm[3] - raster_mm[0])),
        position_pct_of_lever=float((100.0 * deviation[far] / lever[far]).max()),
        position_abs_max_mm=float(deviation.max()),
        attainable_lo_mm_per_px=float(lo),
        attainable_hi_mm_per_px=float(hi),
    )
    return _CACHE[key]


def accepted(kind):
    """Ракурсы, принятые метрикой доверия. Отвергнутые в статистику не входят.

    Статистика по отвергнутым описывала бы поведение на данных, которые конвейер не
    принимает: при низком доверии гомография не строится вовсе (контракт п. 4.2).

    **Размер выборки проверяется ЗДЕСЬ, а не подразумевается.** Четыре проверки
    файла были написаны как `for result in accepted(kind)` без охраны на пустоту:
    при нуле принятых ракурсов цикл не выполнял ни одного утверждения, и тест
    сообщал об успехе. Доказано двумя способами — подменой этой функции на
    возвращающую пустой список и мутацией в `vanishing.py`, после которой 70
    ракурсов из 72 отвергались, а три теста стояли в junit-XML как успешные со
    временем 0.001 с. Поэтому число принятых сверяется с ожидаемым точно: оно есть
    дополнение к списку `SPARSE_REFUSED`, то есть к уже сделанному утверждению о
    том, где именно конвейер отказывается, и самостоятельной величиной не является.
    """
    rows = [chain(kind, *view) for view in GRID if not chain(kind, *view).needs_operator]
    assert len(rows) == EXPECTED_ACCEPTED[kind], (
        f"{kind}: принято {len(rows)} ракурсов из {len(GRID)}, ожидалось "
        f"{EXPECTED_ACCEPTED[kind]}; выборка другого размера несопоставима с замером, "
        f"а пустая сделала бы проверки холостыми"
    )
    return rows


@dataclass(frozen=True)
class Residual:
    """Измеренные медиана, P95 и максимум одной невязки, в процентах.

    В процентах ЧЕГО — определяется тем, какое поле `ChainResult` сюда подано, и
    знаменателей этих два (см. докстринг `ChainResult`). Здесь величина безымянна
    намеренно: сводка одинаково считает и долю расстояния, и долю плеча, а смешивать
    их запрещено на уровне вызова, где имя поля знаменатель и называет.
    """

    median: float
    p95: float
    maximum: float


def summarise(values):
    v = np.asarray(values, dtype=float)
    return Residual(float(np.median(v)), float(np.percentile(v, 95)), float(v.max()))


# ---------------------------------------------------------------------------
# Измеренные значения. Сняты на сетке GRID, OpenCV 4.14.0, NumPy 2.4.6.
# Оценщик точек схода детерминирован (seed=0), детектор отрезков тоже, и это
# проверено: три прогона в отдельных процессах, включая однопоточный BLAS, дали
# побитово одинаковые числа.
#
# **ПРОТОКОЛ СТАТИСТИКИ НАЗВАН ЯВНО.** Невязка положения считается так: на каждом
# ракурсе берётся МАКСИМУМ доли плеча по 28 контрольным точкам с определённым
# плечом, и уже по ракурсам берутся медиана, P95 и максимум. Это чтение (а).
# Два других чтения тех же данных дают ДРУГИЕ числа, и разброс между ними — часть
# результата, а не придирка:
#
#     чтение                                  разреженная P95   обогащённая P95
#     (а) макс по точкам -> P95 по ракурсам   0.8057 %          0.0813 %
#     (б) P95 по объединённой выборке пар     0.3457 %          0.0542 %
#     (в) только дальняя точка габарита       0.4376 %          0.0486 %
#
# Чтение (а) выбрано потому, что допуск п. 2.2 предъявляется КАЖДОЙ измеренной
# точке, а не выборке в среднем; (б) занижает величину в 2.3 раза именно оттого,
# что топит худшие точки каждого ракурса в массе благополучных.
#
# Зависимость от густоты пробы измерена и монотонна: P95 чтения (а) на разреженной
# сцене равен 0.8057 / 0.8706 / 0.9076 / 0.9206 % при 25, 81, 289 и 625 точках
# габарита. Проба редка — значит, число ЗАНИЖЕНО, и в спецификацию оно идёт как
# нижняя оценка, а не как достигнутая точность.
# ---------------------------------------------------------------------------
MEASURED_FOOT = {
    SPARSE: Residual(median=0.0698, p95=0.6012, maximum=1.3603),
    RICH: Residual(median=0.0407, p95=0.0786, maximum=0.9483),
}
MEASURED_DISTANCE = {
    SPARSE: Residual(median=0.0329, p95=0.6328, maximum=1.0620),
    RICH: Residual(median=0.0149, p95=0.0597, maximum=0.1310),
}
MEASURED_POSITION = {
    SPARSE: Residual(median=0.0669, p95=0.8057, maximum=1.2995),
    RICH: Residual(median=0.0389, p95=0.0813, maximum=0.7835),
}

# Полоса воспроизведения замера, двусторонняя и одна на все три величины сводки.
# Прежние «вверх 1.4, вниз 0.3» ничем оправданы не были, и это ИЗМЕРЕНО, а не
# объявлено: разброса между прогонами НЕТ ВОВСЕ. Три прогона всей сетки в отдельных
# процессах — два обычных и один с однопоточными BLAS и OpenMP (`OMP_NUM_THREADS=1`,
# `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) — дали ПОБИТОВО совпадающие числа во
# всех 72 ракурсах обеих сцен. И детектор отрезков, и оценщик точек схода (seed=0,
# полный перебор пар гипотез при 63 и менее прямых) детерминированы. Полосы «на шум»
# здесь быть не может, потому что шума нет.
#
# Полоса в +-5 % допускает ровно одно: различие сборок OpenCV и BLAS, на которых
# замер не повторялся. Она заведомо уже измеренных эффектов, от которых охраняет.
# Удвоение допуска инлайера в `vanishing.py` — та самая мутация, что прежде роняла
# ровно один тест из семидесяти восьми, — сдвигает:
#
#     величина                    разреженная        обогащённая
#     медиана опорной точки       x1.056             x1.000
#     медиана расстояния          x1.116             x0.973
#     медиана положения           x1.096             x1.000
#     P95 положения               x1.043             x0.943
#     максимум опорной точки      x1.000             x0.084
#     максимум положения          x1.000             x0.107
#     принято ракурсов            34 -> 32           36 -> 36
#
# Отсюда видно и ограничение самой полосы, которое незачем скрывать: P95 разреженной
# сцены она ловит впритык (x1.043 против 1.05), а максимум разреженной сцены не
# ловит вовсе. Полоса этому параметру не охрана и охраной быть не может. Ловится он
# тремя другими способами: прямой двусторонней проверкой самого допуска в
# `tests/test_vanishing.py`, медианами обеих сцен и максимумами обогащённой, а также
# числом принятых ракурсов (34 -> 32 роняет `accepted`).
TOLERANCE_UP = 1.05
TOLERANCE_DOWN = 0.95

# Ракурсы, отвергнутые метрикой доверия задачи 7 на разреженной сцене. Список
# точный, а не «не более четырёх»: он и есть утверждение о том, ГДЕ разреженная
# сцена ломается, и его изменение обязано быть замечено, а не поглощено запасом.
# Отображение «ракурс -> подстрока замеренной причины». Не множество: причина есть
# часть утверждения. Подстроки различают два РАЗНЫХ отказа — развал вертикального
# пучка (три причины разом) и одну лишь неустойчивость направления (+-0.88°).
SPARSE_REFUSED = {
    (-1500.0, -2000.0, 9000.0): "направления не ортогональны",
    (4500.0, 2000.0, 12000.0): "определено неустойчиво",
}

# Сколько ракурсов обязано остаться в выборке. Не самостоятельное число, а
# дополнение к SPARSE_REFUSED. Без него `accepted` могла вернуть пустой список, а
# проверки — сообщить об успехе, не выполнив ни одного утверждения.
EXPECTED_ACCEPTED = {SPARSE: len(GRID) - len(SPARSE_REFUSED), RICH: len(GRID)}

# Сколько ракурсов разреженной сцены превышают бюджет п. 2.2. Величина устойчивее
# самого P95: выбрасывание одного ракурса меняет P95 на 30 %, а эту долю — на один
# ракурс из тридцати четырёх. Измерено 2 из 34 = 5.88 % (базовая сетка) и 7 из 139
# = 5.04 % (сгущённая, 165 ракурсов). Формулировка «P95 <= 0.8 %» разрешает ровно
# 5 %, поэтому обе сетки говорят одно и то же: бюджет не выдерживается.
SPARSE_OVER_BUDGET = 2

# Джекнайф P95 невязки положения: наименьшее и наибольшее значение при выбрасывании
# одного ракурса. Число это не проверяется, а ОБЪЯСНЯЕТ, почему у проверки
# `p95 >= 0.8` не может быть запаса: на разреженной сцене P95 определяется двумя
# худшими ракурсами из тридцати четырёх.
JACKKNIFE_P95 = {SPARSE: (0.5544, 0.8230), RICH: (0.0769, 0.0814)}

# Достижимый диапазон масштаба ректифицированного растра, измеренный по всем
# принятым ракурсам. Величины пропорциональны `mm_per_rect_unit`, то есть масштабу
# измерительного слоя: сдвиг масштаба на процент сдвигает на процент и их.
ATTAINABLE_LO_MM_PER_PX = {SPARSE: (0.7666, 1.3785), RICH: (0.7700, 1.3785)}
ATTAINABLE_HI_MM_PER_PX = {SPARSE: (10496.9, 22765.3), RICH: (10532.2, 22765.2)}

# Грубая охрана по каждому ракурсу в отдельности. Ровно та проверка, отсутствие
# которой в прежней редакции пропустило камеру в 27 км от фасада (это 225000 %).
# Тонкая граница держится распределением ниже, здесь — только порядок величины.
GROSS_LIMIT_PCT = 3.0

# Ректификация по существу. Измеренные максимумы: отклонение углов от прямого
# 0.646° (разреженная) и 0.247° (обогащённая); верхняя сторона, в калибровку НЕ
# входившая, отклоняется от 20000 мм на 1.70 % и 0.11 %; боковые от 15000 мм —
# на 0.64 % и 0.52 %.
MAX_ANGLE_DEVIATION_DEG = {SPARSE: 0.90, RICH: 0.35}
MAX_TOP_DEVIATION = {SPARSE: 0.024, RICH: 0.0016}
MAX_SIDE_DEVIATION = {SPARSE: 0.009, RICH: 0.0073}

# Согласие растра с гомографией. Измерено 6.7e-16 по сторонам и 4.3e-14 по углам:
# это одно и то же линейное отображение, записанное дважды, и расходиться они могут
# только на округлении. Допуск, заданный «с запасом» (скажем, 1e-6), пропустил бы
# ошибку в привязке растра к миллиметрам величиной в микрон на метр.
RASTER_AGREEMENT_REL = 1e-12
RASTER_AGREEMENT_DEG = 1e-10


@pytest.mark.parametrize("kind", list(SCENES))
@pytest.mark.parametrize("dx,dy,dist", GRID)
def test_pose_from_real_detector_matches_scene_truth(kind, dx, dy, dist):
    """Поза с каждого ракурса сверяется с истиной сцены, а не сама с собой.

    Проверка по ракурсу — грубая: она ловит разрушенное восстановление позы,
    тонкую границу держит распределение. Отвергнутый метрикой доверия ракурс
    пропускается с названной причиной: гомографии для него не существует.
    """
    result = chain(kind, dx, dy, dist)
    if result.needs_operator:
        pytest.skip(f"доверие {result.confidence:.3f}, причины: {'; '.join(result.reasons)}")

    assert result.half_turn_resolved is False, (
        "двузначность поворота на 180° двумя точками схода плоскости не разрешается; "
        "если признак стал переменным, поза больше не может выдаваться парой"
    )
    assert result.foot_pct_of_distance < GROSS_LIMIT_PCT, (
        f"опорная точка ушла на {result.foot_pct_of_distance:.3f} % расстояния")
    assert result.distance_pct_of_distance < GROSS_LIMIT_PCT, (
        f"расстояние до плоскости ушло на {result.distance_pct_of_distance:.3f} %")


@pytest.mark.parametrize("kind", list(SCENES))
def test_pose_residual_distribution_reproduces_the_measurement(kind):
    """Медиана, P95 и максимум невязки позы — против записанного замера.

    Числа приводятся порознь по двум сценам, потому что различаются кратно:
    P95 опорной точки 0.601 % против 0.079 %. Общая цифра по объединённой выборке
    описывала бы несуществующую смесь.
    """
    rows = accepted(kind)
    foot = summarise([r.foot_pct_of_distance for r in rows])
    distance = summarise([r.distance_pct_of_distance for r in rows])

    for name, got, want in (("опорная точка", foot, MEASURED_FOOT[kind]),
                            ("расстояние", distance, MEASURED_DISTANCE[kind])):
        assert got.median <= want.median * TOLERANCE_UP, (
            f"{kind}, {name}: медиана {got.median:.4f} % против замеренных {want.median} %")
        assert got.p95 <= want.p95 * TOLERANCE_UP, (
            f"{kind}, {name}: P95 {got.p95:.4f} % против замеренных {want.p95} %")
        assert got.maximum <= want.maximum * TOLERANCE_UP, (
            f"{kind}, {name}: максимум {got.maximum:.4f} % против замеренных {want.maximum} %")
        assert got.median >= want.median * TOLERANCE_DOWN, (
            f"{kind}, {name}: медиана {got.median:.4f} % МЕНЬШЕ замеренных "
            f"{want.median} % — замер устарел либо цепочка перестала быть сквозной")
        assert got.p95 >= want.p95 * TOLERANCE_DOWN, (
            f"{kind}, {name}: P95 {got.p95:.4f} % МЕНЬШЕ замеренных {want.p95} %")
        assert got.maximum >= want.maximum * TOLERANCE_DOWN, (
            f"{kind}, {name}: максимум {got.maximum:.4f} % МЕНЬШЕ замеренных "
            f"{want.maximum} % — на обогащённой сцене именно максимум и ловит "
            f"расширение допуска инлайера")


@pytest.mark.parametrize("kind", list(SCENES))
def test_absolute_position_residual_against_the_budget(kind):
    """Невязка абсолютного положения точек фасада — величина допуска п. 2.2.

    Считается по 29 контрольным точкам на ракурс как максимум отношения ошибки к
    расстоянию от начала координат фасада: именно так п. 6.4 выводит допуск
    («0.2-0.5 % на фасаде 25 м это 50-125 мм на дальнем краю»).

    Утверждение теста двустороннее и на разреженной сцене фиксирует ПРЕВЫШЕНИЕ
    бюджета: P95 равен 0.806 % при допуске 0.8 %, и это ещё без погрешности
    выделения самой измеряемой кромки. Одностороннее «меньше допуска» здесь было
    бы неправдой, а «больше нуля» — утверждением, которое не может не выполниться.
    """
    rows = accepted(kind)
    position = summarise([r.position_pct_of_lever for r in rows])
    want = MEASURED_POSITION[kind]

    for name, got, ref in (("медиана", position.median, want.median),
                           ("P95", position.p95, want.p95),
                           ("максимум", position.maximum, want.maximum)):
        assert ref * TOLERANCE_DOWN <= got <= ref * TOLERANCE_UP, (
            f"{kind}, положение, {name}: {got:.4f} % против замеренных {ref} %")

    budget_p95_pct = 0.8
    over = [r for r in rows if r.position_pct_of_lever > budget_p95_pct]
    if kind == SPARSE:
        # Доля превышающих ракурсов — величина УСТОЙЧИВЕЕ самого P95, и проверяется
        # она первой. Формулировка «P95 <= 0.8 %» разрешает 5 % таких ракурсов;
        # измерено 2 из 34 = 5.88 % на базовой сетке и 7 из 139 = 5.04 % на
        # сгущённой, то есть бюджет не выдерживается на обеих.
        assert len(over) == SPARSE_OVER_BUDGET, (
            f"ракурсов свыше {budget_p95_pct} % плеча: {len(over)} из {len(rows)}, "
            f"замерено {SPARSE_OVER_BUDGET}; это и есть утверждение о несоответствии "
            f"бюджету, и оно устойчивее самого P95"
        )
        assert position.p95 >= budget_p95_pct, (
            f"P95 невязки положения {position.p95:.4f} % опустился ниже допуска "
            f"{budget_p95_pct} % при измеренном запасе всего "
            f"{100.0 * (want.p95 / budget_p95_pct - 1.0):.1f} %. "
            f"Устарел ли вывод отчёта — по одному этому числу НЕ определить: "
            f"джекнайф измерен и показывает, что выбрасывание одного ракурса уводит "
            f"P95 на {JACKKNIFE_P95[SPARSE][0]:.4f}…{JACKKNIFE_P95[SPARSE][1]:.4f} %, "
            f"то есть смена сборки OpenCV способна опустить величину ниже допуска, "
            f"ничего не изменив по существу. Решает проверка выше: сколько ракурсов "
            f"превысили бюджет ({len(over)} из {len(rows)})"
        )
    else:
        assert position.p95 <= 0.25 * budget_p95_pct, (
            f"P95 невязки положения {position.p95:.4f} % на обогащённой сцене "
            f"перестал укладываться в четверть допуска {budget_p95_pct} %"
        )
        assert not over, (
            f"обогащённая сцена дала {len(over)} ракурсов свыше {budget_p95_pct} %")


def test_refusals_are_confined_to_the_sparse_scene():
    """Где именно конвейер отказывается строить плоскость.

    Отказ — не сбой, а контракт п. 4.2, и он привязан к сцене: на разреженной
    отвергаются два ракурса из тридцати шести, на обогащённой — ни одного.
    Без этой проверки статистика по «принятым» ракурсам могла бы улучшаться
    молчаливым ростом числа отказов.
    """
    refused = {kind: {view for view in GRID if chain(kind, *view).needs_operator}
               for kind in SCENES}

    assert refused[SPARSE] == set(SPARSE_REFUSED), (
        f"отвергнутые ракурсы разреженной сцены: {sorted(refused[SPARSE])}")
    assert refused[RICH] == set(), (
        f"обогащённая сцена не должна давать отказов, а дала {sorted(refused[RICH])}")

    # Прежние `assert result.reasons` и `assert result.confidence < 0.5` не могли не
    # выполниться: в `vanishing.py` доверие равно нулю ровно тогда, когда список
    # причин непуст, а структурный инвариант обязывает непустой список ниже порога.
    # Измерено: у обоих отвергнутых ракурсов доверие равно 0.000 РОВНО. Проверяется
    # поэтому то, что выполниться может и не всегда, — КАКАЯ причина названа. Два
    # ракурса отвергаются по РАЗНЫМ причинам, и это часть утверждения: одна беда
    # разреженной сцены — развал пучка, другая — неустойчивость направления.
    for view, marker in SPARSE_REFUSED.items():
        result = chain(SPARSE, *view)
        assert any(marker in reason for reason in result.reasons), (
            f"ракурс {view} отвергнут по причинам {list(result.reasons)}, "
            f"а замерена причина, содержащая {marker!r}")
        assert result.confidence == 0.0, (
            f"ракурс {view}: доверие {result.confidence!r}; непустой список причин "
            f"обязан давать ровно нуль, иначе отказ и его объяснение разошлись")


@pytest.mark.parametrize("kind", list(SCENES))
def test_rectified_facade_keeps_right_angles_and_true_proportions(kind):
    """Ректификация проверяется по существу, а не по факту непустоты растра.

    Прямые углы и стороны против ИСТИННЫХ габаритов 20000 x 15000 мм. Нижняя
    сторона в проверку не входит: на ней взят масштаб, и она равна 20000 мм
    тождественно. Верхняя сторона и обе боковые в калибровку не входили и потому
    проверкой являются.
    """
    for result in accepted(kind):
        assert len(result.angles_deg) == 4, (
            f"{kind}: у четырёхугольника фасада {len(result.angles_deg)} углов — "
            "цикл по углам не выполнил бы ни одного утверждения")
        for angle in result.angles_deg:
            assert abs(angle - 90.0) <= MAX_ANGLE_DEVIATION_DEG[kind], (
                f"{kind}: угол ректифицированного фасада {angle:.4f}°")
        assert abs(result.top_mm / FACADE_W_MM - 1.0) <= MAX_TOP_DEVIATION[kind], (
            f"{kind}: верхняя сторона {result.top_mm:.1f} мм против 20000 мм")
        for side in (result.left_mm, result.right_mm):
            assert abs(side / FACADE_H_MM - 1.0) <= MAX_SIDE_DEVIATION[kind], (
                f"{kind}: боковая сторона {side:.1f} мм против 15000 мм")


@pytest.mark.parametrize("kind", list(SCENES))
def test_rectified_raster_carries_the_same_millimetres(kind):
    """Растр `rectify` привязан к миллиметрам так же, как гомография.

    Проверяется не содержимое растра, а его конвенция: масштаб `mm_per_px` и
    начало `origin_rect_px`. Перепутанное начало или сбитый масштаб сдвигают все
    измерения разом и выглядят правдоподобно — поэтому допуск здесь на уровне
    округления, а не «с запасом».
    """
    for result in accepted(kind):
        assert abs(result.raster_top_mm / result.top_mm - 1.0) <= RASTER_AGREEMENT_REL
        assert abs(result.raster_left_mm / result.left_mm - 1.0) <= RASTER_AGREEMENT_REL
        # `zip` короткого с пустым молча даёт нуль итераций: без этой охраны цикл
        # ниже сообщал бы об успехе, не сравнив ни одного угла.
        assert len(result.raster_angles_deg) == len(result.angles_deg) == 4, (
            f"{kind}: углов у растра {len(result.raster_angles_deg)}, у гомографии "
            f"{len(result.angles_deg)} — zip сравнил бы не всё либо вовсе ничего")
        for from_raster, from_homography in zip(result.raster_angles_deg, result.angles_deg):
            assert abs(from_raster - from_homography) <= RASTER_AGREEMENT_DEG


@pytest.mark.parametrize("kind", list(SCENES))
def test_the_attainable_raster_scale_reproduces_the_measurement(kind):
    """Границы достижимого масштаба растра — против замера, а не «лишь бы внутри».

    Прежде здесь стояло `lo <= RECT_MM_PER_PX <= hi` внутри `chain`. Это утверждение
    верно, но почти пусто: запас снизу 7…13 крат, сверху 1050…2280 крат, то есть
    ошибку масштаба измерительного слоя меньше семикратной оно не различает вовсе.
    Само по себе оно к тому же не охрана: `rectify` сверяет `mm_per_px` с теми же
    границами и отказывает громче.

    Проверяется поэтому другое — сами границы. Они пропорциональны
    `mm_per_rect_unit`, то есть масштабу, взятому у оператора: сдвиг масштаба на
    один процент сдвигает на процент и их, и ЭТО полоса воспроизведения различает.
    Контрактное `lo <= 10 <= hi` остаётся, но подаётся тем, чем является:
    утверждением, что выбранный растр возможен, а не проверкой подбора масштаба.
    """
    rows = accepted(kind)
    lo = np.array([r.attainable_lo_mm_per_px for r in rows])
    hi = np.array([r.attainable_hi_mm_per_px for r in rows])

    assert np.all(lo <= RECT_MM_PER_PX) and np.all(RECT_MM_PER_PX <= hi), (
        f"{kind}: масштаб {RECT_MM_PER_PX} мм/px вне достижимого диапазона хотя бы "
        f"на одном ракурсе: lo до {lo.max():.4g}, hi от {hi.min():.4g}")

    want_lo, want_hi = ATTAINABLE_LO_MM_PER_PX[kind], ATTAINABLE_HI_MM_PER_PX[kind]
    for got, want, name in ((lo.min(), want_lo[0], "наименьшее lo"),
                            (lo.max(), want_lo[1], "наибольшее lo"),
                            (hi.min(), want_hi[0], "наименьшее hi"),
                            (hi.max(), want_hi[1], "наибольшее hi")):
        assert want * TOLERANCE_DOWN <= got <= want * TOLERANCE_UP, (
            f"{kind}, {name}: {got:.4f} против замеренных {want:.4f}; границы "
            f"пропорциональны масштабу измерительного слоя, и их уход означает "
            f"уход самого масштаба")


def test_chain_really_runs_through_the_detector():
    """Охрана от подмены реального детектора точными точками схода.

    На точных точках схода та же цепочка даёт невязку опорной точки порядка
    1e-14 % — двенадцать порядков ниже рабочей. Если когда-нибудь сюда подставят
    точные точки схода, все верхние границы выше будут выполнены с колоссальным
    запасом, и проверка перестанет быть сквозной, ничем этого не показав.
    """
    view = (0.0, 2000.0, 12000.0)        # худший ракурс разреженной сцены
    scene = SCENES[SPARSE](dx=view[0], dy=view[1], dist=view[2])

    vh, vv = exact_vps(scene)
    H = homography_from_vanishing_points(vh, vv, K, SIZE)
    rect = apply(H, scene.project(CORNERS))
    mm_per_unit = FACADE_W_MM / float(np.linalg.norm(rect[1] - rect[0]))
    camera, _ = camera_pose(H, vh, vv, K, mm_per_unit, tuple(rect[0]))
    truth = scene.camera_on_plane()
    exact_pct = 100.0 * float(np.hypot(camera.cx - truth.cx, camera.cy - truth.cy)) / truth.cz

    assert exact_pct < 1e-9, (
        f"точные точки схода дают {exact_pct:.3e} % — эталон сравнения испорчен")

    detected_pct = chain(SPARSE, *view).foot_pct_of_distance
    assert detected_pct > 1e6 * exact_pct, (
        f"невязка через детектор ({detected_pct:.4f} %) неотличима от невязки на "
        f"точных точках схода ({exact_pct:.3e} %): цепочка не сквозная"
    )
