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
    """Что цепочка выдала на одном ракурсе. Все невязки — в процентах расстояния."""

    needs_operator: bool
    reasons: tuple
    confidence: float
    half_turn_resolved: bool = False
    foot_pct: float = float("nan")
    distance_pct: float = float("nan")
    angles_deg: tuple = ()
    top_mm: float = float("nan")
    left_mm: float = float("nan")
    right_mm: float = float("nan")
    raster_angles_deg: tuple = ()
    raster_top_mm: float = float("nan")
    raster_left_mm: float = float("nan")
    position_rel_max_pct: float = float("nan")
    position_abs_max_mm: float = float("nan")


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

    lo, hi = attainable_mm_per_px(image, plane.H, mm_per_unit)
    assert lo <= RECT_MM_PER_PX <= hi, (
        f"масштаб растра вне достижимого диапазона [{lo:.4g}, {hi:.4g}]")
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
        foot_pct=100.0 * float(np.hypot(camera.cx - truth.cx, camera.cy - truth.cy)) / truth.cz,
        distance_pct=100.0 * abs(camera.cz - truth.cz) / truth.cz,
        angles_deg=_corner_angles_deg(rect_corners),
        top_mm=float(np.linalg.norm(rect_corners[2] - rect_corners[3])) * mm_per_unit,
        left_mm=float(np.linalg.norm(rect_corners[3] - rect_corners[0])) * mm_per_unit,
        right_mm=float(np.linalg.norm(rect_corners[2] - rect_corners[1])) * mm_per_unit,
        raster_angles_deg=_corner_angles_deg(raster_mm),
        raster_top_mm=float(np.linalg.norm(raster_mm[2] - raster_mm[3])),
        raster_left_mm=float(np.linalg.norm(raster_mm[3] - raster_mm[0])),
        position_rel_max_pct=float((100.0 * deviation[far] / lever[far]).max()),
        position_abs_max_mm=float(deviation.max()),
    )
    return _CACHE[key]


def accepted(kind):
    """Ракурсы, принятые метрикой доверия. Отвергнутые в статистику не входят.

    Статистика по отвергнутым описывала бы поведение на данных, которые конвейер не
    принимает: при низком доверии гомография не строится вовсе (контракт п. 4.2).
    """
    return [chain(kind, *view) for view in GRID if not chain(kind, *view).needs_operator]


@dataclass(frozen=True)
class Residual:
    """Измеренные медиана, P95 и максимум одной невязки, в процентах расстояния."""

    median: float
    p95: float
    maximum: float


def summarise(values):
    v = np.asarray(values, dtype=float)
    return Residual(float(np.median(v)), float(np.percentile(v, 95)), float(v.max()))


# ---------------------------------------------------------------------------
# Измеренные значения. Сняты на сетке GRID, OpenCV 4.14.0, NumPy 2.4.6.
# Оценщик точек схода детерминирован (seed=0), детектор отрезков тоже, поэтому
# величины воспроизводимы на той же сборке.
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

# Допуск на воспроизведение замера. Он НЕ запас «на всякий случай»: медиана невязки
# расстояния на обогащённой сцене равна 0.0149 %, и систематический сдвиг масштаба
# измерительного слоя в один процент поднял бы её в семьдесят раз. Множитель 1.4
# оставлен под различия сборок OpenCV и BLAS, не больше.
TOLERANCE_UP = 1.4

# Нижняя граница медианы. Без неё тест переживает подмену реального детектора
# точными точками схода: на них невязка падает до 1.6e-14 %, то есть верхняя
# граница выполняется с запасом в двенадцать порядков, и «сквозная» проверка
# молча перестаёт быть сквозной. Множитель намеренно свободный: падение теста
# означает не поломку, а необходимость ПЕРЕМЕРИТЬ и переписать числа отчёта —
# ровно то, чего требует история этого проекта.
TOLERANCE_DOWN = 0.3

# Ракурсы, отвергнутые метрикой доверия задачи 7 на разреженной сцене. Список
# точный, а не «не более четырёх»: он и есть утверждение о том, ГДЕ разреженная
# сцена ломается, и его изменение обязано быть замечено, а не поглощено запасом.
SPARSE_REFUSED = {(-1500.0, -2000.0, 9000.0), (4500.0, 2000.0, 12000.0)}

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
    assert result.foot_pct < GROSS_LIMIT_PCT, (
        f"опорная точка ушла на {result.foot_pct:.3f} % расстояния")
    assert result.distance_pct < GROSS_LIMIT_PCT, (
        f"расстояние до плоскости ушло на {result.distance_pct:.3f} %")


@pytest.mark.parametrize("kind", list(SCENES))
def test_pose_residual_distribution_reproduces_the_measurement(kind):
    """Медиана, P95 и максимум невязки позы — против записанного замера.

    Числа приводятся порознь по двум сценам, потому что различаются кратно:
    P95 опорной точки 0.601 % против 0.079 %. Общая цифра по объединённой выборке
    описывала бы несуществующую смесь.
    """
    rows = accepted(kind)
    foot = summarise([r.foot_pct for r in rows])
    distance = summarise([r.distance_pct for r in rows])

    for name, got, want in (("опорная точка", foot, MEASURED_FOOT[kind]),
                            ("расстояние", distance, MEASURED_DISTANCE[kind])):
        assert got.median <= want.median * TOLERANCE_UP, (
            f"{kind}, {name}: медиана {got.median:.4f} % против замеренных {want.median} %")
        assert got.p95 <= want.p95 * TOLERANCE_UP, (
            f"{kind}, {name}: P95 {got.p95:.4f} % против замеренных {want.p95} %")
        assert got.maximum <= want.maximum * TOLERANCE_UP, (
            f"{kind}, {name}: максимум {got.maximum:.4f} % против замеренных {want.maximum} %")
        assert got.median >= want.median * TOLERANCE_DOWN, (
            f"{kind}, {name}: медиана {got.median:.4f} % много МЕНЬШЕ замеренных "
            f"{want.median} % — замер устарел либо цепочка перестала быть сквозной")


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
    position = summarise([r.position_rel_max_pct for r in rows])
    want = MEASURED_POSITION[kind]

    assert position.median <= want.median * TOLERANCE_UP
    assert position.p95 <= want.p95 * TOLERANCE_UP
    assert position.maximum <= want.maximum * TOLERANCE_UP
    assert position.median >= want.median * TOLERANCE_DOWN

    budget_p95_pct = 0.8
    if kind == SPARSE:
        assert position.p95 >= budget_p95_pct, (
            f"P95 невязки положения {position.p95:.4f} % опустился ниже допуска "
            f"{budget_p95_pct} %: вывод отчёта о несоответствии бюджету устарел"
        )
    else:
        assert position.p95 <= 0.25 * budget_p95_pct, (
            f"P95 невязки положения {position.p95:.4f} % на обогащённой сцене "
            f"перестал укладываться в четверть допуска {budget_p95_pct} %"
        )


def test_refusals_are_confined_to_the_sparse_scene():
    """Где именно конвейер отказывается строить плоскость.

    Отказ — не сбой, а контракт п. 4.2, и он привязан к сцене: на разреженной
    отвергаются два ракурса из тридцати шести, на обогащённой — ни одного.
    Без этой проверки статистика по «принятым» ракурсам могла бы улучшаться
    молчаливым ростом числа отказов.
    """
    refused = {kind: {view for view in GRID if chain(kind, *view).needs_operator}
               for kind in SCENES}

    assert refused[SPARSE] == SPARSE_REFUSED, (
        f"отвергнутые ракурсы разреженной сцены: {sorted(refused[SPARSE])}")
    assert refused[RICH] == set(), (
        f"обогащённая сцена не должна давать отказов, а дала {sorted(refused[RICH])}")

    for view in SPARSE_REFUSED:
        result = chain(SPARSE, *view)
        assert result.reasons, "отказ обязан называть причину"
        assert result.confidence < 0.5


@pytest.mark.parametrize("kind", list(SCENES))
def test_rectified_facade_keeps_right_angles_and_true_proportions(kind):
    """Ректификация проверяется по существу, а не по факту непустоты растра.

    Прямые углы и стороны против ИСТИННЫХ габаритов 20000 x 15000 мм. Нижняя
    сторона в проверку не входит: на ней взят масштаб, и она равна 20000 мм
    тождественно. Верхняя сторона и обе боковые в калибровку не входили и потому
    проверкой являются.
    """
    for result in accepted(kind):
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
        for from_raster, from_homography in zip(result.raster_angles_deg, result.angles_deg):
            assert abs(from_raster - from_homography) <= RASTER_AGREEMENT_DEG


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

    detected_pct = chain(SPARSE, *view).foot_pct
    assert detected_pct > 1e6 * exact_pct, (
        f"невязка через детектор ({detected_pct:.4f} %) неотличима от невязки на "
        f"точных точках схода ({exact_pct:.3e} %): цепочка не сквозная"
    )
