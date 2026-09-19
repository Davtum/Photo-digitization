from itertools import pairwise

import numpy as np
import pytest

from facade_digitizer.geometry.angles import (
    angle_map,
    local_gsd,
    local_gsd_field,
    usable_mask,
)
from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import homography_from_vanishing_points
from tests.test_homography import CORNERS, GRAZING_VIEW, apply, exact_vps
from tests.test_synth import SIZE, K, make_scene

# Прямое измерение истины: веер направлений по фасаду и центральная разность.
# Формула кода здесь СОЗНАТЕЛЬНО не повторяется — тест, переписывающий правило из
# кода, подтверждает только то, что его удалось переписать.
FAN_DIRECTIONS = 720
FAN_STEP_MM = 0.05

# Допуск сверки с истиной при shape=(1200, 1200). Измерено на трёх точках сверки:
# наибольшее отклонение узлового значения от веерного — 2.876e-04, и вносит его
# почти целиком округление до узла сетки, а не формула (собственная невязка
# формулы при точном попадании в точку — 1.5e-07). Запас к измеренному — 3.5 раза.
# Прежние границы брифа (0.98 и 1.15) были шире истинной невязки в 70 и 500 раз:
# сквозь них молча прошло бы систематическое завышение на 5-10 процентов, то есть
# тест не отличал бы верную формулу от завышающей.
GSD_TRUTH_REL = 1e-3

# Тот же допуск при shape=(64, 64): измерено 7.713e-03, запас к измеренному — 2.6.
GSD_TRUTH_REL_COARSE = 2e-2

# Доля узлов за линией схода на почти рёберном ракурсе. Измерено: 0.492 при
# shape=(256, 256). Допуск широкий намеренно — число зависит от густоты сетки,
# закрепляется здесь сам факт «около половины кадра», а не третий знак.
GRAZING_BEHIND_FRACTION = 0.49

CENTRE_ON_VANISHING_LINE = "знаменатель гомографии в центре кадра вырожден"


def rectifying_homography(sc):
    return homography_from_vanishing_points(*exact_vps(sc), K, SIZE)


def mm_per_rect_unit(sc, H):
    r = apply(H, sc.project(CORNERS))
    return 20000.0 / np.linalg.norm(r[1] - r[0])


def worst_gsd_measured(sc, fx, fy):
    """Худшее разрешение в точке фасада, измеренное ПРЯМО, без формулы кода.

    Две близкие точки фасада проецируются полной моделью камеры, расстояние между
    их образами меряется в пикселях. Веер из `FAN_DIRECTIONS` направлений вместо
    двух осей: утверждение «худшее по ЛЮБОМУ направлению» двумя осями не
    проверяется — измерено, что худшее направление вообще не осевое, и по двум
    осям оно занижается до 8 процентов.
    """
    angs = np.linspace(0.0, np.pi, FAN_DIRECTIONS, endpoint=False)
    plus = sc.project(np.column_stack([fx + FAN_STEP_MM * np.cos(angs),
                                       fy + FAN_STEP_MM * np.sin(angs)]))
    minus = sc.project(np.column_stack([fx - FAN_STEP_MM * np.cos(angs),
                                        fy - FAN_STEP_MM * np.sin(angs)]))
    return float((2.0 * FAN_STEP_MM / np.linalg.norm(plus - minus, axis=1)).max())


def node_at(grid, sc, fx, fy):
    a = sc.project(np.array([[fx, fy]]))[0]
    rows, cols = grid.shape
    col = np.clip(round(float(a[0]) / (SIZE[0] - 1) * (cols - 1)), 0, cols - 1)
    row = np.clip(round(float(a[1]) / (SIZE[1] - 1) * (rows - 1)), 0, rows - 1)
    return float(grid[row, col])


def test_angle_is_zero_at_foot_and_grows_outward():
    cam = CameraOnPlane(cx=10000.0, cy=7500.0, cz=10000.0)
    am = angle_map(cam, (0.0, 0.0, 20000.0, 15000.0), (101, 101))
    assert am.theta_full.min() == pytest.approx(0.0, abs=0.5)
    assert am.theta_full[0, 0] > am.theta_full[50, 50]


def test_components_have_expected_signs():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    am = angle_map(cam, (-5000.0, -5000.0, 5000.0, 5000.0), (3, 3))
    assert am.theta_x[1, 0] < 0
    assert am.theta_x[1, 2] > 0


def test_usable_mask_excludes_steep_angles():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    am = angle_map(cam, (0.0, 0.0, 20000.0, 15000.0), (51, 51))
    m = usable_mask(am, 30.0)
    assert m.any() and not m.all()


def test_local_gsd_majorises_every_direction():
    """Величина обязана мажорировать разрешение по ЛЮБОМУ направлению.

    Сверка с истиной: спроецировать близкие точки фасада и померить пиксели.
    """
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=12000.0)
    H = rectifying_homography(sc)
    mmu = mm_per_rect_unit(sc, H)

    step = 20.0
    grid = local_gsd(H, mmu, SIZE, shape=(1200, 1200))
    for fx, fy in [(4000.0, 4000.0), (14000.0, 10000.0), (2000.0, 12000.0)]:
        a = sc.project(np.array([[fx, fy]]))[0]
        along_x = step / np.linalg.norm(sc.project(np.array([[fx + step, fy]]))[0] - a)
        along_y = step / np.linalg.norm(sc.project(np.array([[fx, fy + step]]))[0] - a)
        got = node_at(grid, sc, fx, fy)
        # Мажорирование по осям — слабое утверждение, оставлено как читаемое.
        assert got >= max(along_x, along_y) * 0.98
        # Сверка с истиной по всему вееру — сильное, и допуск по измеренной невязке.
        assert got == pytest.approx(worst_gsd_measured(sc, fx, fy),
                                    rel=GSD_TRUTH_REL, abs=1e-12)


def test_local_gsd_majorises_every_direction_on_default_grid():
    """То же на умолчательной сетке: при shape=(64, 64) невязка втрое шире."""
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = rectifying_homography(sc)
    grid = local_gsd(H, mm_per_rect_unit(sc, H), SIZE)
    for fx, fy in [(4000.0, 4000.0), (14000.0, 10000.0), (2000.0, 12000.0)]:
        assert node_at(grid, sc, fx, fy) == pytest.approx(
            worst_gsd_measured(sc, fx, fy), rel=GSD_TRUTH_REL_COARSE, abs=1e-12)


def test_worst_direction_exceeds_geometric_mean():
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = rectifying_homography(sc)
    mmu = mm_per_rect_unit(sc, H)
    worst = local_gsd(H, mmu, SIZE)
    mean = local_gsd(H, mmu, SIZE, worst_direction=False)
    assert np.all(worst >= mean - 1e-9)
    assert worst.max() > mean.max()


def test_local_gsd_varies_across_tilted_frame():
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = rectifying_homography(sc)
    g = local_gsd(H, mm_per_rect_unit(sc, H), SIZE)
    assert g.max() / g.min() > 1.2


def test_working_view_has_no_nodes_behind_vanishing_line():
    """На рабочем ракурсе охрана знака не срабатывает и ничего не портит."""
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = rectifying_homography(sc)
    field = local_gsd_field(H, mm_per_rect_unit(sc, H), SIZE, shape=(256, 256))
    assert field.behind_vanishing_line == pytest.approx(0.0, abs=1e-12)
    assert not np.isnan(field.gsd).any()
    assert np.isfinite(field.gsd.max())


def test_nodes_behind_vanishing_line_are_excluded_and_counted():
    """Почти рёберный ракурс: половина кадра лежит ЗА линией схода.

    Ракурс тот самый, на котором дыра и была найдена: камера в 200 мм от плоскости
    и в 20 м вбок. Без охраны знака узлы за линией схода дают конечные значения с
    разбросом в 1.5e+07 раз и молча попадают в максимум наравне с видимыми, хотя
    соответствуют точкам, камерой не видимым вовсе.
    """
    sc = make_scene(*GRAZING_VIEW)
    H = rectifying_homography(sc)
    field = local_gsd_field(H, 1.0, SIZE, shape=(256, 256))

    assert field.behind_vanishing_line == pytest.approx(GRAZING_BEHIND_FRACTION, abs=0.05)
    assert np.isnan(field.gsd).any()
    # Доля исключённых узлов согласована с самим полем, а не объявлена отдельно.
    assert np.isnan(field.gsd).mean() == pytest.approx(field.behind_vanishing_line,
                                                       abs=1e-12)
    # Неисправность обязана сообщить о себе: обычный max даёт nan, а не число.
    assert np.isnan(field.gsd.max())
    assert np.isfinite(np.nanmax(field.gsd))


def test_local_gsd_refuses_when_vanishing_line_crosses_frame_centre():
    """Опорный знак взять неоткуда — отказ, а не выбор знака наугад."""
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = rectifying_homography(sc).copy()
    H[2, 2] -= H[2, 0] * (SIZE[0] / 2.0) + H[2, 1] * (SIZE[1] / 2.0) + H[2, 2]
    with pytest.raises(ValueError, match=CENTRE_ON_VANISHING_LINE):
        local_gsd_field(H, 1.0, SIZE, shape=(32, 32))


def test_grid_maximum_converges_when_vanishing_line_outside_frame():
    """Максимум по сетке есть оценка по выборке — но здесь она сходится.

    `w` линеен по координатам кадра, поэтому его минимум лежит в УГЛУ
    прямоугольника, а углы входят в любую сетку `np.linspace(0, n-1, cols)`.
    Измерено: от 16 до 2048 узлов максимум совпадает до последнего знака.
    """
    for kw in ({"dx": 3000.0, "dy": -2000.0, "dist": 12000.0},
               {"dx": 5000.0, "dy": -3500.0, "dist": 9000.0}):
        sc = make_scene(**kw)
        H = rectifying_homography(sc)
        mmu = mm_per_rect_unit(sc, H)
        maxima = [float(np.nanmax(local_gsd(H, mmu, SIZE, shape=(n, n))))
                  for n in (16, 64, 256, 1024)]
        for coarse, fine in pairwise(maxima):
            assert fine == pytest.approx(coarse, rel=1e-9, abs=1e-12)


def test_grid_maximum_does_not_converge_when_vanishing_line_crosses_frame():
    """Закрепляет ТЕКУЩЕЕ поведение и причину, по которой оно неизбежно.

    У самой линии схода точка фасада бесконечно далека, поэтому истинная верхняя
    грань разрешения на видимой части кадра БЕСКОНЕЧНА. Сгущение сетки максимум не
    стабилизирует: он растёт на порядки и немонотонно — узел то подходит к линии
    схода ближе, то дальше. Отсюда вывод для спецификации: худшее разрешение на
    таком ракурсе берётся по пригодной области (`usable_mask`), а не по всему кадру.

    Тест характеризующий. Если в будущем максимум станут брать по маске, его надо
    обновить сознательно, а не ждать молчаливого прохождения.
    """
    sc = make_scene(*GRAZING_VIEW)
    H = rectifying_homography(sc)
    maxima = [float(np.nanmax(local_gsd(H, 1.0, SIZE, shape=(n, n))))
              for n in (64, 256, 512)]
    assert max(maxima) / min(maxima) > 100.0
