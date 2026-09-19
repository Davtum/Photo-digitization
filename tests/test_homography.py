import re

import numpy as np
import pytest

from facade_digitizer.geometry.homography import (
    _ORIENTATION_MIN_RELATIVE,
    _rectified_x_grows_with_image_x,
    camera_pose,
    foot_point_in_rectified,
    homography_from_four_points,
    homography_from_vanishing_points,
    rectified_to_facade_mm,
)
from facade_digitizer.synth.scene import SyntheticScene
from tests.test_synth import SIZE, K, make_scene

CORNERS = np.array([[0.0, 0], [20000.0, 0], [20000.0, 15000.0], [0, 15000.0]])

# Причины отказа ручного варианта. Подстроки подобраны так, чтобы КАЖДАЯ подходила
# только к своему сообщению: `pytest.raises(ValueError)` без `match=` принял бы любой
# ValueError, в том числе будущий, пришедший совсем из другого места.
FOUR_POINT_UNDERDETERMINED = "четырёх точек недостаточно"
CALIBRATED_NEEDS_K = "для assume_calibrated нужны K и image_size"

# Причины отказа доопределения ориентации. Два разных вырождения, и подстроки
# обязаны их различать: «центр кадра на линии схода» и «градиент вырожден».
CENTRE_NOT_IN_FRONT = "центр кадра не лежит перед плоскостью фасада"
GRADIENT_DEGENERATE = "градиент ректифицированной X в центре кадра вырожден"
CENTRE_NOT_FINITE = "образ центра кадра неконечен"
FOOT_NOT_CONSISTENT = "образ точки схода нормали вырожден"
TOO_MANY_DISAMBIGUATIONS = "доопределение должно быть ровно одно"

REFUSAL_PATTERNS = (FOUR_POINT_UNDERDETERMINED, CALIBRATED_NEEDS_K,
                    CENTRE_NOT_IN_FRONT, GRADIENT_DEGENERATE,
                    CENTRE_NOT_FINITE, FOOT_NOT_CONSISTENT,
                    TOO_MANY_DISAMBIGUATIONS)

# Допуски для ТОЧНЫХ точек схода. Истинная невязка на этих ракурсах — единицы ULP
# (до 3.6e-12 мм по опорной точке) и 3e-16 относительно по расстоянию. Прежние
# `abs=1.0` и `rel=0.02` были шире истинной невязки на одиннадцать и четырнадцать
# порядков: систематический сдвиг `cz` в один процент — а это масштаб всего
# измерительного слоя — проходил сквозь них молча.
EXACT_FOOT_ABS_MM = 1e-9
EXACT_DISTANCE_REL = 1e-12

# Околоплоскостные ракурсы обусловлены хуже: при 0.001 мм от плоскости невязка
# опорной точки доходит до 5e-5 мм, расстояния — до 5.5e-9 относительно. Допуск
# всё равно на пять порядков уже однопроцентного сдвига.
NEAR_PLANE_FOOT_ABS_MM = 1e-3
NEAR_PLANE_DISTANCE_REL = 1e-7

# Почти рёберный ракурс: камера в 200 мм от плоскости и в 20 м вбок. Линия схода
# плоскости проходит в 36 px от главной точки — ближе, чем прежний фиксированный
# сдвиг пробы в 50 px.
GRAZING_VIEW = (-20000.0, 0.0, 200.0)


def exact_vps(sc):
    """Точки схода осей плоскости, вычисленные точно из позы сцены."""
    return K @ (sc.R_wc @ np.array([1.0, 0, 0])), K @ (sc.R_wc @ np.array([0, 1.0, 0]))


def apply(H, pts):
    p = np.column_stack([np.atleast_2d(pts), np.ones(len(np.atleast_2d(pts)))]) @ H.T
    return p[:, :2] / p[:, 2:3]


# Шаг центральной разности. Независимый измеритель наклона: формула кода здесь
# сознательно НЕ повторяется — тест, переписывающий правило из кода, подтверждает
# только то, что его удалось переписать, и переживает согласованную ошибку в нём.
NUMERIC_STEP_PX = 1.0


def numeric_x_slope(H, u, v, axis=0, step=NUMERIC_STEP_PX):
    """Численная производная ректифицированной X по координате кадра.

    Центральная разность с фиксированным шагом, без обращения к внутренностям
    `homography.py`: гомография применяется как чёрный ящик.
    """
    offset = np.zeros(3)
    offset[axis] = step

    def x_rect(point):
        p = H @ point
        return p[0] / p[2]

    base = np.array([u, v, 1.0])
    return (x_rect(base + offset) - x_rect(base - offset)) / (2.0 * step)


def test_rectification_restores_right_angles_and_ratio():
    sc = make_scene()
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(np.array([[0.0, 0], [1500.0, 0], [1500.0, 1500.0], [0, 1500.0]])))
    a, b = r[1] - r[0], r[3] - r[0]
    ang = np.degrees(np.arccos(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert ang == pytest.approx(90.0, abs=1e-3)
    assert np.linalg.norm(a) / np.linalg.norm(b) == pytest.approx(1.0, rel=1e-6)


def test_rectified_raster_keeps_image_convention():
    """Ректифицированный растр — изображение: X вправо, Y ВНИЗ."""
    sc = make_scene()
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    assert r[1, 0] > r[0, 0]
    assert r[3, 1] < r[0, 1]


def test_facade_millimetres_have_y_up():
    """После перевода в миллиметры ось Y направлена вверх. Глобальные ограничения."""
    sc = make_scene()
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    mm = rectified_to_facade_mm(r, tuple(r[0]), mmu)
    assert mm[1, 0] == pytest.approx(20000.0, abs=1.0)
    assert mm[3, 1] == pytest.approx(15000.0, abs=1.0)


@pytest.mark.parametrize("dx,dy,dist", [
    (2500.0, -1800.0, 12000.0), (-3000.0, 2200.0, 9000.0),
    (0.0, -4000.0, 15000.0), (4500.0, 0.0, 11000.0),
])
def test_camera_pose_recovered_against_known_truth(dx, dy, dist):
    """Поза сравнивается с истиной сцены, а не с самой собой."""
    sc = make_scene(dx, dy, dist)
    vh, vv = exact_vps(sc)
    H = homography_from_vanishing_points(vh, vv, K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])

    got, half_turn = camera_pose(H, vh, vv, K, mmu, tuple(r[0]))
    truth = sc.camera_on_plane()
    assert half_turn.resolved is False
    assert got.cx == pytest.approx(truth.cx, abs=EXACT_FOOT_ABS_MM)
    assert got.cy == pytest.approx(truth.cy, abs=EXACT_FOOT_ABS_MM)
    assert got.cz == pytest.approx(truth.cz, rel=EXACT_DISTANCE_REL)


def test_four_point_requires_disambiguation():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError, match=FOUR_POINT_UNDERDETERMINED) as excinfo:
        homography_from_four_points(pts)
    # Проверяется названная причина, а не факт исключения: подстрока соседнего
    # отказа подойти к этому сообщению не должна.
    assert not re.search(CALIBRATED_NEEDS_K, str(excinfo.value))


def test_four_point_with_sizes_produces_those_sizes():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    H = homography_from_four_points(pts, size_mm=(1460.0, 1900.0))
    r = apply(H, pts)
    assert np.linalg.norm(r[1] - r[0]) == pytest.approx(1460.0, rel=1e-6)
    assert np.linalg.norm(r[3] - r[0]) == pytest.approx(1900.0, rel=1e-6)


def test_four_point_with_aspect_ratio_produces_that_ratio():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    H = homography_from_four_points(pts, aspect_ratio=2.0)
    r = apply(H, pts)
    assert np.linalg.norm(r[1] - r[0]) / np.linalg.norm(r[3] - r[0]) == pytest.approx(2.0, rel=1e-6)


def test_four_point_calibrated_variant_needs_K():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError, match=CALIBRATED_NEEDS_K) as excinfo:
        homography_from_four_points(pts, assume_calibrated=True)
    assert not re.search(FOUR_POINT_UNDERDETERMINED, str(excinfo.value))


def test_refusal_patterns_are_literal_and_mutually_exclusive():
    """Охрана самих охран.

    Вертикальная черта в `match=` читается как альтернатива и делает проверку
    почти всегда успешной; любой другой метасимвол размывает подстроку так же.
    Обе подстроки обязаны быть литеральными и обязаны различать свои сообщения.
    """
    # `re.escape` здесь не годится: он экранирует и пробел, который вне класса
    # символов ничего не значит. Перечисляются те метасимволы, которые
    # действительно размывают подстроку.
    metacharacters = set(r"|()[]{}*+?^$\.")
    for pattern in REFUSAL_PATTERNS:
        assert "|" not in pattern
        assert not (set(pattern) & metacharacters)

    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    messages = {}
    for key, call in (("underdetermined", {}),
                      ("calibrated", {"assume_calibrated": True})):
        with pytest.raises(ValueError) as excinfo:
            homography_from_four_points(pts, **call)
        messages[key] = str(excinfo.value)
    assert messages["underdetermined"] != messages["calibrated"]
    assert re.search(FOUR_POINT_UNDERDETERMINED, messages["underdetermined"])
    assert not re.search(FOUR_POINT_UNDERDETERMINED, messages["calibrated"])
    assert re.search(CALIBRATED_NEEDS_K, messages["calibrated"])
    assert not re.search(CALIBRATED_NEEDS_K, messages["underdetermined"])


# --- Различители, добавленные по итогам мутационной проверки ---------------------
#
# Тесты брифа целиком построены на ТОЧНЫХ точках схода осей сцены. Для них
# Kinv @ v уже единичен и уже ортогонален второму направлению, поэтому и
# нормировка, и ортогонализация по Граму — Шмидту оказываются тождественными
# операциями: их снятие не меняет ни одного числа. Мутации 1 и 6 выживали.
# Три теста ниже подбирают вход так, чтобы каждая из этих операций делала работу.


def skewed_vertical_vp(sc, slant=0.3):
    """Точка схода наклонного направления плоскости: 0.3·X + Y.

    Такое направление лежит в плоскости фасада, поэтому НОРМАЛЬ (d1 x d2) от подмены
    не меняется — меняется только неортогональность пары. Ортогонализация обязана
    вернуть ровно ту же гомографию, что и точная вертикаль; её снятие даёт сдвиг.
    """
    return K @ (sc.R_wc @ np.array([slant, 1.0, 0.0]))


def test_skewed_in_plane_direction_is_orthogonalised_back_to_the_axis():
    """Замена вертикали на наклонное направление плоскости не меняет результат."""
    sc = make_scene()
    vh, vv = exact_vps(sc)
    image_pts = sc.project(CORNERS)

    reference = apply(homography_from_vanishing_points(vh, vv, K, SIZE), image_pts)
    skewed = apply(homography_from_vanishing_points(vh, skewed_vertical_vp(sc), K, SIZE), image_pts)

    assert np.all(np.isfinite(skewed))
    assert skewed == pytest.approx(reference, abs=1e-9)


def test_homography_ignores_the_arbitrary_scale_of_vanishing_points():
    """Точка схода однородна: v и 7.3·v — одна точка, гомография обязана совпасть.

    Без нормировки направлений на единицу масштабы входов протекают в строки
    гомографии и растягивают оси по-разному.
    """
    sc = make_scene()
    vh, vv = exact_vps(sc)
    image_pts = sc.project(CORNERS)

    reference = apply(homography_from_vanishing_points(vh, vv, K, SIZE), image_pts)
    scaled = apply(homography_from_vanishing_points(7.3 * vh, 0.017 * vv, K, SIZE), image_pts)

    assert np.all(np.isfinite(scaled))
    assert scaled == pytest.approx(reference, abs=1e-9)


def test_normal_vanishing_point_lands_in_the_rectified_origin():
    """Опорная точка восстанавливается по точке схода НОРМАЛИ, а не оси плоскости.

    Точка схода оси плоскости лежит на бесконечности ректифицированных координат:
    подстановка её вместо нормали не даёт опорной точки вовсе.
    """
    sc = make_scene()
    vh, vv = exact_vps(sc)
    H = homography_from_vanishing_points(vh, vv, K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])

    foot_rect = foot_point_in_rectified(H, vh, vv, K)
    assert np.all(np.isfinite(foot_rect))
    assert foot_rect == pytest.approx((0.0, 0.0), abs=1e-9)

    foot_mm = rectified_to_facade_mm([foot_rect], tuple(r[0]), mmu)[0]
    truth = sc.camera_on_plane()
    assert foot_mm[0] == pytest.approx(truth.cx, abs=1.0)
    assert foot_mm[1] == pytest.approx(truth.cy, abs=1.0)


def recover(sc, vh, vv):
    """Ректифицированные углы фасада и поза камеры — то, что задача выдаёт наружу."""
    H = homography_from_vanishing_points(vh, vv, K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    return r, camera_pose(H, vh, vv, K, mmu, tuple(r[0])).camera


@pytest.mark.parametrize("sh,sv", [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_sign_of_a_vanishing_point_changes_nothing(sh, sv):
    """v и -v — одна и та же точка схода: результат обязан совпасть во всех четырёх сочетаниях.

    Произвольность знака и есть источник всей путаницы с ориентацией осей: знак
    первой точки схода переворачивает не X, а Y, потому что третья строка
    гомографии есть d1 x d2. Доопределение обязано сводить все четыре входа к
    одной паре осей, а не к двум зеркальным.
    """
    sc = make_scene()
    vh, vv = exact_vps(sc)

    r_ref, pose_ref = recover(sc, vh, vv)
    r_flipped, pose_flipped = recover(sc, sh * vh, sv * vv)

    assert np.all(np.isfinite(r_flipped))
    assert r_flipped == pytest.approx(r_ref, abs=1e-9)
    assert pose_flipped.cx == pytest.approx(pose_ref.cx, abs=1e-6)
    assert pose_flipped.cy == pytest.approx(pose_ref.cy, abs=1e-6)
    assert pose_flipped.cz == pytest.approx(pose_ref.cz, rel=1e-12)


def test_orientation_survives_a_grazing_view_that_defeats_a_fixed_pixel_probe():
    """Почти рёберный ракурс: линия схода плоскости в 36 px от главной точки.

    Прежняя проба с фиксированным сдвигом в 50 px уходила ЗА неё, знаменатель
    гомографии там отрицателен, и ориентация выбиралась зеркальной. Симптом
    измерен: растр переворачивался по обеим осям, поза выходила (+10000, -7500)
    вместо (-10000, +7500), а прямоугольность и отношение сторон при этом
    оставались безупречны — зеркало ими не ловится.
    """
    sc = make_scene(*GRAZING_VIEW)
    vh, vv = exact_vps(sc)
    r, got = recover(sc, vh, vv)

    assert np.all(np.isfinite(r))
    assert r[1, 0] > r[0, 0]      # X вправо
    assert r[3, 1] < r[0, 1]      # Y вниз

    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=EXACT_FOOT_ABS_MM)
    assert got.cy == pytest.approx(truth.cy, abs=EXACT_FOOT_ABS_MM)
    assert got.cz == pytest.approx(truth.cz, rel=EXACT_DISTANCE_REL)


def test_grazing_view_really_is_the_hard_case():
    """Различитель выше имеет смысл, только если ракурс действительно трудный.

    Трудность выражена через саму геометрию, а не через константу прежней
    реализации: линия схода плоскости проходит ближе 40 px от главной точки.
    Конечная разность такого размаха неизбежно пересекла бы её; аналитическая
    производная, взятая в центре, пересечь её не может — и обязана остаться
    невырожденной. Без этой проверки различитель мог бы незаметно выродиться в
    рядовой ракурс после правки сцены или матрицы камеры.
    """
    sc = make_scene(*GRAZING_VIEW)
    vanishing_line = np.cross(*exact_vps(sc))          # прямая через обе точки схода
    principal = np.array([K[0, 2], K[1, 2], 1.0])
    distance_px = abs(vanishing_line @ principal) / np.hypot(*vanishing_line[:2])
    assert np.isfinite(distance_px)
    assert distance_px < 40.0

    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    w, h = SIZE
    p = H @ np.array([w / 2.0, h / 2.0, 1.0])
    assert np.all(np.isfinite(p))
    assert p[2] > 0                                    # центр кадра перед плоскостью

    # Производная считается ЧИСЛЕННО, независимо от формулы кода: согласованно
    # неверная формула переписанную проверку пережила бы.
    slope = numeric_x_slope(H, w / 2.0, h / 2.0)
    assert np.isfinite(slope)
    assert abs(slope) > 1e-3                           # далека от вырождения


def test_camera_lying_in_the_facade_plane_is_refused():
    """Камера ровно в плоскости фасада: центр кадра попадает НА линию схода.

    Знаменатель гомографии в центре обращается в нуль ТОЧНО, деление дало бы
    nan, а `nan > 0` ложно — то есть без охраны знак осей выбрался бы молча, и
    гомография вернулась бы как ни в чём не бывало. Отказ обязан быть явным.
    """
    sc = make_scene(-20000.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=CENTRE_NOT_IN_FRONT) as excinfo:
        homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    assert not re.search(GRADIENT_DEGENERATE, str(excinfo.value))


@pytest.mark.parametrize("dist_mm", [200.0, 5.0, 2.0, 0.1, 0.001, 1e-11])
def test_analytic_criterion_works_arbitrarily_close_to_the_plane(dist_mm):
    """Аналитический признак почти не имеет нижней границы по расстоянию.

    Лестница проб её имела на 2 мм: линия схода (около 0.36 px от главной точки)
    оказывалась ближе наименьшей пробы (0.42 px), и доопределение отказывало.
    Производная берётся в самом центре и такой границы не знает. Граница всё же
    есть, но лежит на одиннадцать порядков ниже: 1e-11 мм ещё работает, 1e-12 мм
    уже отвергается (соседний тест). Формулировка «отказ только при расстоянии
    ровно нуль» была бы шире измеренного.
    """
    sc = make_scene(-20000.0, 0.0, dist_mm)
    r, got = recover(sc, *exact_vps(sc))
    assert np.all(np.isfinite(r))
    assert r[1, 0] > r[0, 0]
    assert r[3, 1] < r[0, 1]
    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=NEAR_PLANE_FOOT_ABS_MM)
    assert got.cy == pytest.approx(truth.cy, abs=NEAR_PLANE_FOOT_ABS_MM)
    assert got.cz == pytest.approx(truth.cz, rel=NEAR_PLANE_DISTANCE_REL)


@pytest.mark.parametrize("dist_mm", [1e-12, 1e-13, 0.0])
def test_refusal_boundary_near_the_plane_is_where_it_was_measured(dist_mm):
    """Граница отказа названа измеренным числом, а не «ровно нулём».

    При таком расстоянии знаменатель гомографии в центре кадра перестаёт быть
    положительным, и доопределение отвергается явно. Соседний тест показывает,
    что 1e-11 мм ещё работает: граница закреплена с обеих сторон.
    """
    sc = make_scene(-20000.0, 0.0, dist_mm)
    with pytest.raises(ValueError, match=CENTRE_NOT_IN_FRONT):
        homography_from_vanishing_points(*exact_vps(sc), K, SIZE)


def test_degenerate_gradient_is_refused_by_name():
    """Охрана вырожденного градиента, проверенная напрямую.

    Построить такую `H` из точек схода нельзя: нулевая строка якобиана означала бы
    вырожденную гомографию, а `H` собирается из матрицы поворота. Поэтому охрана
    проверяется на заведомо порченой матрице — иначе она осталась бы кодом, до
    которого не доходит ни один тест, и снять её можно было бы незаметно.

    Матрица подобрана так, чтобы центр кадра был ПЕРЕД плоскостью (третья
    компонента положительна), то есть первая охрана не срабатывает, а обе
    производные ректифицированной X были нулевыми.
    """
    broken = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    p = broken @ np.array([SIZE[0] / 2.0, SIZE[1] / 2.0, 1.0])
    assert p[2] > 0     # первая охрана заведомо не срабатывает

    with pytest.raises(ValueError, match=GRADIENT_DEGENERATE) as excinfo:
        _rectified_x_grows_with_image_x(broken, SIZE)
    assert not re.search(CENTRE_NOT_IN_FRONT, str(excinfo.value))


def rolled_scene(roll_deg):
    """Сцена по умолчанию, камера дополнительно повёрнута вокруг оптической оси.

    `make_scene` крена не даёт: `look_at` всегда держит мировую вертикаль вдоль
    вертикали кадра. Крен добавляется домножением слева — оптическая ось и центр
    камеры остаются прежними, меняется только поворот кадра.
    """
    base = make_scene()
    t = np.radians(roll_deg)
    roll = np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])
    return SyntheticScene(20000.0, 15000.0, base.openings, base.camera_center,
                          roll @ base.R_wc, K, SIZE)


@pytest.mark.parametrize("roll_deg",
                         [0.0, 45.0, -45.0, 89.0, -89.0, -89.999, -89.9999, 90.0])
def test_orientation_is_correct_for_any_roll_within_the_rule(roll_deg):
    """Крен камеры в (-90, +90]: правило применимо, поза обязана совпасть с истиной.

    Крен ровно +90 градусов входит сюда не сам собой: производная по x кадра там
    обращается в нуль (ось X фасада идёт по вертикали кадра), и знак берётся
    запасным путём — по производной той же X, но по y кадра. Конечная разность
    на этом ракурсе даёт шаг порядка 1e-17, то есть шум округления, и вернула бы
    знак наугад.
    """
    sc = rolled_scene(roll_deg)
    vh = K @ (sc.R_wc @ np.array([1.0, 0, 0]))
    vv = K @ (sc.R_wc @ np.array([0, 1.0, 0]))
    r, got = recover(sc, vh, vv)

    assert np.all(np.isfinite(r))
    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=EXACT_FOOT_ABS_MM)
    assert got.cy == pytest.approx(truth.cy, abs=EXACT_FOOT_ABS_MM)
    assert got.cz == pytest.approx(truth.cz, rel=EXACT_DISTANCE_REL)


def test_quarter_turn_roll_is_exactly_where_the_fallback_takes_over():
    """Условие перехода на запасной путь названо числом, а не «примерно».

    При крене 90 градусов производная по x кадра теряется во взаимном вычитании
    полностью (относительная величина порядка 1e-16, при пороге 1e-12), а
    производная по y кадра остаётся полновесной. При крене 89 градусов основной
    путь ещё работает. Без этой проверки запасной путь мог бы незаметно стать
    основным — или, наоборот, перестать вызываться вовсе.
    """
    def numeric_slopes(roll_deg):
        """Оба наклона, измеренные численно — без формулы из кода."""
        sc = rolled_scene(roll_deg)
        vh = K @ (sc.R_wc @ np.array([1.0, 0, 0]))
        vv = K @ (sc.R_wc @ np.array([0, 1.0, 0]))
        H = homography_from_vanishing_points(vh, vv, K, SIZE)
        u, v = SIZE[0] / 2.0, SIZE[1] / 2.0
        by_u = numeric_x_slope(H, u, v, axis=0)
        by_v = numeric_x_slope(H, u, v, axis=1)
        assert np.isfinite(by_u) and np.isfinite(by_v)
        return by_u, by_v

    by_u_at_90, by_v_at_90 = numeric_slopes(90.0)
    assert abs(by_u_at_90) < 1e-15      # основной путь вырожден: чистый шум
    assert abs(by_v_at_90) > 1e-5       # запасной путь полновесен

    by_u_at_89, by_v_at_89 = numeric_slopes(89.0)
    assert abs(by_u_at_89) > 1e-7       # при 89 градусах основной путь ещё работает

    # Порог сверяется с КОНСТАНТОЙ МОДУЛЯ, а не с переписанным литералом.
    relative_at_90 = abs(by_u_at_90) / abs(by_v_at_90)
    relative_at_89 = abs(by_u_at_89) / abs(by_v_at_89)
    assert relative_at_90 < _ORIENTATION_MIN_RELATIVE
    assert relative_at_89 > _ORIENTATION_MIN_RELATIVE

    # Сверки с константой мало: она обязана быть НЕСУЩЕЙ. Крен -89.999 держит её
    # снизу — там относительная величина 2.3e-4, а основной и запасной пути дают
    # РАЗНЫЕ знаки, поэтому поднятие порога выше 2.3e-4 включило бы запасной путь
    # и перевернуло ответ. Сам ответ проверяется в тесте области применимости,
    # куда этот крен внесён; здесь закрепляется зазор.
    by_u_near, by_v_near = numeric_slopes(-89.999)
    relative_at_minus_89_999 = abs(by_u_near) / abs(by_v_near)
    assert _ORIENTATION_MIN_RELATIVE < relative_at_minus_89_999 < 1e-3


@pytest.mark.parametrize("roll_deg", [91.0, 135.0, 180.0, -90.0])
def test_roll_outside_the_rule_turns_the_result_by_half_a_circle(roll_deg):
    """ГРАНИЦА ПОСТАНОВКИ, а не дефект реализации. Закреплена намеренно.

    Из точек схода поворот на 180 градусов не определяется в принципе: `(vh, vv)`
    и `(-vh, -vv)` — одни и те же точки, а два физически допустимых варианта осей
    различаются ровно этим поворотом. Правило «X растёт вместе с x изображения»
    снимает неоднозначность, опираясь на то, что снимок примерно не перевёрнут.
    Вне промежутка (-90, +90] это предположение неверно, и результат оказывается
    повёрнут на 180 градусов вокруг начала фасада.

    Тест закрепляет ИЗМЕРЕННОЕ поведение на границе, чтобы всякая будущая правка
    доопределения была решением, а не случайностью. Чтобы получить верный ответ
    при перевёрнутом снимке, нужен внешний источник вертикали (EXIF, гравитация),
    которого у этой функции нет.
    """
    sc = rolled_scene(roll_deg)
    vh = K @ (sc.R_wc @ np.array([1.0, 0, 0]))
    vv = K @ (sc.R_wc @ np.array([0, 1.0, 0]))
    r, got = recover(sc, vh, vv)
    truth = sc.camera_on_plane()

    assert np.all(np.isfinite(r))
    # Поворот на 180 градусов вокруг начала фасада: обе координаты меняют знак.
    assert got.cx == pytest.approx(-truth.cx, abs=EXACT_FOOT_ABS_MM)
    assert got.cy == pytest.approx(-truth.cy, abs=EXACT_FOOT_ABS_MM)
    # Расстояние до плоскости поворотом не затрагивается и остаётся верным.
    assert got.cz == pytest.approx(truth.cz, rel=EXACT_DISTANCE_REL)


# --- Успешная ветка assume_calibrated и незакрытые охраны ------------------------


def test_four_point_calibrated_variant_reproduces_the_vanishing_point_result():
    """Успешная ветка `assume_calibrated`, а не только её отказ.

    Четыре угла фасада в кадре задают те же два пучка, что и точки схода, поэтому
    результат обязан совпасть с `homography_from_vanishing_points` на тех же осях.
    Прежде проверялся только отказ, и подмена `vh` на `vv` внутри ветки проходила
    незамеченной.
    """
    sc = make_scene()
    image_pts = sc.project(CORNERS)

    H = homography_from_four_points(image_pts, assume_calibrated=True,
                                    K=K, image_size=SIZE)
    r = apply(H, image_pts)

    # Прямые углы и отношение сторон фасада 20000 x 15000.
    a, b = r[1] - r[0], r[3] - r[0]
    ang = np.degrees(np.arccos(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert ang == pytest.approx(90.0, abs=1e-6)
    assert np.linalg.norm(a) / np.linalg.norm(b) == pytest.approx(20000.0 / 15000.0, rel=1e-9)

    # Ориентация растра та же, что у основного пути.
    assert r[1, 0] > r[0, 0]
    assert r[3, 1] < r[0, 1]

    # И та же поза. Это и есть различитель подмены vh на vv: при перестановке
    # пучков оси меняются местами и опорная точка уезжает.
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    got, _ = camera_pose(H, *exact_vps(sc), K, mmu, tuple(r[0]))
    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=1e-6)
    assert got.cy == pytest.approx(truth.cy, abs=1e-6)
    assert got.cz == pytest.approx(truth.cz, rel=1e-9)


@pytest.mark.parametrize("kwargs", [
    {"assume_calibrated": True, "K": None, "image_size": SIZE},
    {"assume_calibrated": True, "K": K, "image_size": None},
])
def test_calibrated_variant_needs_both_k_and_image_size(kwargs):
    """Нужны ОБА аргумента. Проверка `K is None or image_size is None`.

    С `and` вместо `or` каждый из этих двух вызовов прошёл бы охрану насквозь.
    """
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError, match=CALIBRATED_NEEDS_K):
        homography_from_four_points(pts, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"aspect_ratio": 2.0, "size_mm": (1460.0, 1900.0)},
    {"aspect_ratio": 2.0, "assume_calibrated": True, "K": K, "image_size": SIZE},
    {"size_mm": (1460.0, 1900.0), "assume_calibrated": True, "K": K, "image_size": SIZE},
])
def test_four_point_rejects_more_than_one_disambiguation(kwargs):
    """Противоречивый вход отвергается, а не толкуется.

    `given` прежде сравнивалась только с нулём, то есть использовалась по знаку:
    при двух доопределениях сразу одно молча отбрасывалось (ветка размеров берёт
    `size_mm` и на `aspect_ratio` не смотрит).
    """
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError, match=TOO_MANY_DISAMBIGUATIONS) as excinfo:
        homography_from_four_points(pts, **kwargs)
    assert not re.search(FOUR_POINT_UNDERDETERMINED, str(excinfo.value))


def test_zero_vanishing_point_is_refused_instead_of_yielding_nan():
    """Нулевая точка схода: нормировка делит на нуль и даёт nan по всей матрице.

    Путь не умозрительный — именно так выглядит вырожденная оценка. Без охраны
    конечности сравнение `p[2] <= 0` на nan ложно, и доопределение пошло бы
    дальше по мусору.
    """
    sc = make_scene()
    _, vv = exact_vps(sc)
    # Деление на нуль здесь — предмет проверки, а не неожиданность.
    with np.errstate(all="ignore"),             pytest.raises(ValueError, match=CENTRE_NOT_FINITE) as excinfo:
        homography_from_vanishing_points(np.zeros(3), vv, K, SIZE)
    assert not re.search(CENTRE_NOT_IN_FRONT, str(excinfo.value))


def test_non_finite_derivative_does_not_decide_the_sign():
    """Охрана конечности ВНУТРИ перебора направлений.

    Матрица подобрана так, чтобы образ центра кадра остался конечным и лежал
    перед плоскостью, а числитель производной обратился в nan через inf - inf.
    Без охраны `abs(nan) <= порог` ложно, и функция вернула бы `nan > 0`, то есть
    знак наугад; с охраной — переходит к следующему направлению и отказывает.
    """
    big = 1e200
    broken = np.array([[big, 0.0, 1.0],
                       [0.0, 0.0, 1.0],
                       [big, 0.0, 1.0]])
    with np.errstate(all="ignore"):          # переполнение — предмет проверки
        centre = broken @ np.array([SIZE[0] / 2.0, SIZE[1] / 2.0, 1.0])
        derivative = broken @ np.array([1.0, 0.0, 0.0])
        assert np.all(np.isfinite(centre)) and centre[2] > 0
        assert np.all(np.isfinite(derivative))
        assert not np.isfinite(derivative[0] * centre[2] - centre[0] * derivative[2])

        with pytest.raises(ValueError, match=GRADIENT_DEGENERATE):
            _rectified_x_grows_with_image_x(broken, SIZE)


@pytest.mark.parametrize("bad_H", [
    np.zeros((3, 3)),
    np.full((3, 3), np.nan),
])
def test_foot_point_refuses_a_homography_that_does_not_match(bad_H):
    """Охрана в `foot_point_in_rectified`, заменившая мёртвую.

    Прежняя `abs(p[2]) < 1e-12` с сообщением «фасад строго фронтален» не
    срабатывала никогда: для согласованной гомографии третья компонента
    тождественно равна +-1, а фронтальность её не вырождает. На nan прежняя
    охрана вдобавок молчала и возвращала (nan, nan).
    """
    sc = make_scene()
    vh, vv = exact_vps(sc)
    with np.errstate(all="ignore"),             pytest.raises(ValueError, match=FOOT_NOT_CONSISTENT):
        foot_point_in_rectified(bad_H, vh, vv, K)


def test_foot_point_third_component_is_unit_for_a_matching_homography():
    """Почему прежняя охрана была мертва — закреплено измерением.

    Для согласованной гомографии образ точки схода нормали имеет третью
    компоненту ровно +-1 на любом ракурсе, включая строго фронтальный, который
    прежнее сообщение называло причиной вырождения.
    """
    for dx, dy, dist in [(0.0, 0.0, 12000.0), (3000.0, -2000.0, 12000.0),
                         (-20000.0, 0.0, 200.0)]:
        sc = make_scene(dx, dy, dist)
        vh, vv = exact_vps(sc)
        H = homography_from_vanishing_points(vh, vv, K, SIZE)
        d1, d2 = np.linalg.inv(K) @ vh, np.linalg.inv(K) @ vv
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / np.linalg.norm(d2)
        third = (H @ (K @ np.cross(d1, d2)))[2]
        assert abs(third) == pytest.approx(1.0, rel=1e-12)
