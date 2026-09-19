import re

import numpy as np
import pytest

from facade_digitizer.geometry.homography import (
    camera_pose,
    foot_point_in_rectified,
    homography_from_four_points,
    homography_from_vanishing_points,
    rectified_to_facade_mm,
)
from tests.test_synth import SIZE, K, make_scene

CORNERS = np.array([[0.0, 0], [20000.0, 0], [20000.0, 15000.0], [0, 15000.0]])

# Причины отказа ручного варианта. Подстроки подобраны так, чтобы КАЖДАЯ подходила
# только к своему сообщению: `pytest.raises(ValueError)` без `match=` принял бы любой
# ValueError, в том числе будущий, пришедший совсем из другого места.
FOUR_POINT_UNDERDETERMINED = "четырёх точек недостаточно"
CALIBRATED_NEEDS_K = "для assume_calibrated нужны K и image_size"

# Причины отказа доопределения ориентации. Два разных вырождения, и подстроки
# обязаны их различать: «центр кадра на линии схода» и «линия схода ближе
# наименьшей пробы».
CENTRE_NOT_IN_FRONT = "центр кадра не лежит перед плоскостью фасада"
NO_PROBE_IN_FRONT = "ни одна проба ориентации не осталась перед плоскостью фасада"

REFUSAL_PATTERNS = (FOUR_POINT_UNDERDETERMINED, CALIBRATED_NEEDS_K,
                    CENTRE_NOT_IN_FRONT, NO_PROBE_IN_FRONT)

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

    got = camera_pose(H, vh, vv, K, mmu, tuple(r[0]))
    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=1.0)
    assert got.cy == pytest.approx(truth.cy, abs=1.0)
    assert got.cz == pytest.approx(truth.cz, rel=0.02)


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
    return r, camera_pose(H, vh, vv, K, mmu, tuple(r[0]))


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
    assert got.cx == pytest.approx(truth.cx, abs=1.0)
    assert got.cy == pytest.approx(truth.cy, abs=1.0)
    assert got.cz == pytest.approx(truth.cz, rel=0.02)


def test_grazing_view_really_is_the_hard_case():
    """Различитель выше имеет смысл, только если ракурс действительно трудный.

    Проверяется то, ради чего он взят: сдвиг пробы в 50 px от центра кадра на
    этом ракурсе даёт ОТРИЦАТЕЛЬНЫЙ знаменатель, то есть точку за линией схода.
    Без этой проверки тест-различитель мог бы незаметно выродиться в ещё один
    рядовой ракурс после любой правки сцены или матрицы камеры.
    """
    sc = make_scene(*GRAZING_VIEW)
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    w, h = SIZE
    centre = H @ np.array([w / 2.0, h / 2.0, 1.0])
    shifted = H @ np.array([w / 2.0 + 50.0, h / 2.0, 1.0])

    assert np.all(np.isfinite(centre)) and np.all(np.isfinite(shifted))
    assert centre[2] > 0      # центр кадра — перед плоскостью
    assert shifted[2] < 0     # сдвиг на 50 px — уже за линией схода


def test_camera_lying_in_the_facade_plane_is_refused():
    """Камера ровно в плоскости фасада: центр кадра попадает НА линию схода.

    Знаменатель гомографии в центре обращается в нуль, деление даёт nan, и
    сравнение `nan > 0` ложно — то есть без охраны знак осей выбрался бы молча,
    и гомография вернулась бы как ни в чём не бывало. Отказ обязан быть явным.
    """
    sc = make_scene(-20000.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=CENTRE_NOT_IN_FRONT) as excinfo:
        homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    assert not re.search(NO_PROBE_IN_FRONT, str(excinfo.value))


def test_view_too_close_to_edge_on_is_refused_by_name():
    """Камера в 2 мм от плоскости и в 20 м вбок: линия схода ближе наименьшей пробы.

    Центр кадра ещё перед плоскостью, но ни одна проба из лестницы за неё не
    попадает. Отказ обязан назвать именно это вырождение, а не соседнее.
    """
    sc = make_scene(-20000.0, 0.0, 2.0)
    with pytest.raises(ValueError, match=NO_PROBE_IN_FRONT) as excinfo:
        homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    assert not re.search(CENTRE_NOT_IN_FRONT, str(excinfo.value))


def test_a_workable_grazing_view_is_not_refused():
    """Охраны не должны отвергать рабочие ракурсы.

    Между 2 мм (отказ) и 200 мм (различитель выше) лежит граница. Проверяется,
    что 5 мм уже проходит: иначе охрана, отвергающая всё подряд, прошла бы оба
    теста на отказ и осталась бы незамеченной.
    """
    sc = make_scene(-20000.0, 0.0, 5.0)
    r, got = recover(sc, *exact_vps(sc))
    assert np.all(np.isfinite(r))
    assert r[1, 0] > r[0, 0]
    assert r[3, 1] < r[0, 1]
    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=1.0)
    assert got.cy == pytest.approx(truth.cy, abs=1.0)
    assert got.cz == pytest.approx(truth.cz, rel=0.02)
