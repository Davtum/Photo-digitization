import numpy as np
import pytest

from facade_digitizer.geometry.homography import (
    camera_pose,
    foot_point_in_rectified,
    homography_from_four_points,
    homography_from_vanishing_points,
    rectified_to_facade_mm,
)
from tests.test_synth import K, SIZE, make_scene

CORNERS = np.array([[0.0, 0], [20000.0, 0], [20000.0, 15000.0], [0, 15000.0]])


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
    with pytest.raises(ValueError):
        homography_from_four_points(pts)


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
    with pytest.raises(ValueError):
        homography_from_four_points(pts, assume_calibrated=True)


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
