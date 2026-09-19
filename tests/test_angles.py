import numpy as np
import pytest

from facade_digitizer.geometry.angles import angle_map, local_gsd, usable_mask
from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import homography_from_vanishing_points
from tests.test_homography import CORNERS, apply, exact_vps
from tests.test_synth import K, SIZE, make_scene


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
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])

    step = 20.0
    grid = local_gsd(H, mmu, SIZE, shape=(1200, 1200))
    for fx, fy in [(4000.0, 4000.0), (14000.0, 10000.0), (2000.0, 12000.0)]:
        a = sc.project(np.array([[fx, fy]]))[0]
        along_x = step / np.linalg.norm(sc.project(np.array([[fx + step, fy]]))[0] - a)
        along_y = step / np.linalg.norm(sc.project(np.array([[fx, fy + step]]))[0] - a)
        col = np.clip(int(round(a[0] / (SIZE[0] - 1) * 1199)), 0, 1199)
        row = np.clip(int(round(a[1] / (SIZE[1] - 1) * 1199)), 0, 1199)
        got = grid[row, col]
        assert got >= max(along_x, along_y) * 0.98
        assert got <= max(along_x, along_y) * 1.15


def test_worst_direction_exceeds_geometric_mean():
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    worst = local_gsd(H, mmu, SIZE)
    mean = local_gsd(H, mmu, SIZE, worst_direction=False)
    assert np.all(worst >= mean - 1e-9)
    assert worst.max() > mean.max()


def test_local_gsd_varies_across_tilted_frame():
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    g = local_gsd(H, mmu, SIZE)
    assert g.max() / g.min() > 1.2
