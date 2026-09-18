import cv2
import numpy as np
import pytest

from facade_digitizer.synth.scene import Opening, SyntheticScene

K = np.array([[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]])
SIZE = (5280, 3956)


def make_scene(dx=3000.0, dy=-2000.0, dist=12000.0, depth=150.0):
    ops = [Opening(x=4000.0, y=3000.0, width=1460.0, height=1900.0, depth=depth)]
    return SyntheticScene.looking_at_centre(20000.0, 15000.0, ops, dist, dx, dy, K, SIZE)


def test_projection_is_actually_perspective():
    """Ключевой приёмочный критерий: равные шаги по фасаду дают РАЗНЫЕ шаги в кадре."""
    sc = make_scene()
    xs = np.array([[x, 7500.0] for x in range(0, 20001, 5000)], dtype=float)
    steps = np.diff(sc.project(xs)[:, 0])
    assert steps.std() / steps.mean() > 0.02


def test_render_produces_image_of_requested_size():
    img = make_scene().render()
    assert img.shape == (3956, 5280)
    assert img.dtype == np.uint8


def test_render_has_enough_structure_for_line_detection():
    img = make_scene().render()
    assert img.std() > 5.0
    assert len(np.unique(img)) >= 4     # фон, стена, членения, откос, полотно


def count_long_segments(img, min_len_px=40.0):
    """Отрезки длиной от min_len_px, найденные LSD, — то, чем питается оценка точек схода."""
    lines = cv2.createLineSegmentDetector().detect(img)[0]
    if lines is None:
        return 0
    seg = lines.reshape(-1, 4)
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return int(np.count_nonzero(lengths >= min_len_px))


def test_floor_bands_give_enough_segments_for_vanishing_points():
    """Голая стена даёт ~9 отрезков — для устойчивой оценки точек схода мало.

    Членения — не украшение: они и есть тот горизонтальный пучок, по которому
    задача 6 ищет точку схода. Измерено: 35 отрезков против 9 на голой стене.
    """
    ops = [Opening(x=4000.0, y=3000.0, width=1460.0, height=1900.0, depth=150.0)]
    bare = SyntheticScene.looking_at_centre(20000.0, 15000.0, ops, 12000.0,
                                            3000.0, -2000.0, K, SIZE,
                                            floor_band_step_mm=0.0)
    n_default = count_long_segments(make_scene().render())
    n_bare = count_long_segments(bare.render())
    assert n_default >= 25
    assert n_default > 2 * n_bare


def test_camera_on_plane_reports_true_pose():
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=12000.0)
    cam = sc.camera_on_plane()
    assert cam.cx == pytest.approx(10000.0 + 3000.0)
    assert cam.cy == pytest.approx(7500.0 - 2000.0)
    assert cam.cz == pytest.approx(12000.0)


def test_deeper_opening_shows_wider_reveal():
    assert make_scene(depth=300.0).reveal_width_px() > make_scene(depth=50.0).reveal_width_px()


def test_point_behind_camera_is_rejected():
    sc = make_scene()
    with pytest.raises(ValueError):
        sc.project(np.array([[0.0, 0.0]]), depth=-20000.0)
