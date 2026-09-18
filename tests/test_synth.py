import cv2
import numpy as np
import pytest

from facade_digitizer.synth.scene import Opening, SyntheticScene, look_at

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


def test_look_at_rejects_gaze_collinear_with_world_up():
    """Взгляд строго вдоль мировой вертикали не задаёт ориентацию камеры.

    Без охраны np.cross(f, WORLD_UP) — нулевой вектор, нормировка даёт nan,
    матрица поворота молча становится мусором, а сцена всё равно рендерится.
    """
    with pytest.raises(ValueError, match="коллинеарно"):
        look_at(np.array([0.0, 0.0, 0.0]), np.array([0.0, 5000.0, 0.0]))


def test_look_at_rejects_degenerate_gaze_direction():
    """Центр камеры, совпавший с точкой наведения, не задаёт направления взгляда."""
    with pytest.raises(ValueError, match="совпадает"):
        look_at(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_look_at_returns_finite_rotation_for_valid_gaze():
    """Охрана не должна отвергать рабочие ракурсы: матрица конечна и ортонормальна."""
    R = look_at(np.array([13000.0, 5500.0, 12000.0]), np.array([10000.0, 7500.0, 0.0]))
    assert np.all(np.isfinite(R))
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)


def make_rich_scene(dx=3000.0, dy=-2000.0, dist=12000.0, depth=150.0,
                    vertical_band_step_mm=4000.0):
    """Обогащённая сцена: к межэтажным членениям добавлены вертикальные.

    Служит различителем при отладке оценщика точек схода: если оценка плоха на
    бедной сцене и хороша на обогащённой — дело в данных, а не в алгоритме.
    """
    ops = [Opening(x=4000.0, y=3000.0, width=1460.0, height=1900.0, depth=depth)]
    return SyntheticScene.looking_at_centre(20000.0, 15000.0, ops, dist, dx, dy, K, SIZE,
                                            vertical_band_step_mm=vertical_band_step_mm)


def count_vertical_segments(img, min_len_px=40.0):
    """Отрезки, отнесённые к вертикальному пучку тем же правилом, что и в оценщике."""
    lines = cv2.createLineSegmentDetector().detect(img)[0]
    if lines is None:
        return 0
    seg = lines.reshape(-1, 4)
    seg = seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= min_len_px]
    ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180.0
    return int(np.count_nonzero((ang >= 45.0) & (ang <= 135.0)))


def test_vertical_bands_are_off_by_default():
    """Умолчание не меняется сознательно: бедная сцена остаётся основной."""
    assert make_scene().vertical_band_step_mm == 0.0
    assert np.array_equal(make_scene().render(), make_rich_scene(vertical_band_step_mm=0.0).render())


def test_vertical_bands_add_vertical_segments():
    """Вертикальный пучок бедной сцены — 4 прямые; обогащённая даёт заметно больше."""
    n_poor = count_vertical_segments(make_scene().render())
    n_rich = count_vertical_segments(make_rich_scene().render())
    assert n_rich > 2 * n_poor
