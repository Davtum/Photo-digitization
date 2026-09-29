"""Плоскость в окне: доверие, ручное задание, переопределение, область. План 3, задача 10.

Окно ведётся через `handle_click` — тот же метод, в который приходит клик холста, —
поэтому путь от клика до плоскости проверяется без мыши.
"""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

FACADE_W, FACADE_H = 20000.0, 15000.0


def _wait(qapp, predicate, seconds=60.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    return predicate()


@pytest.fixture(scope="module")
def scene_file(tmp_path_factory):
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    path = tmp_path_factory.mktemp("окно") / "facade.png"
    save_image(path, sc.render())
    bl, br, tr, tl = np.array([[0, 0], [FACADE_W, 0], [FACADE_W, FACADE_H], [0, FACADE_H]])
    quad_bl_br_tr_tl = sc.project(np.array([bl, br, tr, tl]))
    return sc, path, quad_bl_br_tr_tl


@pytest.fixture
def window(qapp, scene_file):
    from facade_digitizer.ui.window import MainWindow

    _sc, path, _quad = scene_file
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    yield w
    w.close()


def test_needs_operator_opens_the_manual_plane_tool(qapp, tmp_path):
    from facade_digitizer.pipeline.io import save_image
    from facade_digitizer.ui.window import MainWindow

    rng = np.random.default_rng(0)
    path = tmp_path / "пусто.png"
    save_image(path, (128 + rng.normal(0, 3, (600, 800))).clip(0, 255).astype(np.uint8))
    w = MainWindow()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    assert w.mode == "plane"
    assert "Укажите четыре угла" in w.side.plane.info.text()
    assert w.side.plane.apply_manual.isEnabled()
    w.close()


@pytest.mark.parametrize("which, values", [
    ("aspect", {"aspect_ratio": FACADE_W / FACADE_H}),
    ("sizes", {"size_mm": (FACADE_W, FACADE_H)}),
    ("calibrated", {}),
])
def test_manual_plane_with_each_constraint_reaches_the_frame(qapp, window, scene_file,
                                                             which, values):
    sc, _path, quad = scene_file
    w = window
    w.set_mode("plane")
    for i in (2, 0, 3, 1):                               # порядок кликов не важен
        w.handle_click(*quad[i], 2.0)
    assert "4 из 4" in w.side.plane.corners_label.text()
    w.side.plane.set_constraint(which, **values)
    w.apply_manual_plane()
    plane = w.session.frame.plane
    assert plane.method == "manual_four_point"
    assert w.session.plane_override is not None
    assert "задана оператором" in w.status_label.text()
    assert w.mode == "navigate"
    # Плоскость годна для измерения: поза по ней совпадает с позой сцены.
    from facade_digitizer.pipeline import run

    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    ref = run.OperatorReference(origin_px=tuple(base[0]),
                                span_px=(tuple(base[0]), tuple(base[1])), span_mm=FACADE_W)
    cam = run.scale_stage(w.session.frame, ref).camera
    truth = sc.camera_on_plane()
    # K здесь из типового поля зрения (снимок без EXIF): поза расходится на ~2 %.
    assert cam.cz == pytest.approx(truth.cz, rel=0.03)


def test_manual_plane_with_three_corners_is_refused_by_name(window, scene_file):
    _sc, _path, quad = scene_file
    w = window
    w.set_mode("plane")
    for pt in quad[:3]:
        w.handle_click(*pt, 2.0)
    before = w.session.frame
    w.apply_manual_plane()
    assert w.session.frame is before
    assert "углов 3 из 4" in w.status_label.text()


def test_override_is_available_at_high_confidence(window):
    w = window
    plane = w.session.frame.plane
    assert not plane.needs_operator and plane.confidence.value > 0.5
    assert w.side.plane.pick_corners.isEnabled() and w.side.plane.apply_manual.isEnabled()
    assert "Переопределить" in w.side.plane.info.text()


def test_roi_is_passed_to_the_frame_stage(qapp, window, monkeypatch):
    from facade_digitizer.pipeline import run

    seen = {}
    real = run.replace_plane

    def spy(fs, **kwargs):
        seen.update(kwargs)
        return real(fs, **kwargs)

    monkeypatch.setattr(run, "replace_plane", spy)
    w = window
    w.set_mode("roi")
    for pt in [(0.0, 0.0), (5279.0, 0.0), (5279.0, 3955.0), (0.0, 3955.0)]:
        w.handle_click(*pt, 0.25)
    before = w.session.frame
    w.apply_roi()
    assert _wait(qapp, lambda: w.session.frame is not before)
    assert seen["roi"].shape == (4, 2)
    assert w.session.roi is not None
    assert w.session.frame.plane.method == "vanishing_points"


def test_reset_returns_to_the_automatic_plane(qapp, window, scene_file):
    _sc, _path, quad = scene_file
    w = window
    w.set_mode("plane")
    for pt in quad:
        w.handle_click(*pt, 2.0)
    w.side.plane.set_constraint("sizes", size_mm=(FACADE_W, FACADE_H))
    w.apply_manual_plane()
    manual = w.session.frame
    w.reset_plane()
    assert _wait(qapp, lambda: w.session.frame is not manual)
    assert w.session.frame.plane.method == "vanishing_points"
    assert w.session.plane_override is None
