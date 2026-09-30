"""`Desk`: открытие снимка, кадр, плоскость, профиль. Веб-интерфейс, задача 3.

Перенос проверок окна Qt (`test_ui_window`, `test_ui_plane`, `test_ui_profile`,
`test_ui_quality`) на `Desk` — логику экрана без HTTP и без Qt.
"""
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.io import save_image
from tests import web_scenes as ws


@pytest.fixture(autouse=True)
def _cached_frames(monkeypatch):
    monkeypatch.setattr(run, "frame_stage", ws.cached_frame_stage)


@pytest.fixture(scope="module")
def far(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("кадр"), ws.FAR)


@pytest.fixture(scope="module")
def near(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("кадр_ближе"), ws.NEAR)


@pytest.fixture(scope="module")
def no_lines(tmp_path_factory):
    """Фактура без линий фасада и светлая табличка 2:1 — плоскость автоматически
    не восстанавливается (как `04_без_линий.jpg` приёмки, но меньше)."""
    rng = np.random.default_rng(7)
    blank = cv2.GaussianBlur(rng.normal(120, 25, (900, 1200)).clip(0, 255).astype(np.uint8),
                             (0, 0), 3)
    plaque = np.array([[400, 300], [800, 340], [790, 540], [410, 505]], np.int32)
    cv2.fillConvexPoly(blank, plaque, 230)
    path = tmp_path_factory.mktemp("без_линий") / "без_линий.png"
    save_image(path, blank)
    return path, plaque.astype(float)


def _click_all(desk, points, scale=2.0):
    for x, y in points:
        desk.act("click", x=float(x), y=float(y), view_scale=scale)


def test_web_modules_do_not_import_qt():
    code = ("import sys, facade_digitizer.web.desk, facade_digitizer.web.jobs, "
            "facade_digitizer.web.images; "
            "print(any(m.startswith('PySide6') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_open_runs_frame_phase_and_encodes_lossless_png(far):
    desk = ws.new_desk(far.path)
    fs = desk.session.frame
    assert fs is not None and desk.busy is None
    decoded = cv2.imdecode(np.frombuffer(desk.frame_png, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, fs.frame.color)
    state = desk.state()
    assert state["frame"]["width"] == fs.image_size[0]
    assert state["frame"]["height"] == fs.image_size[1]
    assert state["frame"]["version"] == desk.frame_version >= 1
    assert desk.usable_png is not None


def test_state_is_json_serialisable(far):
    desk = ws.new_desk(far.path)
    ws.set_base(desk, far, span_mm=ws.FACADE_W)
    json.dumps(desk.state())


def test_needs_operator_opens_manual_plane_placing(no_lines):
    path, _plaque = no_lines
    desk = ws.new_desk(path)
    state = desk.state()
    assert desk.session.frame.plane.needs_operator
    assert state["step"] == "frame" and state["placing"] == "plane"
    assert "из 4" in state["hint"]["text"]
    assert state["steps"][0]["status"] == "attention"


@pytest.mark.parametrize("constraint", [
    {"kind": "aspect", "aspect": 2.0},
    {"kind": "sizes", "width_mm": 2000.0, "height_mm": 1000.0},
    {"kind": "calibrated"},
])
def test_manual_plane_each_constraint_reaches_the_frame(no_lines, constraint):
    path, plaque = no_lines
    desk = ws.new_desk(path)
    version = desk.frame_version
    _click_all(desk, plaque[[2, 0, 3, 1]])              # порядок кликов не важен
    desk.act("set_plane_constraint", **constraint)
    desk.act("apply_manual_plane")
    fs = desk.session.frame
    assert not fs.plane.needs_operator and fs.plane.method == "manual_four_point"
    assert desk.placing is None
    assert desk.frame_version == version                  # кадр тот же — не перечитан
    assert desk.state()["frame"]["plane"]["manual"] is True


def test_three_corners_refused_by_name(no_lines):
    path, plaque = no_lines
    desk = ws.new_desk(path)
    _click_all(desk, plaque[:3])
    desk.act("apply_manual_plane")
    assert desk.session.frame.plane.needs_operator
    assert "углов 3 из 4" in desk.state()["notice"]["text"]


def test_plane_and_roi_placings_are_exclusive(far):
    desk = ws.new_desk(far.path)
    desk.act("start_manual_plane")
    assert desk.placing == "plane"
    desk.act("start_roi")
    assert desk.placing == "roi"
    desk.act("click", x=100.0, y=100.0, view_scale=1.0)
    assert len(desk.session.roi_points) == 1 and not desk.session.plane_points


def test_roi_is_passed_to_the_frame_stage(far, monkeypatch):
    seen = {}
    real = run.replace_plane

    def spy(fs, **kw):
        seen.update(kw)
        return real(fs, **kw)

    monkeypatch.setattr(run, "replace_plane", spy)
    desk = ws.new_desk(far.path)
    desk.act("start_roi")
    w, h = desk.session.frame.image_size
    _click_all(desk, [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], scale=0.25)
    desk.act("apply_roi")
    assert seen["roi"].shape == (4, 2)
    assert desk.session.roi is not None and desk.placing is None
    assert desk.session.frame.plane.method == "vanishing_points"


def test_reset_returns_to_the_automatic_plane(far):
    desk = ws.new_desk(far.path)
    auto_conf = desk.session.frame.plane.confidence.value
    desk.act("start_manual_plane")
    _click_all(desk, far.facade_quad, scale=0.25)
    desk.act("set_plane_constraint", kind="sizes", width_mm=ws.FACADE_W, height_mm=ws.FACADE_H)
    desk.act("apply_manual_plane")
    assert desk.session.plane_override is not None
    desk.act("reset_plane")
    assert desk.session.plane_override is None and desk.session.roi is None
    assert desk.session.frame.plane.confidence.value == pytest.approx(auto_conf)


def test_new_plane_recomputes_scale_and_model_when_ready(far):
    desk = ws.new_desk(far.path)
    ws.set_base(desk, far, span_mm=ws.FACADE_W)
    ws.add_opening(desk, far, reveal=False)
    assert desk.session.model is not None
    desk.act("set_step", step="frame")
    desk.act("start_manual_plane")
    _click_all(desk, far.facade_quad, scale=0.25)
    desk.act("set_plane_constraint", kind="sizes", width_mm=ws.FACADE_W, height_mm=ws.FACADE_H)
    desk.act("apply_manual_plane")
    assert desk.session.scale is not None and desk.session.model is not None


def test_preliminary_verdict_uses_core_words(far):
    from facade_digitizer.pipeline import quality

    desk = ws.new_desk(far.path)
    q = desk.state()["quality"]
    preview = run.theta_preview(desk.session.frame)
    expected = quality.assess_preliminary(desk.session.frame.sharpness, preview.usable_fraction)
    assert q["stage"] == "preliminary"
    assert q["verdict"] == expected.verdict in ("ok", "degraded")
    assert q["reasons"] == list(expected.reasons)
    assert "пригодно по углу" in " ".join(desk.state()["frame"]["info"])


def test_profile_change_with_clicks_needs_discard(near, tmp_path):
    desk = ws.new_desk(near.path)
    desk.act("set_step", step="scale")
    desk.act("click", x=float(near.base_px[0][0]), y=float(near.base_px[0][1]), view_scale=1.0)
    profile = ws.profile_for(tmp_path, near.sc)
    version = desk.frame_version
    desk.act("set_profile", path=str(profile), discard=False)
    notice = desk.state()["notice"]
    assert notice["kind"] == "confirm" and "сбросит" in notice["text"]
    assert notice["confirm"] == {"name": "set_profile",
                                 "args": {"path": str(profile), "discard": True}}
    assert len(desk.session.reference.ends) == 1
    desk.act("set_profile", path=str(profile), discard=True)
    assert desk.session.reference.ends == []
    assert desk.frame_version == version + 1
    assert desk.session.frame.camera_record.calibration == "target"


def test_foreign_profile_is_refused_by_camera_name(near, tmp_path):
    desk = ws.new_desk(near.path)
    profile = ws.profile_for(tmp_path, near.sc, model="OTHER-CAM")
    desk.act("set_profile", path=str(profile), discard=True)
    assert desk.session.frame is None
    assert "OTHER-CAM" in desk.state()["notice"]["text"]
    desk.act("set_profile", path=None, discard=True)
    assert desk.session.frame is not None


def test_actions_while_busy_raise_busy(far):
    from facade_digitizer.web.desk import DeskBusy

    desk = ws.new_desk(far.path)
    desk.busy = "обрабатывается"
    with pytest.raises(DeskBusy):
        desk.act("click", x=1.0, y=1.0, view_scale=1.0)
    desk.act("set_step", step="scale")               # переход между шагами разрешён
    assert desk.step == "scale"


def test_click_outside_the_frame_is_refused(far):
    desk = ws.new_desk(far.path)
    desk.act("start_roi")
    w, _h = desk.session.frame.image_size
    desk.act("click", x=float(w), y=10.0, view_scale=1.0)
    assert desk.session.roi_points == []
    assert "вне снимка" in desk.state()["notice"]["text"]
    desk.act("click", x=float(w - 1), y=0.0, view_scale=1.0)
    assert len(desk.session.roi_points) == 1


def test_unknown_action_is_refused(far):
    desk = ws.new_desk(far.path)
    with pytest.raises(KeyError):
        desk.act("__init__")
