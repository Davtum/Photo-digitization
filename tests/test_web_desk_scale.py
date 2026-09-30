"""`Desk`: опорная база, масштаб, качество, выровненный вид. Веб-интерфейс, задача 4.

Перенос `test_ui_base`, `test_ui_quality`, `test_ui_rectified` окна Qt.
"""
import math

import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline import run
from tests import web_scenes as ws


@pytest.fixture(autouse=True)
def _cached_frames(monkeypatch):
    monkeypatch.setattr(run, "frame_stage", ws.cached_frame_stage)


@pytest.fixture(scope="module")
def far(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("масштаб"), ws.FAR)


@pytest.fixture
def desk(far):
    return ws.new_desk(far.path)


def _base_far(desk, far, scales=(1.0, 1.0), span=ws.FACADE_W, **options):
    base = far.sc.project(np.array([[0.0, 0.0], [ws.FACADE_W, 0.0]]))
    desk.act("set_step", step="scale")
    desk.act("restart_base")
    for (x, y), s in zip(base, scales):
        desk.act("click", x=float(x), y=float(y), view_scale=s)
    desk.act("set_base", span_mm=span, span_sigma_mm=options.get("span_sigma_mm"),
             scale_source=options.get("scale_source", "operator_reference"),
             origin_is_facade_corner=options.get("origin_is_facade_corner", False))
    return base


def test_entering_scale_step_starts_base_placing(desk):
    desk.act("set_step", step="scale")
    assert desk.placing == "base"
    assert "первый конец" in desk.state()["hint"]["text"]


def test_base_clicks_keep_view_scale_and_per_end_sigma(desk, far):
    from facade_digitizer.ui.zoom import click_sigma, sigma_image_px

    _base_far(desk, far, scales=(1.0, 0.25))
    ref = desk.session.operator_reference(click_sigma)
    assert ref.end_sigma_px == (sigma_image_px(1.0), sigma_image_px(0.25))
    lines = desk.state()["scale"]["end_lines"]
    assert len(lines) == 2 and "грубее 1:1" in lines[1]
    assert desk.placing is None


def test_third_click_does_nothing(desk, far):
    base = _base_far(desk, far)
    desk.act("click", x=10.0, y=10.0, view_scale=1.0)
    assert [p.xy for p in desk.session.reference.ends] == [tuple(map(float, p)) for p in base]


def test_set_base_computes_scale_and_final_verdict(desk, far):
    _base_far(desk, far)
    state = desk.state()
    assert desk.session.scale is not None
    assert state["quality"]["stage"] == "final"
    assert state["quality"]["verdict"] == desk.session.scale.quality.verdict
    assert f"{desk.session.scale.sigma_rel:.3%}" in state["scale"]["result"]
    assert state["steps"][1]["status"] == "done"


@pytest.mark.parametrize("span", [None, 0.0, -5.0, float("nan")])
def test_invalid_span_is_not_a_reject(desk, far, span):
    _base_far(desk, far, span=span)
    assert desk.session.scale is None
    state = desk.state()
    assert state["quality"]["stage"] == "preliminary"
    assert "длина опорной базы" in state["scale"]["error"]


def test_final_verdict_flips_with_the_base_length(desk, far):
    _base_far(desk, far, span=ws.FACADE_W)
    first = desk.session.scale.quality.verdict
    _base_far(desk, far, span=ws.FACADE_W * 10)
    assert desk.session.scale.quality.verdict == "reject" != first or first == "reject"
    assert "недостаточное разрешение" in " ".join(desk.state()["quality"]["reasons"])


def test_span_uncertainty_enters_sigma_rel(desk, far):
    _base_far(desk, far)
    plain = desk.session.scale.sigma_rel
    _base_far(desk, far, span_sigma_mm=10.0)
    assert desk.session.scale.sigma_rel == pytest.approx(
        math.hypot(plain, 10.0 / ws.FACADE_W), rel=1e-12)


def test_dragging_a_base_end_recomputes_scale(desk, far):
    base = _base_far(desk, far)
    before = desk.session.scale
    assert [d["key"] for d in desk.state()["draggable"]] == ["base:0", "base:1"]
    desk.act("drop", key="base:1", x=float(base[1][0] - 30), y=float(base[1][1]),
             view_scale=2.0, moved=True)
    assert desk.session.scale is not None and desk.session.scale is not before
    assert desk.session.reference.ends[1].view_scale == 2.0


def test_restart_base_clears_ends_and_places_again(desk, far):
    _base_far(desk, far)
    desk.act("restart_base")
    assert desk.session.reference.ends == [] and desk.placing == "base"
    assert desk.session.scale is None


def test_rectified_view_is_built_after_scale(desk, far):
    desk.runner.run_raster = True
    from facade_digitizer.geometry.homography import apply_homography

    _base_far(desk, far)
    rect = desk.state()["rectified"]
    assert rect["status"] == "ready"
    H = np.array(rect["H"])
    corners = far.corners
    raster = run.raster_stage(desk.session.frame, run.raster_geometry_for(
        desk.session.frame, desk.session.scale), color=True)
    assert np.allclose(apply_homography(H, corners), apply_homography(raster.H, corners))
    assert np.allclose(np.array(rect["H_inv"]) @ H / (np.array(rect["H_inv"]) @ H)[2, 2],
                       np.eye(3), atol=1e-9)
    image = cv2.imdecode(np.frombuffer(desk.rect_jpg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[1] == rect["width"] and image.shape[0] == rect["height"]
    assert 0.5 < rect["coverage"] < 1.0 and f"{rect['coverage']:.0%}" in rect["text"]


def test_origin_lands_on_origin_rect_px(desk, far):
    desk.runner.run_raster = True
    from facade_digitizer.geometry.homography import apply_homography

    _base_far(desk, far)
    rect = desk.state()["rectified"]
    origin = desk.session.reference.ends[0].xy
    raster = run.raster_stage(desk.session.frame, run.raster_geometry_for(
        desk.session.frame, desk.session.scale))
    assert apply_homography(np.array(rect["H"]), [origin])[0] == pytest.approx(
        raster.origin_rect_px, abs=0.01)


def test_rectified_is_none_without_scale(desk, far):
    _base_far(desk, far, span=None)
    assert desk.state()["rectified"]["status"] == "none"


def test_reject_does_not_block_marking(desk, far):
    _base_far(desk, far, span=ws.FACADE_W * 10)
    assert desk.session.scale.quality.verdict == "reject"
    mark_id = ws.add_opening(desk, far, reveal=False)
    assert len(desk.session.mark(mark_id).corners) == 4
    assert desk.session.model is not None
