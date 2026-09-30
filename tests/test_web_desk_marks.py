"""`Desk`: проёмы, правка, результат, экспорт. Веб-интерфейс, задача 5.

Перенос `test_ui_marks`, `test_ui_edit`, `test_ui_result`, `test_ui_session_file`.
"""
import dataclasses
import json
import re
import shlex
import time

import numpy as np
import pytest

from facade_digitizer.pipeline import run
from tests import web_scenes as ws

EDIT_BUDGET_S = 0.3


@pytest.fixture(autouse=True)
def _cached_frames(monkeypatch):
    monkeypatch.setattr(run, "frame_stage", ws.cached_frame_stage)


@pytest.fixture(scope="module")
def near(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("проёмы"), ws.NEAR)


@pytest.fixture(scope="module")
def far(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("проёмы_далеко"), ws.FAR)


@pytest.fixture
def desk(near):
    d = ws.new_desk(near.path)
    ws.set_base(d, near)
    assert d.session.scale is not None
    return d


def _element(desk, mark_id):
    return next(e for e in desk.session.model.elements if e.id == mark_id)


def _shifted(scene, dx=0.0, dy=0.0):
    return dataclasses.replace(scene, corners=scene.corners + np.array([dx, dy]))


def test_four_clicks_any_order_make_an_element_with_own_sigma(desk, near):
    from facade_digitizer.ui.zoom import sigma_image_px

    mark_id = ws.add_opening(desk, near, scale=4.0, order=(3, 1, 0, 2), reveal=False)
    assert desk.placing is None
    em = desk.session.element_marks(lambda p: sigma_image_px(p.view_scale))[-1]
    assert em.id == mark_id and em.sigma_px == pytest.approx(sigma_image_px(4.0))
    assert np.allclose(em.corners_px, near.corners)
    assert "проверьте установку и тип кромки" in desk.state()["notice"]["text"]


def test_new_mark_inherits_attributes_of_previous(desk, near):
    first = ws.add_opening(desk, near, reveal=False)
    desk.act("set_mark_attrs", id=first, mounting="flush", edge_type="surround")
    desk.act("new_mark", class_="door")
    second = desk.session.mark(desk.current_mark)
    assert second.id.startswith("d_")
    assert (second.mounting, second.edge_type) == ("flush", "surround")
    assert desk.placing == "corners"


def test_mounting_and_edge_reach_the_model(desk, near):
    mark_id = ws.add_opening(desk, near, reveal=False)
    desk.act("set_mark_attrs", id=mark_id, mounting="flush", edge_type="surround")
    element = _element(desk, mark_id)
    assert element.mounting == "flush" and element.edge_type == "surround"
    assert element.edge_reference == "offset_plane"


def test_reveal_side_is_suggested(desk, near):
    mark_id = ws.add_opening(desk, near, reveal=False)
    marks = desk.state()["marks"]
    assert marks["selected"] == mark_id
    assert marks["side_choice"] == near.side
    assert "видны грани" in marks["suggestion"]


def test_changing_reveal_side_clears_points_at_once(desk, near):
    mark_id = ws.add_opening(desk, near)
    mark = desk.session.mark(mark_id)
    assert len(mark.reveal_points) == 2
    other = "right" if near.side == "left" else "left"
    desk.act("set_reveal_side", side=other)
    assert mark.reveal_points == [] and mark.reveal_side is None
    assert "стёрты" in desk.state()["notice"]["text"]


def test_coarse_click_warns_with_mm_cost(desk, near):
    desk.act("set_step", step="marks")
    desk.act("new_mark", class_="window")
    desk.act("click", x=float(near.corners[0][0]), y=float(near.corners[0][1]),
             view_scale=0.5)
    text = desk.state()["marks"]["warning"]
    assert "грубее рекомендованного" in text and "мм" in text


def test_coarse_reveal_click_warns_too(desk, near):
    ws.add_opening(desk, near, reveal=False)
    desk.act("set_reveal_side", side=near.side)
    desk.act("start_reveal")
    desk.act("click", x=float(near.inner[0][0]), y=float(near.inner[0][1]), view_scale=1.0)
    assert "грубее рекомендованного" in desk.state()["marks"]["warning"]


def test_undo_removes_the_last_point(desk, near):
    desk.act("set_step", step="marks")
    desk.act("new_mark", class_="window")
    for x, y in near.corners[:2]:
        desk.act("click", x=float(x), y=float(y), view_scale=2.0)
    desk.act("undo_point")
    assert len(desk.session.mark(desk.current_mark).corners) == 1
    assert desk.placing == "corners"


def test_esc_on_empty_mark_deletes_it(desk):
    desk.act("set_step", step="marks")
    desk.act("new_mark", class_="window")
    desk.act("stop_placing")
    assert desk.session.marks == [] and desk.current_mark is None


def test_esc_on_incomplete_mark_keeps_it_with_reason(desk, near):
    desk.act("set_step", step="marks")
    desk.act("new_mark", class_="window")
    desk.act("click", x=float(near.corners[0][0]), y=float(near.corners[0][1]), view_scale=2.0)
    desk.act("stop_placing")
    assert len(desk.session.marks) == 1 and desk.placing is None
    assert any("углов 1 из 4" in r for r in desk.state()["marks"]["reasons"])


def test_model_recomputes_itself_when_ready(desk, near):
    mark_id = ws.add_opening(desk, near)
    element = _element(desk, mark_id)
    op = near.sc.openings[0]
    assert abs(element.size_mm.width - op.width) <= element.size_mm.sigma_width
    assert abs(element.size_mm.height - op.height) <= element.size_mm.sigma_height
    assert element.recess.origin == "measured_from_reveal"
    assert abs(element.recess.value_mm - op.depth) <= element.recess.sigma_mm
    item = desk.state()["marks"]["items"][0]
    assert "±" in item["size"]


def test_drop_recomputes_fully_and_logs_one_edit(desk, near):
    from facade_digitizer.ui.zoom import click_sigma

    mark_id = ws.add_opening(desk, near, reveal=False)
    before = _element(desk, mark_id).size_mm.width
    x, y = near.corners[1]
    desk.act("drop", key=f"{mark_id}:corner:1", x=float(x + 30), y=float(y),
             view_scale=2.0, moved=True)
    assert _element(desk, mark_id).size_mm.width != pytest.approx(before, abs=1.0)
    assert desk.session.model == desk.session.compute_elements(click_sigma)
    assert desk.session.labour.elements[mark_id].edits == 1


def test_drop_without_move_is_not_an_edit(desk, near):
    mark_id = ws.add_opening(desk, near, reveal=False)
    model = desk.session.model
    x, y = near.corners[1]
    desk.act("drop", key=f"{mark_id}:corner:1", x=float(x), y=float(y),
             view_scale=2.0, moved=False)
    assert desk.session.labour.elements[mark_id].edits == 0
    assert desk.session.model is model


def test_edit_calls_neither_frame_nor_scale_stage(desk, near, monkeypatch):
    mark_id = ws.add_opening(desk, near, reveal=False)
    calls = []
    for name in ("frame_stage", "scale_stage", "estimate_plane"):
        real = getattr(run, name)
        monkeypatch.setattr(run, name,
                            lambda *a, _n=name, _r=real, **k: calls.append(_n) or _r(*a, **k))
    x, y = near.corners[3]
    desk.act("drop", key=f"{mark_id}:corner:3", x=float(x - 5), y=float(y),
             view_scale=2.0, moved=True)
    assert desk.session.model is not None and calls == []


def test_edit_recount_within_budget_on_100_elements(desk, near):
    ids = [ws.add_opening(desk, _shifted(near, (i % 10) * 4.0, (i // 10) * 4.0), reveal=False)
           for i in range(100)]
    assert len(desk.session.model.elements) == 100
    x, y = near.corners[0]
    t0 = time.perf_counter()
    desk.act("drop", key=f"{ids[50]}:corner:0", x=float(x + 3), y=float(y + 1),
             view_scale=2.0, moved=True)
    elapsed = time.perf_counter() - t0
    assert len(desk.session.model.elements) == 100
    assert elapsed < EDIT_BUDGET_S, f"{elapsed * 1000:.0f} мс на правку"


def test_points_are_draggable_only_outside_placing(desk, near):
    mark_id = ws.add_opening(desk, near, reveal=False)
    keys = [d["key"] for d in desk.state()["draggable"]]
    assert keys == [f"{mark_id}:corner:{i}" for i in range(4)]
    desk.act("new_mark", class_="window")
    assert desk.placing == "corners" and desk.state()["draggable"] == []


def test_click_inside_contour_selects_mark(desk, near):
    first = ws.add_opening(desk, near, reveal=False)
    second = ws.add_opening(desk, _shifted(near, 600.0, 0.0), reveal=False)
    assert desk.current_mark == second
    cx, cy = near.corners.mean(axis=0)
    desk.act("click", x=float(cx), y=float(cy), view_scale=1.0)
    assert desk.current_mark == first
    assert len(desk.session.mark(first).corners) == 4


def test_ids_are_stable_after_delete(desk, near):
    first = ws.add_opening(desk, near, reveal=False)
    second = ws.add_opening(desk, _shifted(near, 40.0), reveal=False)
    third = ws.add_opening(desk, _shifted(near, 80.0), reveal=False)
    desk.act("delete_mark", id=second)
    assert [m.id for m in desk.session.marks] == [first, third]
    assert [e.id for e in desk.session.model.elements] == [first, third]
    fourth = ws.add_opening(desk, _shifted(near, 120.0), reveal=False)
    assert fourth not in (first, second, third)


def test_result_rows_have_sigma_or_reason(desk, near):
    ws.add_opening(desk, near)
    rows = desk.state()["result"]["rows"]
    for key in ("width", "height", "recess"):
        assert "±" in rows[0][key], (key, rows[0][key])
    assert re.search(r"\d+ / -?\d+ ± \d", rows[0]["position"])
    assert "соседа нет" in rows[0]["position"]
    header = desk.state()["result"]["header"]
    assert "не от угла здания" in header and "зависит от введённой длины" in header


def test_unmarked_reveal_is_named_not_blank(desk, near):
    ws.add_opening(desk, near, reveal=False)
    assert desk.state()["result"]["rows"][0]["recess"] == "грань откоса не размечена"


def test_withheld_tolerance_shows_its_reason(far, near):
    d = ws.new_desk(far.path)
    ws.set_base(d, far)
    ws.add_opening(d, far)
    assert d.session.model.images[0].quality.verdict == "reject"
    assert "кадр отбракован по разрешению" in d.state()["result"]["rows"][0]["tolerance"]
    d = ws.new_desk(near.path)
    ws.set_base(d, near)
    mark_id = ws.add_opening(d, near)
    d.act("set_mark_attrs", id=mark_id, mounting="embedded", edge_type="surround")
    assert "кромка вне плоскости стены" in d.state()["result"]["rows"][0]["tolerance"]


def test_export_writes_json_dxf_marks_command_in_export_dir(desk, near):
    ws.add_opening(desk, near)
    out = desk.act("export")
    export_dir = near.path.parent / "экспорт"
    stem = near.path.stem
    names = {p.name for p in export_dir.iterdir()}
    assert {f"{stem}.json", f"{stem}.dxf", f"{stem}.marks.json",
            f"{stem}.command.txt"} <= names
    info = desk.state()["result"]["export"]
    assert info["cli"] == out.cli_line
    assert {f["name"] for f in info["files"]} >= {f"{stem}.json", f"{stem}.dxf"}
    assert desk.state()["steps"][3]["status"] == "done"
    desk.act("drop", key=f"{desk.current_mark}:corner:0", x=float(near.corners[0][0] + 2),
             y=float(near.corners[0][1]), view_scale=2.0, moved=True)
    assert desk.state()["steps"][3]["status"] == "attention"      # правки после выгрузки


def test_export_is_reproduced_by_cli_bytewise(desk, near, tmp_path):
    ws.add_opening(desk, near)
    out = desk.act("export")
    args = shlex.split(out.cli_line)[1:]
    args[args.index("--out-dir") + 1] = str(tmp_path)
    assert run.main(args) == 0
    again = (tmp_path / out.json_path.name).read_bytes()
    assert again == out.json_path.read_bytes()


def test_state_marks_carry_choices_and_sigma_params(desk):
    from facade_digitizer.ui import zoom

    state = desk.state()
    assert state["sigma"] == {"screen_px": zoom.SIGMA_SCREEN_PX, "edge_px": zoom.SIGMA_EDGE_PX,
                              "recommended_min_scale": zoom.RECOMMENDED_MIN_SCALE}
    assert [k for k, _ in state["choices"]["edges"]] == [
        "sharp_wall_edge", "surround", "cladding_edge", "unknown"]
    json.dumps(state)
