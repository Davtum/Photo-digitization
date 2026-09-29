"""Обучение операторов: блоки с истиной и критерий. План 3, задача 24."""
import json

import numpy as np
import pytest

from facade_digitizer.training import (
    ERROR_TOLERANCE_MM,
    BlockStats,
    block_stats,
    converged,
    make_block,
    score_scene,
    truth_record,
    write_block,
)


def test_blocks_are_deterministic_and_openings_are_distinct():
    assert make_block(3, seed=7) == make_block(3, seed=7)
    assert make_block(3, seed=7) != make_block(4, seed=7)
    for spec in make_block(0):
        sizes = {(o["width"], o["height"]) for o in spec.openings}
        assert len(sizes) == len(spec.openings)
        widths = [o["width"] for o in spec.openings]
        assert len(set(widths)) == len(widths)


def test_perfect_marking_scores_within_sigma(tmp_path):
    """Разметка по истинным углам через окно-сессию → промахи в пределах σ."""
    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.ui.session import ClickedPoint, OperatorSession
    from facade_digitizer.ui.session_file import export
    from facade_digitizer.ui.zoom import click_sigma

    paths = write_block(tmp_path, 0)
    truth = json.loads((paths[0].parent / "истина.json").read_text(encoding="utf-8"))[0]
    spec = make_block(0)[0]
    sc = spec.scene()
    s = OperatorSession()
    s.open_image(paths[0])
    assert s.compute_frame() is not None, s.error
    first = sc.openings[0]
    for x, y in sc.project(np.array([[first.x, first.y], [first.x + first.width, first.y]])):
        s.add_reference_end(ClickedPoint(x, y, 2.0))
    s.set_span_mm(first.width)
    cam = sc.camera_on_plane()
    for op in sc.openings:
        mark = s.new_mark()
        for x, y in sc.project(op.corners_mm()):
            s.add_corner(mark, ClickedPoint(x, y, 3.0))
        side, _ = visible_reveal_side(cam, op.x, op.x + op.width, op.y, op.y + op.height)
        edge_x = op.x if side == "left" else op.x + op.width
        if side in ("left", "right"):
            inner = sc.project(np.array([[edge_x, op.y], [edge_x, op.y + op.height]]),
                               depth=op.depth)
            for x, y in inner:
                s.add_reveal_point(mark, side, ClickedPoint(x, y, 3.0))
    out = export(s, tmp_path / "экспорт", click_sigma, dxf=False)
    model = json.loads(out.json_path.read_text(encoding="utf-8"))
    scores = score_scene(model, truth)
    assert len(scores) == 3
    by_id = {e["id"]: e for e in model["elements"]}
    for score in scores:
        size = by_id[score.element_id]["size_mm"]
        assert abs(score.width_error_mm) <= 2 * size["sigma_width"]
        assert abs(score.height_error_mm) <= 2 * size["sigma_height"]
    stats = block_stats(0, scores)
    assert stats.elements == 3 and stats.median_abs_error_mm < 20.0


def _stats(times, errors):
    return [BlockStats(i, 9, t, e) for i, (t, e) in enumerate(zip(times, errors))]


def test_convergence_needs_three_stable_blocks():
    ok, why = converged(_stats([60, 40], [9, 7]))
    assert not ok and "минимум 3" in why
    ok, _ = converged(_stats([90, 60, 41, 40, 38], [12, 8, 6.5, 6, 6.2]))
    assert ok
    ok, why = converged(_stats([60, 40, 30], [6, 6, 6]))         # время ещё падает
    assert not ok and "время" in why
    ok, _ = converged(_stats([40, 40, 40], [6, 6 + ERROR_TOLERANCE_MM + 1, 6]))
    assert not ok
    ok, why = converged(_stats([None, None, None], [6, 6, 6]))
    assert not ok and "хронометража" in why


def test_unmatched_scene_scores_nothing():
    truth = truth_record(make_block(0)[0])
    assert score_scene({"elements": []}, truth) == []
    with pytest.raises(ValueError, match="ни одного"):
        block_stats(0, [])
