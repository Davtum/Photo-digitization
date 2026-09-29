"""Панель результата. План 3, задача 15.

Правило панели: величина без σ не показывается; где σ ещё не считается, ячейка
называет причину. Признак допуска — только вместе с происхождением K.
"""
import re
import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")


def _wait(qapp, predicate, seconds=60.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    return predicate()


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    """Два кадра: дальний (reject по разрешению) и ближний (degraded, глубина измерима)."""
    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sys.path.insert(0, "scripts")
    from demo_synthetic import operator_base

    d = tmp_path_factory.mktemp("результат")
    out = {}
    for name, view in (("far", {"dx": 3000.0, "dy": -2000.0, "dist": 20000.0}),
                       ("near", {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0})):
        sc = make_scene(depth=150.0, **view)
        path = d / f"{name}.png"
        save_image(path, sc.render())
        base, span, _ = operator_base(sc)
        op = sc.openings[0]
        side, _h = visible_reveal_side(sc.camera_on_plane(), op.x, op.x + op.width,
                                       op.y, op.y + op.height)
        x = op.x if side == "left" else op.x + op.width
        inner = sc.project(np.array([[x, op.y], [x, op.y + op.height]]), depth=op.depth)
        out[name] = (path, base, span, sc.project(op.corners_mm()), inner)
    return out


def _measure(qapp, scene, *, reveal=True, edge="sharp_wall_edge", corner_origin=False):
    from facade_digitizer.ui.window import MainWindow

    path, base, span, corners, inner = scene
    w = MainWindow()
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    w.set_mode("base")
    for pt in base:
        w.handle_click(*pt, 1.0)
    w.side.base.set_values(span_mm=span, origin_is_facade_corner=corner_origin)
    w.apply_base()
    marks = w.side.marks
    marks.edge.setCurrentIndex(marks.edge.findData(edge))
    marks.new_mark.click()
    for pt in corners:
        w.handle_click(*pt, 4.0)
    if reveal:
        marks.pick_reveal.click()
        for pt in inner:
            w.handle_click(*pt, 4.0)
    model = w.compute()
    return w, model


def test_no_value_without_sigma_or_named_reason(qapp, scenes):
    w, _model = _measure(qapp, scenes["near"])
    panel = w.side.result
    for column in ("ширина, мм", "высота, мм", "заглубление, мм"):
        text = panel.cell(0, column)
        assert "±" in text, (column, text)
    position = panel.cell(0, "положение X / Y, мм")
    assert re.search(r"\d", position) and "задача 21" in position
    w.close()


def test_unmarked_reveal_is_named_not_blank(qapp, scenes):
    w, _model = _measure(qapp, scenes["near"], reveal=False)
    assert w.side.result.cell(0, "заглубление, мм") == "грань откоса не размечена"
    w.close()


def test_withheld_tolerance_shows_its_reason(qapp, scenes):
    w, model = _measure(qapp, scenes["far"])
    assert model.images[0].quality.verdict == "reject"
    assert model.elements[0].meets_tolerance is None
    assert "кадр отбракован по разрешению" in w.side.result.cell(0, "допуск")
    w.close()
    w, model = _measure(qapp, scenes["near"], edge="surround")
    assert "кромка вне плоскости стены" in w.side.result.cell(0, "допуск")
    w.close()


def test_tolerance_is_shown_with_calibration_origin(qapp, scenes):
    w, model = _measure(qapp, scenes["near"])
    header = w.side.result.header.text()
    assert model.images[0].camera.calibration == "database"
    assert "условие п. 2.2 «камера откалибрована» не выполнено" in header
    assert "зависит от введённой длины опорной базы" in header
    w.close()


def test_origin_is_labelled(qapp, scenes):
    w, _model = _measure(qapp, scenes["near"])
    assert "не от угла здания" in w.side.result.header.text()
    w.close()
    w, model = _measure(qapp, scenes["near"], corner_origin=True)
    assert model.facade.origin == "bottom_left"
    assert "от левого нижнего угла фасада" in w.side.result.header.text()
    w.close()
