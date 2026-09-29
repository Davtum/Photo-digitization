"""Шлюз качества в две ступени. План 3, задача 12; спецификация, п. 4.1.

Вердикт по разрешению зависит от длины опорной базы (разрешение считается от неё),
поэтому до базы показывается только то, что от неё не зависит.
"""
import time

import numpy as np
import pytest

from facade_digitizer.pipeline import quality

FACADE_W = 20000.0


def test_preliminary_never_rejects_and_uses_the_same_strings():
    t = quality.DEFAULT
    pre = quality.assess_preliminary(t.sharpness_min / 10, usable=0.1)
    assert pre.verdict == "degraded"
    assert pre.reasons == [
        quality.SHARPNESS_REASON.format(value=t.sharpness_min / 10,
                                        threshold=t.sharpness_min),
        quality.USABLE_FRACTION_REASON.format(value=0.1)]
    assert quality.assess_preliminary(t.sharpness_min * 10, usable=0.9).verdict == "ok"


def test_preliminary_refuses_undefined_sharpness():
    with pytest.raises(ValueError, match="резкость"):
        quality.assess_preliminary(float("nan"), usable=0.5)


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
    path = tmp_path_factory.mktemp("качество") / "facade.png"
    save_image(path, sc.render())
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    corners = sc.project(sc.openings[0].corners_mm())
    return path, base, corners


@pytest.fixture
def window(qapp, scene_file):
    pytest.importorskip("PySide6")
    from facade_digitizer.ui.window import MainWindow

    path, _base, _corners = scene_file
    w = MainWindow()
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    yield w
    w.close()


def _apply_base(w, base, span):
    w.set_mode("base")
    for pt in base:
        w.handle_click(*pt, 1.0)
    w.side.base.set_values(span_mm=span)
    w.apply_base()


def test_preliminary_verdict_is_shown_before_the_base(window):
    panel = window.side.quality
    assert "Предварительный" in panel.stage.text()
    assert "после ввода опорной базы" in panel.note.text()
    assert window.session.scale is None


def test_final_verdict_flips_with_the_base_length(window, scene_file):
    _path, base, _corners = scene_file
    panel = window.side.quality
    _apply_base(window, base, 20000.0)
    assert "Окончательный" in panel.stage.text()
    assert panel.verdict.text() == "Вердикт: reject"
    _apply_base(window, base, 15000.0)
    assert panel.verdict.text() == "Вердикт: ok"
    assert "зависит от длины опорной базы" in panel.note.text()


def test_reasons_are_the_json_strings(window, scene_file):
    from facade_digitizer.ui.session import ClickedPoint

    _path, base, corners = scene_file
    _apply_base(window, base, 20000.0)
    mark = window.session.new_mark()
    for pt in corners:
        window.session.add_corner(mark, ClickedPoint(*pt, 2.0))
    model = window.session.compute_elements(lambda p: 1.0)
    written = model.images[0].quality.reasons
    shown = window.side.quality.reason_list
    assert shown and all(r in written for r in shown)


def test_reject_warns_but_does_not_block_marking(window, scene_file):
    from facade_digitizer.ui.session import ClickedPoint

    _path, base, corners = scene_file
    _apply_base(window, base, 20000.0)
    assert "Разметка разрешена" in window.side.quality.note.text()
    mark = window.session.new_mark()
    for pt in corners:
        window.session.add_corner(mark, ClickedPoint(*pt, 2.0))
    model = window.session.compute_elements(lambda p: 1.0)
    assert model is not None and len(model.elements) == 1
    assert model.elements[0].meets_tolerance is None
