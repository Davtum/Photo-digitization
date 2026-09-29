"""Правка разметки — полный пересчёт. План 3, задача 16."""
import dataclasses
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from tests.test_ui_marks import VIEW, _wait

#: Бюджет пересчёта 100 элементов после одной правки, с. Замер (отчёт плана 3,
#: operator-ui-edit.md) — 75 мс; бюджет взят с запасом на медленную машину CI.
EDIT_BUDGET_S = 1.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    import sys

    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sc = make_scene(**VIEW)
    path = tmp_path_factory.mktemp("правка") / "facade.png"
    save_image(path, sc.render())
    sys.path.insert(0, "scripts")
    from demo_synthetic import operator_base

    base_px, span_mm, _origin = operator_base(sc)
    corners = sc.project(sc.openings[0].corners_mm())
    return path, base_px, span_mm, corners


@pytest.fixture
def window(qapp, scene):
    from facade_digitizer.ui.window import MainWindow

    path, base_px, span_mm, _corners = scene
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    w.set_mode("base")
    for pt in base_px:
        w.handle_click(*pt, 1.0)
    w.side.base.set_values(span_mm=span_mm)
    w.apply_base()
    assert w.session.scale is not None
    yield w
    w.close()


def _mark(w, corners, shift=(0.0, 0.0)):
    w.side.marks.new_mark.click()
    for x, y in corners:
        w.handle_click(x + shift[0], y + shift[1], 2.0)
    w.set_mode("navigate")
    return w.current_mark


def _element(model, mark_id):
    return next(e for e in model.elements if e.id == mark_id)


def test_dragging_a_corner_recomputes_the_element(window, scene):
    from facade_digitizer.ui.zoom import click_sigma

    *_, corners = scene
    mark_id = _mark(window, corners)
    before = _element(window.compute(), mark_id).size_mm.width
    x, y = corners[1]
    window.drop_point(f"{mark_id}:corner:1", x + 30.0, y, 2.0)
    after = _element(window.session.model, mark_id).size_mm.width
    assert after != pytest.approx(before, abs=1.0)
    # Правка равна полному пересчёту той же разметки с нуля.
    edited = window.session.model
    assert edited == window.session.compute_elements(click_sigma)


def test_mouse_drag_on_the_canvas_moves_the_point(qapp, window, scene):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtTest import QTest

    *_, corners = scene
    mark_id = _mark(window, corners)
    canvas = window.canvas
    canvas.set_view_scale(2.0)
    x, y = corners[2]
    canvas.centerOn(QPointF(x + 0.5, y + 0.5))
    qapp.processEvents()
    start = canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))
    end = start + QPoint(20, 0)
    vp = canvas.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(vp, end)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end)
    moved = window.session.mark(mark_id).corners[2]
    assert moved.x == pytest.approx(x + 10.0, abs=0.6)       # 20 экранных px при 2:1
    assert moved.y == pytest.approx(y, abs=0.6)
    assert moved.view_scale == pytest.approx(2.0)
    assert window.session.model is not None


def test_edit_recount_within_budget_on_100_elements(window, scene):
    *_, corners = scene
    ids = [_mark(window, corners, shift=((i % 10) * 4.0, (i // 10) * 4.0))
           for i in range(100)]
    assert len(window.compute().elements) == 100
    x, y = corners[0]
    t0 = time.perf_counter()
    window.drop_point(f"{ids[50]}:corner:0", x + 3.0, y + 1.0, 2.0)
    elapsed = time.perf_counter() - t0
    print(f"\nправка при 100 элементах: {elapsed * 1000:.0f} мс")
    assert len(window.session.model.elements) == 100
    assert elapsed < EDIT_BUDGET_S, f"{elapsed * 1000:.0f} мс на правку"


def test_edit_calls_neither_frame_nor_scale_stage(window, scene, monkeypatch):
    from facade_digitizer.pipeline import run

    *_, corners = scene
    mark_id = _mark(window, corners)
    window.compute()
    calls = []
    for name in ("frame_stage", "scale_stage", "estimate_plane"):
        real = getattr(run, name)
        monkeypatch.setattr(run, name,
                            lambda *a, _n=name, _r=real, **k: calls.append(_n) or _r(*a, **k))
    x, y = corners[3]
    window.drop_point(f"{mark_id}:corner:3", x - 5.0, y, 2.0)
    assert window.session.model is not None
    assert calls == []


def test_ids_are_stable_after_deleting_a_mark(window, scene):
    *_, corners = scene
    first = _mark(window, corners)
    second = _mark(window, corners, shift=(40.0, 0.0))
    third = _mark(window, corners, shift=(80.0, 0.0))
    window.current_mark = second
    window.delete_mark()
    assert [m.id for m in window.session.marks] == [first, third]
    x, y = corners[0]
    window.drop_point(f"{third}:corner:0", x + 82.0, y, 2.0)
    assert [e.id for e in window.session.model.elements] == [first, third]
    assert window.session.mark(third).corners[0].x == pytest.approx(x + 82.0)
    fourth = _mark(window, corners, shift=(120.0, 0.0))
    assert fourth not in (first, second, third)      # удалённый номер не переиспользован


def test_reject_branch_is_applied_after_every_edit(window, scene):
    *_, corners = scene
    a = _mark(window, corners)
    b = _mark(window, corners, shift=(40.0, 0.0))
    ss = window.session.scale
    window.session.scale = dataclasses.replace(
        ss, quality=ss.quality.model_copy(update={"verdict": "reject"}))
    x, y = corners[0]
    for dx in (1.0, 2.0, 3.0):
        window.drop_point(f"{a}:corner:0", x + dx, y, 2.0)
        model = window.session.model
        assert {e.id for e in model.elements} == {a, b}
        assert all(e.meets_tolerance is None for e in model.elements)
        assert any("reject" in r for r in model.images[0].quality.reasons)


def test_drag_before_the_base_only_moves_the_point(qapp, scene):
    """Без базы пересчитывать нечего: точка переносится, модель не появляется."""
    from facade_digitizer.ui.window import MainWindow

    path, _base, _span, corners = scene
    w = MainWindow()
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    mark_id = _mark(w, corners)
    w.drop_point(f"{mark_id}:corner:0", corners[0][0] + 5.0, corners[0][1], 2.0)
    assert w.session.mark(mark_id).corners[0].x == pytest.approx(corners[0][0] + 5.0)
    assert w.session.model is None
    w.close()


def test_points_are_grabbed_only_in_navigate_mode(window, scene):
    *_, corners = scene
    _mark(window, corners)
    assert window.canvas.editable_points()
    window.set_mode_keep("opening")
    assert window.canvas.editable_points() == []
    assert np.isfinite(corners).all()
