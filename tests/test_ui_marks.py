"""Разметка проёма и откоса в окне. План 3, задача 13."""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

FACADE_W = 20000.0
VIEW = {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0, "depth": 150.0}   # глубина измерима


def _wait(qapp, predicate, seconds=60.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    return predicate()


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sc = make_scene(**VIEW)
    path = tmp_path_factory.mktemp("проёмы") / "facade.png"
    save_image(path, sc.render())
    # База — самая длинная видимая хорда (фасад целиком в кадр не входит).
    import sys
    sys.path.insert(0, "scripts")
    from demo_synthetic import operator_base

    base_px, span_mm, _origin = operator_base(sc)
    op = sc.openings[0]
    corners = sc.project(op.corners_mm())
    side, _h = visible_reveal_side(sc.camera_on_plane(), op.x, op.x + op.width,
                                   op.y, op.y + op.height)
    x = op.x if side == "left" else op.x + op.width
    inner = sc.project(np.array([[x, op.y], [x, op.y + op.height]]), depth=op.depth)
    return sc, path, base_px, span_mm, corners, side, inner


@pytest.fixture
def window(qapp, scene):
    from facade_digitizer.ui.window import MainWindow

    _sc, path, base_px, span_mm, *_ = scene
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


def _opening(w, corners, scale=2.0, order=(0, 1, 2, 3)):
    w.side.marks.new_mark.click()
    for i in order:
        w.handle_click(*corners[i], scale)
    return w.session.mark(w.current_mark)


def test_four_clicks_produce_an_element_mark_with_its_own_sigma(window, scene):
    from facade_digitizer.ui.zoom import sigma_image_px

    *_, corners, _side, _inner = scene
    mark = _opening(window, corners, scale=4.0, order=(3, 1, 0, 2))
    assert window.mode == "navigate"
    em = window.session.element_marks(lambda p: sigma_image_px(p.view_scale))[-1]
    assert em.id == mark.id
    assert em.sigma_px == pytest.approx(sigma_image_px(4.0))
    assert np.allclose(em.corners_px, corners)


def test_reveal_side_is_suggested_by_visible_reveal_side(window, scene):
    *_, corners, side, _inner = scene
    _opening(window, corners)
    assert window.side.marks.side.currentData() == side
    assert "видны грани" in window.side.marks.suggestion.text()


def test_mounting_and_edge_come_from_schema_values(window, scene):
    *_, corners, _side, _inner = scene
    marks = window.side.marks
    assert [marks.mounting.itemData(i) for i in range(marks.mounting.count())] == \
        ["embedded", "flush", "protruding"]
    marks.mounting.setCurrentIndex(marks.mounting.findData("flush"))
    marks.edge.setCurrentIndex(marks.edge.findData("surround"))
    _opening(window, corners)
    model = window.compute()
    element = model.elements[-1]
    assert element.mounting == "flush" and element.edge_type == "surround"
    assert element.edge_reference == "offset_plane"      # п. 5.6: кромка вне Π


def test_coarse_click_warns_with_the_millimetre_cost(window, scene):
    *_, corners, _side, _inner = scene
    window.side.marks.new_mark.click()
    window.handle_click(*corners[0], 0.5)
    text = window.side.marks.warning.text()
    assert "грубее рекомендованного" in text and "мм" in text


def test_undo_removes_the_last_point(window, scene):
    *_, corners, _side, _inner = scene
    window.side.marks.new_mark.click()
    window.handle_click(*corners[0], 2.0)
    window.handle_click(*corners[1], 2.0)
    window.side.marks.undo_button.click()
    assert len(window.session.mark(window.current_mark).corners) == 1


def test_marking_without_a_current_mark_is_refused_by_name(window):
    window.current_mark = None
    window.set_mode_keep("opening")
    window.handle_click(10.0, 10.0, 2.0)
    assert "Новый проём" in window.status_label.text()


def test_opening_with_reveal_goes_end_to_end(window, scene):
    """Клики по углам и по кромке откоса → габарит и заглубление против истины сцены."""
    sc, _path, _base, _span, corners, side, inner = scene
    _opening(window, corners)
    window.side.marks.pick_reveal.click()
    for pt in inner:
        window.handle_click(*pt, 4.0)
    model = window.compute()
    element = next(e for e in model.elements if e.id == window.current_mark)
    op = sc.openings[0]
    assert abs(element.size_mm.width - op.width) <= element.size_mm.sigma_width
    assert abs(element.size_mm.height - op.height) <= element.size_mm.sigma_height
    assert element.recess.origin == "measured_from_reveal"
    assert element.recess.reveal_side == side
    assert abs(element.recess.value_mm - op.depth) <= element.recess.sigma_mm


def test_side_panel_is_usable_on_a_laptop_screen(qapp, scene):
    """На экране ноутбука 1366×768 поля колонки не сжимаются — колонка прокручивается.

    Колонка выше такого экрана; без прокрутки Qt ужимает поля ввода. Проверяется
    УСТАНОВИВШЕЕСЯ состояние: сразу после смены текста Qt на один оборот цикла
    событий показывает промежуточную вёрстку, и снимок экрана, сделанный в этот
    момент, однажды был принят за дефект.
    """
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox

    from facade_digitizer.ui.window import MainWindow

    _sc, path, base_px, span_mm, corners, _side, _inner = scene
    w = MainWindow()
    w.resize(1366, 768)
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    w.set_mode("base")
    for pt in base_px:
        w.handle_click(*pt, 1.0)
    w.side.base.set_values(span_mm=span_mm)
    w.apply_base()
    w.side.marks.new_mark.click()
    for pt in corners:
        w.handle_click(*pt, 2.0)
    w.compute()
    for _ in range(50):
        qapp.processEvents()
    squeezed = [(type(x).__name__, x.height())
                for x in w.side.findChildren(QAbstractSpinBox) + w.side.findChildren(QComboBox)
                if x.height() < x.minimumSizeHint().height()]
    window_height = w.height()
    w.close()
    assert squeezed == []
    # И окно умещается в экран: без прокрутки колонка не сжимается, а растягивает окно
    # выше экрана, и низ колонки уходит за край (проверено мутацией).
    assert window_height <= 768
