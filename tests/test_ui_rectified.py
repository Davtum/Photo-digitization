"""Выровненный вид. План 3, задача 14."""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

FACADE_W = 20000.0


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
    path = tmp_path_factory.mktemp("растр") / "facade.png"
    save_image(path, sc.render())
    return path, sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]])), \
        sc.project(sc.openings[0].corners_mm())


@pytest.fixture
def window(qapp, scene_file):
    from facade_digitizer.ui.window import MainWindow

    path, base, corners = scene_file
    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    w.set_mode("base")
    for pt in base:
        w.handle_click(*pt, 1.0)
    w.side.base.set_values(span_mm=FACADE_W)
    w.apply_base()
    w.side.marks.new_mark.click()
    for pt in corners:
        w.handle_click(*pt, 2.0)
    assert _wait(qapp, lambda: w.rectified is not None)
    yield w
    w.close()


def test_mapping_matches_reference_values(window, scene_file):
    """Угол проёма, переведённый QTransform, — туда же, куда его переводит `homography.H`
    ВЫХОДНОГО ФАЙЛА. Опорное значение, а не «туда и обратно»: та проверка проходит
    и при транспонированной матрице (рецензия плана 3)."""
    from PySide6.QtCore import QPointF

    from facade_digitizer.geometry.homography import apply_homography

    _path, _base, corners = scene_file
    model = window.compute()
    written_H = np.array(model.images[0].homography["H"])
    expected = apply_homography(written_H, corners)
    for (x, y), (ex, ey) in zip(corners, expected):
        p = window.rect_view.to_raster.map(QPointF(x, y))
        assert (p.x(), p.y()) == pytest.approx((ex, ey), abs=0.01)


def test_row_major_transform_would_be_caught(window, scene_file):
    """Мутация, от которой охраняет транспонирование: построчная H промахивается."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QTransform

    _path, _base, corners = scene_file
    H = window.rect_view.H
    wrong = QTransform(*H.ravel())
    good = window.rect_view.to_raster.map(QPointF(*corners[0]))
    bad = wrong.map(QPointF(*corners[0]))
    assert abs(bad.x() - good.x()) + abs(bad.y() - good.y()) > 10.0


def test_origin_lands_on_origin_rect_px(window):
    from PySide6.QtCore import QPointF

    origin_frame = window.session.reference.ends[0].xy
    p = window.rect_view.to_raster.map(QPointF(*origin_frame))
    assert (p.x(), p.y()) == pytest.approx(window.rectified.origin_rect_px, abs=0.01)


def test_click_on_rectified_view_centres_the_frame_and_adds_no_point(qapp, window, scene_file):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _path, _base, corners = scene_file
    before = [len(m.corners) for m in window.session.marks]
    window.canvas.set_view_scale(2.0)
    target = corners.mean(axis=0)
    vp = window.rect_view.mapFromScene(window.rect_view.frame_to_scene(*target))
    # Точка кадра, в которую клик попал на самом деле: вид вписан в окно, и один его
    # экранный пиксель стоит нескольких пикселей растра — это свойство клика по
    # уменьшенному виду, а не центрирования. Проверяется центрирование.
    hit = window.rect_view.scene_to_frame(window.rect_view.mapToScene(vp))
    assert hit == pytest.approx(tuple(target), abs=15.0)
    QTest.mouseClick(window.rect_view.viewport(), Qt.LeftButton, pos=vp)
    qapp.processEvents()
    centre = window.canvas.mapToScene(window.canvas.viewport().rect().center())
    assert (centre.x() - 0.5, centre.y() - 0.5) == pytest.approx(hit, abs=1.0)
    assert [len(m.corners) for m in window.session.marks] == before


def test_invalid_area_is_hatched_and_coverage_is_named(window):
    view = window.rect_view
    assert view.invalid_item is not None
    assert 0.5 < view.coverage < 1.0                         # косой кадр: углы пустые
    assert f"{view.coverage:.0%}" in window.rect_label.text()
    img = view.invalid_item.pixmap().toImage()
    mask = np.asarray(window.rectified.valid_mask)
    ys, xs = np.nonzero(~mask)
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    assert img.pixelColor(x, y).alpha() > 0                  # вне охвата — штриховка
    ys, xs = np.nonzero(mask)
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    assert img.pixelColor(x, y).alpha() == 0                 # в охвате — фасад как есть
