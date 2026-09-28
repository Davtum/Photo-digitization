"""Просмотрщик кадра. План 3, задача 9.

Соглашение о пикселях: у Qt пиксель `i` занимает `[i, i+1)` сцены, у конвейера (OpenCV)
центр пикселя `i` — в целой координате `i`, и допустимая точка лежит в `[0, w−1]`.
Перевод сцена → кадр — `x − 0.5`; клик в любую часть пикселя снимка даёт точку
этого пикселя (края прижимаются к `[0, w−1]`), клик вне снимка точкой не становится.
"""
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

W, H = 400, 300


def _frame(w=W, h=H):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return img


@pytest.fixture
def canvas(qapp):
    from facade_digitizer.ui.canvas import FrameCanvas

    c = FrameCanvas()
    c.resize(800, 600)
    c.set_frame(_frame())
    c.show()
    qapp.processEvents()
    yield c
    c.close()


def _viewport_point(canvas, frame_xy):
    """Точка вьюпорта, в которую отображается точка кадра — через сцену Qt."""
    from PySide6.QtCore import QPointF

    x, y = frame_xy
    return canvas.mapFromScene(QPointF(x + 0.5, y + 0.5))


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_screen_point_maps_to_frame_pixel_at_several_scales(canvas, qapp, scale):
    canvas.set_view_scale(scale)
    canvas.centerOn(W / 2, H / 2)
    qapp.processEvents()
    for target in [(10.0, 20.0), (200.0, 150.0), (390.0, 290.0)]:
        vp = _viewport_point(canvas, target)
        got = canvas.frame_point_at(vp)
        if got is None:                    # точка не в видимой части при сильном увеличении
            continue
        # Экранный пиксель стоит 1/s пикселей кадра: точнее перевести нельзя.
        assert got[0] == pytest.approx(target[0], abs=max(1.0 / scale, 0.5) + 1e-6)
        assert got[1] == pytest.approx(target[1], abs=max(1.0 / scale, 0.5) + 1e-6)


def test_mapping_survives_panning(canvas, qapp):
    canvas.set_view_scale(4.0)
    canvas.centerOn(50, 60)
    qapp.processEvents()
    vp = _viewport_point(canvas, (52.0, 61.0))
    assert canvas.frame_point_at(vp) == pytest.approx((52.0, 61.0), abs=0.25)


def test_view_scale_is_reported_and_matches_the_transform(canvas):
    canvas.set_view_scale(2.5)
    assert canvas.view_scale() == pytest.approx(2.5)
    assert canvas.transform().m11() == pytest.approx(2.5)
    assert "2.5:1" in canvas.scale_text()
    canvas.set_view_scale(0.25)
    assert "1:4" in canvas.scale_text()


def test_one_to_one_action_sets_scale_exactly_one(canvas):
    canvas.fit_to_window()
    assert canvas.view_scale() != 1.0
    canvas.one_to_one()
    assert canvas.view_scale() == 1.0


def test_half_pixel_convention_matches_the_pipeline(canvas):
    """Центр пикселя (i, j) сцены Qt — точка (i, j) кадра конвейера, ровно."""
    from PySide6.QtCore import QPointF

    assert canvas.scene_to_frame(QPointF(10.5, 20.5)) == (10.0, 20.0)
    assert canvas.scene_to_frame(QPointF(0.0, 0.0)) == (0.0, 0.0)      # край прижат


def test_click_on_the_last_column_is_accepted_and_beyond_refused(canvas):
    from PySide6.QtCore import QPointF

    inside_last = canvas.scene_to_frame(QPointF(W - 0.01, H - 0.01))
    assert inside_last == (W - 1.0, H - 1.0)            # в пределах [0, w−1] конвейера
    assert canvas.scene_to_frame(QPointF(W + 0.5, 10.0)) is None
    assert canvas.scene_to_frame(QPointF(-0.5, 10.0)) is None


def test_click_emits_frame_point_with_view_scale(canvas, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    canvas.set_view_scale(2.0)
    canvas.centerOn(100, 100)
    qapp.processEvents()
    got = []
    canvas.pointClicked.connect(lambda x, y, s: got.append((x, y, s)))
    QTest.mouseClick(canvas.viewport(), Qt.LeftButton, pos=_viewport_point(canvas, (100, 100)))
    assert len(got) == 1
    x, y, s = got[0]
    assert (x, y) == pytest.approx((100.0, 100.0), abs=0.5)
    assert s == 2.0


def test_click_outside_the_image_emits_nothing(canvas, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    canvas.set_view_scale(0.5)
    canvas.centerOn(W / 2, H / 2)
    qapp.processEvents()
    got = []
    canvas.pointClicked.connect(lambda *a: got.append(a))
    QTest.mouseClick(canvas.viewport(), Qt.LeftButton, pos=QPoint(2, 2))   # серое поле
    assert got == []


def test_usable_overlay_can_be_shown_and_hidden(canvas):
    mask = np.zeros((H, W), bool)
    mask[:, : W // 2] = True
    canvas.set_usable_mask(mask)
    assert canvas.usable_item is not None and canvas.usable_item.isVisible()
    canvas.set_usable_mask(None)
    assert canvas.usable_item is None or not canvas.usable_item.isVisible()


def test_background_job_runs_off_the_ui_thread(qapp):
    from facade_digitizer.ui.worker import run_in_background

    main = threading.get_ident()
    done = []
    job = run_in_background(threading.get_ident, on_done=done.append,
                            on_error=lambda msg: done.append(("ошибка", msg)))
    deadline = time.monotonic() + 5
    while not done and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done and done[0] != main
    assert job.finished


def test_background_job_reports_errors_as_text(qapp):
    from facade_digitizer.ui.worker import run_in_background

    def fail():
        raise ValueError("плоскость не восстановлена")

    got = []
    run_in_background(fail, on_done=got.append, on_error=lambda m: got.append(("err", m)))
    deadline = time.monotonic() + 5
    while not got and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert got == [("err", "плоскость не восстановлена")]


def test_cancelled_job_does_not_deliver(qapp):
    """Отмена: результат отброшен, окно свободно сразу. Прервать расчёт OpenCV
    изнутри нельзя — отмена отказывается от результата, а не останавливает его."""
    from facade_digitizer.ui.worker import run_in_background

    got = []
    job = run_in_background(lambda: time.sleep(0.3) or "готово", on_done=got.append,
                            on_error=got.append)
    job.cancel()
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert got == [] and job.cancelled
