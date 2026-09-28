"""Окно приложения оператора. План 3, задача 1.

Тесты идут без дисплея (`QT_QPA_PLATFORM=offscreen`, выставляется в `conftest.py`).
Пропуск модуля допустим ТОЛЬКО при отсутствии пакета PySide6: `import PySide6` не
загружает QtWidgets, поэтому при установленном пакете без системных библиотек Qt
модуль не пропускается, а падает с `ImportError`, — так и должно быть (план 3,
глобальные ограничения). В CI пропуск запрещён переменной `REQUIRE_UI_TESTS`.
"""
import pytest

pytest.importorskip("PySide6")


def test_window_opens_and_can_be_grabbed(qapp):
    from facade_digitizer.ui.window import MainWindow

    w = MainWindow()
    w.resize(1280, 800)
    w.show()
    qapp.processEvents()
    shot = w.grab()
    assert (shot.width(), shot.height()) == (1280, 800)
    w.close()


def test_window_has_a_title_and_a_status_bar(qapp):
    from facade_digitizer.ui.window import MainWindow

    w = MainWindow()
    assert w.windowTitle()
    assert w.statusBar() is not None
    w.close()


def _wait(qapp, predicate, seconds=30.0):
    import time

    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    return predicate()


def test_opening_an_image_runs_the_frame_phase_in_background(qapp, tmp_path):
    """Открыть снимок → фаза кадра в фоне → кадр, пригодная зона, строка состояния."""
    import numpy as np

    from facade_digitizer.pipeline.io import save_image
    from facade_digitizer.ui.window import MainWindow
    from tests.test_synth import make_scene

    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    path = tmp_path / "фасад.png"
    save_image(path, sc.render())

    w = MainWindow()
    w.resize(1280, 800)
    w.show()
    w.open_image(path)
    assert "обрабатывается" in w.status_label.text()      # окно не замерло
    assert _wait(qapp, lambda: w.session.frame is not None)
    qapp.processEvents()

    assert w.canvas.frame_size == (5280, 3956)
    assert w.canvas.usable_item is not None                 # пригодная зона показана
    text = w.status_label.text()
    assert "доверие" in text and "поворот EXIF" in text and "полкруга" in text
    assert w.scale_label.text() == w.canvas.scale_text()
    w.canvas.one_to_one()
    qapp.processEvents()
    assert w.scale_label.text() == "1:1"
    assert np.isfinite(w.session.frame.sharpness)
    w.close()


def test_opening_a_frame_without_plane_says_so(qapp, tmp_path):
    import numpy as np

    from facade_digitizer.pipeline.io import save_image
    from facade_digitizer.ui.window import MainWindow

    rng = np.random.default_rng(0)
    path = tmp_path / "пусто.png"
    save_image(path, (128 + rng.normal(0, 3, (600, 800))).clip(0, 255).astype(np.uint8))
    w = MainWindow()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    qapp.processEvents()
    assert "плоскость не восстановлена" in w.status_label.text()
    assert w.canvas.usable_item is None
    w.close()
