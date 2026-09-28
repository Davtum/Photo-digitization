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
