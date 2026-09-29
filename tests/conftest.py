"""Общая обвязка тестов.

`QT_QPA_PLATFORM=offscreen` выставляется до первого импорта Qt: тесты интерфейса
идут без дисплея, в том числе в CI. Уже заданное значение не перекрывается —
разработчик вправе смотреть на окно, запустив тесты с `QT_QPA_PLATFORM=xcb`.
"""
import os

import pytest

from tests.ui_guard import pytest_make_collect_report, pytest_runtest_makereport  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Один `QApplication` на весь прогон: второй экземпляр в процессе роняет Qt.

    Импорт обычный, а не `importorskip`: пакет PySide6 уже проверен в модуле теста,
    и если при нём не грузится QtWidgets (нет системных библиотек Qt), это отказ
    окружения, который обязан быть виден, а не пропуском (план 3, глобальные
    ограничения).
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _release_windows():
    """После каждого теста — удалить окна верхнего уровня и собрать мусор.

    Окно с открытым снимком держит цветной кадр 5280×3956, серый кадр, растр и их
    пиксмапы — около 300 МБ. `close()` окно лишь прячет, а Python-обёртку держат
    циклы ссылок (слоты-лямбды на `self`), и без явного удаления прогон всего
    набора дорастал до 14 ГБ и убивался по памяти (план 3, задача 19).
    """
    yield
    import sys

    if "PySide6.QtWidgets" not in sys.modules:
        return
    import gc

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    gc.collect()
