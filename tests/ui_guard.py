"""Охрана от пропуска тестов интерфейса. План 3, задача 1.

При `REQUIRE_UI_TESTS=1` (ставится в CI) любой пропуск в модуле `test_ui_*` —
пропуск всего модуля через `importorskip` или пропуск изнутри теста — становится
падением. Без переменной поведение прежнее: на машине без PySide6 тесты
интерфейса честно пропускаются.

Хуки вынесены в модуль, а не записаны в `conftest.py`, чтобы их можно было
подключить во внутренний прогон `pytester` и проверить исполнением
(`tests/test_ui_guard.py`).
"""
import os

import pytest

ENV_FLAG = "REQUIRE_UI_TESTS"
UI_MODULE_PREFIX = "test_ui_"


def _required() -> bool:
    return os.environ.get(ENV_FLAG, "") not in ("", "0")


def _is_ui_node(nodeid: str) -> bool:
    module = nodeid.split("::", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
    return module.startswith(UI_MODULE_PREFIX)


def _reason(nodeid: str) -> str:
    return (f"{nodeid}: тест интерфейса пропущен при {ENV_FLAG}=1 — в этом окружении "
            "пропуск запрещён, потому что критерий «pytest зелёный» выполнялся бы "
            "при нуле прогнанных тестов интерфейса (план 3, задача 1)")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and _required() and _is_ui_node(report.nodeid):
        report.outcome = "failed"
        report.longrepr = _reason(report.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and _required() and _is_ui_node(report.nodeid):
        report.outcome = "failed"
        report.longrepr = _reason(report.nodeid)
