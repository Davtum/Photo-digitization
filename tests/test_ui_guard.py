"""Запрет пропуска тестов интерфейса в CI. План 3, задача 1.

Без этой охраны критерий «pytest зелёный» выполнялся бы при нуле прогнанных тестов
интерфейса: модуль `test_ui_*`, пропущенный через `importorskip`, выглядит в сводке
безобидной буквой `s`. Проверяется на внутреннем прогоне pytest (`pytester`), а не
чтением `conftest.py`: охрана, которую не запускали, есть надежда.
"""

_INNER_CONFTEST = "from tests.ui_guard import *  # noqa: F401,F403\n"

_SKIPPED_UI_MODULE = """
import pytest
pytest.importorskip("module_that_does_not_exist_anywhere")

def test_never_runs():
    pass
"""

_SKIPPED_UI_TEST = """
import pytest

def test_skips_itself():
    pytest.skip("пропуск изнутри теста")
"""

_SKIPPED_CORE_TEST = """
import pytest

def test_core_skip_is_allowed():
    pytest.skip("пропуск вне интерфейса")
"""


def _run(pytester, monkeypatch, files: dict, require: bool):
    if require:
        monkeypatch.setenv("REQUIRE_UI_TESTS", "1")
    else:
        monkeypatch.delenv("REQUIRE_UI_TESTS", raising=False)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(**files)
    return pytester.runpytest_inprocess("-q", "-p", "no:cacheprovider")


def test_module_level_skip_of_ui_tests_is_a_skip_without_the_flag(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, {"test_ui_fake": _SKIPPED_UI_MODULE}, require=False)
    result.assert_outcomes(skipped=1)


def test_module_level_skip_of_ui_tests_fails_with_the_flag(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, {"test_ui_fake": _SKIPPED_UI_MODULE}, require=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*REQUIRE_UI_TESTS*"])


def test_in_test_skip_of_ui_tests_fails_with_the_flag(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, {"test_ui_fake": _SKIPPED_UI_TEST}, require=True)
    result.assert_outcomes(failed=1)


def test_skips_outside_ui_tests_are_untouched_by_the_flag(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, {"test_core_fake": _SKIPPED_CORE_TEST}, require=True)
    result.assert_outcomes(skipped=1)


def test_guard_module_is_importable():
    from tests import ui_guard

    assert callable(ui_guard.pytest_runtest_makereport)
    assert callable(ui_guard.pytest_make_collect_report)
