"""Общая обвязка тестов: охрана от пропуска тестов интерфейса (`tests/ui_guard.py`)."""
from tests.ui_guard import pytest_make_collect_report, pytest_runtest_makereport  # noqa: F401
