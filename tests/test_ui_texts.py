"""Тексты интерфейса оператора без Qt. Веб-интерфейс, задача 1.

Строки перенесены из окна Qt дословно: оператор, прошедший обучение на окне, и
выходной файл говорят одними словами.
"""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from facade_digitizer.ui import texts


def test_texts_module_does_not_import_qt():
    code = ("import sys, facade_digitizer.ui.texts; "
            "print(any(m.startswith('PySide6') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def _element(meets, edge="wall_plane"):
    return SimpleNamespace(meets_tolerance=meets, edge_reference=edge)


def test_tolerance_text_names_each_reason():
    assert texts.tolerance_text(_element(True), "ok") == "соответствует допуску п. 2.2"
    assert texts.tolerance_text(_element(False), "ok").startswith("не соответствует")
    assert "кромка вне плоскости стены" in texts.tolerance_text(_element(None, "unknown"), "ok")
    assert "кадр отбракован по разрешению" in texts.tolerance_text(_element(None), "reject")
    assert texts.tolerance_text(_element(None), "ok") == "не выпущено"


@pytest.mark.parametrize("scale, name", [(1.0, "1:1"), (2.0, "2:1"), (4.0, "4:1"),
                                         (0.5, "1:2"), (0.25, "1:4"), (1.5, "1.5:1")])
def test_scale_name(scale, name):
    assert texts.scale_name(scale) == name


def test_quality_notes_follow_the_qt_panel():
    assert "Вердикт по разрешению будет после ввода опорной базы" in texts.quality_note(
        "preliminary", "ok", confidence="доверие к плоскости 0.90")
    assert "Кадр отбракован по разрешению" in texts.quality_note("final", "reject")
    assert "опечатка в ней меняет его" in texts.quality_note("final", "ok")


def test_verdict_words_are_the_core_words():
    assert set(texts.VERDICT_TEXT) == {"ok", "degraded", "reject"}


def test_schema_choices_are_listed_with_russian_names():
    assert dict(texts.EDGES)["surround"] == "обрамление"
    assert dict(texts.MOUNTINGS)["embedded"] == "заглублён в проём"
    assert [k for k, _ in texts.SIDES] == ["left", "right", "top", "bottom"]
    assert dict(texts.CLASSES) == {"window": "окно", "door": "дверь"}
