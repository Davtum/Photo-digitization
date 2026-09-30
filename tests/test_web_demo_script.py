"""`scripts/demo_ui.py` проходит весь сценарий приёмки через веб-интерфейс без браузера.

План 3, задача 19; веб-интерфейс, задача 10. Сценарий идёт через `Workbench` и
`Desk` — ту же логику, что стоит за страницей, — и сверяет итог с истиной сцены.
"""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


def _demo():
    spec = importlib.util.spec_from_file_location(
        "demo_ui", Path(__file__).resolve().parents[1] / "scripts" / "demo_ui.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


def test_web_demo_script_runs_end_to_end(tmp_path, capsys):
    assert _demo().main(["--out-dir", str(tmp_path)]) == 0
    export = tmp_path / "экспорт"
    out = json.loads((export / "facade_demo.json").read_text(encoding="utf-8"))
    element = out["elements"][0]
    # Истина сцены — 1460 × 1900 мм; расхождение в пределах σ.
    assert abs(element["size_mm"]["width"] - 1460.0) <= element["size_mm"]["sigma_width"]
    assert abs(element["size_mm"]["height"] - 1900.0) <= element["size_mm"]["sigma_height"]
    assert (export / "facade_demo.dxf").is_file()
    session = json.loads((tmp_path / "facade_demo.session.json").read_text(encoding="utf-8"))
    assert session["marks"][0]["corners"][0]["view_scale"] == 3.0
    printed = capsys.readouterr().out
    assert "facade-digitize" in printed and "Истина сцены" in printed
