"""`scripts/demo_ui.py` проходит весь сценарий приёмки offscreen. План 3, задача 19."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def test_ui_demo_script_runs_end_to_end(qapp, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "demo_ui", Path(__file__).resolve().parents[1] / "scripts" / "demo_ui.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    assert demo.main(["--out-dir", str(tmp_path), "--pause", "0"]) == 0
    shots = sorted(p.name for p in (tmp_path / "shots").glob("*.png"))
    assert len(shots) == 7 and shots[-1] == "07-exported.png"
    out = json.loads((tmp_path / "facade_demo.json").read_text(encoding="utf-8"))
    element = out["elements"][0]
    # Истина сцены — 1460 × 1900 мм; расхождение в пределах σ.
    assert abs(element["size_mm"]["width"] - 1460.0) <= element["size_mm"]["sigma_width"]
    assert abs(element["size_mm"]["height"] - 1900.0) <= element["size_mm"]["sigma_height"]
    assert (tmp_path / "facade_demo.dxf").is_file()
    assert (tmp_path / "facade_demo.session.json").is_file()
    assert "facade-digitize" in capsys.readouterr().out
