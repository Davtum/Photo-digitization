"""Входные файлы ручной приёмки. План 3, задача 19."""
import importlib.util
import json
from pathlib import Path

import piexif


def test_acceptance_inputs_cover_every_checklist_scenario(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "make_acceptance_inputs",
        Path(__file__).resolve().parents[1] / "scripts" / "make_acceptance_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--out-dir", str(tmp_path)]) == 0
    names = {p.name for p in tmp_path.iterdir()}
    assert {"01_фасад.jpg", "02_фасад_поворот_EXIF.jpg", "03_фасад_далеко.jpg",
            "04_без_линий.jpg", "профиль_DEMO-CAM.json", "профиль_чужой.json",
            "ИСТИНА.txt"} <= names
    exif = piexif.load(str(tmp_path / "02_фасад_поворот_EXIF.jpg"))
    assert exif["0th"][piexif.ImageIFD.Orientation] == 6
    assert exif["0th"][piexif.ImageIFD.Model] == b"DEMO-CAM"
    own = json.loads((tmp_path / "профиль_DEMO-CAM.json").read_text(encoding="utf-8"))
    foreign = json.loads((tmp_path / "профиль_чужой.json").read_text(encoding="utf-8"))
    assert own["model"] == "DEMO-CAM" and foreign["model"] != own["model"]
