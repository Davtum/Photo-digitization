"""Проверка по измеренным размерам на синтетике. План 3, задача 27."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from facade_digitizer.training import make_block, render

_spec = importlib.util.spec_from_file_location(
    "verify_measured", Path(__file__).resolve().parents[1] / "scripts" / "verify_measured.py")
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


@pytest.fixture(scope="module")
def measured_scene(tmp_path_factory):
    """Сцена обучения через конвейер + «измерения заказчика» из истины сцены."""
    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline import run
    from facade_digitizer.pipeline.elements import ElementMark
    from facade_digitizer.pipeline.io import save_image

    d = tmp_path_factory.mktemp("измерения")
    spec = make_block(1)[0]
    sc = spec.scene()
    path = d / "фасад.png"
    save_image(path, render(spec, 3))
    first = sc.openings[0]
    base = sc.project(np.array([[first.x, first.y], [first.x + first.width, first.y]]))
    ref = run.OperatorReference(origin_px=tuple(base[0]),
                                span_px=(tuple(base[0]), tuple(base[1])),
                                span_mm=first.width, sigma_px=1.14)
    cam = sc.camera_on_plane()
    marks = []
    for op in sc.openings:
        payload = {"class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
                   "corners_px": sc.project(op.corners_mm()).tolist(), "sigma_px": 1.14}
        side, _ = visible_reveal_side(cam, op.x, op.x + op.width, op.y, op.y + op.height)
        if side in ("left", "right"):
            x = op.x if side == "left" else op.x + op.width
            inner = sc.project(np.array([[x, op.y], [x, op.y + op.height]]), depth=op.depth)
            payload["reveal"] = {"side": side, "inner_edge_px": inner.tolist(),
                                 "sigma_px": 1.14}
        marks.append(ElementMark.model_validate(payload))
    model = run.process(path, operator_reference=ref, raster_mm_per_px=5.0, marks=marks)
    out = d / "фасад.json"
    out.write_text(model.model_dump_json(), encoding="utf-8")
    rows = [{"image": "фасад", "element_id": "", "quantity": "base",
             "value_mm": first.width, "sigma_ref_mm": 1.0, "is_base": 1, "note": "база"}]
    for element, op in zip(model.elements, sc.openings, strict=True):
        truths = {"width": op.width, "height": op.height,
                  "x": op.x - first.x, "y": op.y - first.y, "recess": op.depth}
        for quantity, value in truths.items():
            rows.append({"image": "фасад", "element_id": element.id, "quantity": quantity,
                         "value_mm": value, "sigma_ref_mm": 2.0, "is_base": 0, "note": ""})
    table = d / "измерения.csv"
    with open(table, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return table, out, rows


def test_base_row_is_excluded_and_groups_are_separate(measured_scene):
    table, out, rows = measured_scene
    found, skipped = verify.pairs(verify.read_table(table),
                                  {"фасад": json.loads(out.read_text(encoding="utf-8"))})
    checked = [r for r in rows if not r["is_base"]]
    assert len(found) + len(skipped) == len(checked)
    assert {p.group for p in found} >= {"габарит", "положение"}
    sizes = verify.summary([p for p in found if p.group == "габарит"])
    assert abs(sizes["bias"]) < 5.0 and sizes["coverage_2"] >= 0.8


def test_script_prints_first_figures_not_a_tolerance_check(measured_scene, capsys):
    table, out, _rows = measured_scene
    assert verify.main([str(table), str(out)]) == 0
    text = capsys.readouterr().out
    assert "НЕ проверка статистического допуска" in text
    assert "смещение" in text and "покрытие 1σ" in text and "мало для статистики" in text


def test_table_without_base_row_is_refused(measured_scene, tmp_path, capsys):
    _table, out, rows = measured_scene
    bad = tmp_path / "без_базы.csv"
    with open(bad, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([r for r in rows if not r["is_base"]])
    assert verify.main([str(bad), str(out)]) == 1
    assert "нет строки опорной базы" in capsys.readouterr().err
