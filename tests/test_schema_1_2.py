"""Схема 1.2: σ и идентификатор у разметки, σ у каждого конца базы. План 3, задача 6.

Три дефекта ядра, найденных рецензией плана 3 прототипом:
* σ клика одна на весь набор элементов (`digitize_elements(..., sigma_px: float)`),
  хотя разные проёмы размечаются при разном увеличении;
* идентификатор элемента — номер в списке: удаление любой разметки сдвигает
  идентификаторы всех следующих;
* у двух концов опорной базы одна σ, хотя они указываются при разном масштабе.
"""
import json
import math

import numpy as np
import pytest

from facade_digitizer import schema
from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.elements import ElementMark, parse_marks
from facade_digitizer.pipeline.io import save_image
from tests.test_synth import make_scene

VIEW = {"dx": 3000.0, "dy": -2000.0, "dist": 20000.0, "depth": 150.0}
FACADE_W = 20000.0
RASTER = 10.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("схема")
    sc = make_scene(**VIEW)
    path = d / "facade.png"
    save_image(path, sc.render())
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    reference = run.OperatorReference(origin_px=tuple(base[0]),
                                      span_px=(tuple(base[0]), tuple(base[1])),
                                      span_mm=FACADE_W)
    corners = [[float(u), float(v)] for u, v in sc.project(sc.openings[0].corners_mm())]
    fs = run.frame_stage(path)
    ss = run.scale_stage(fs, reference)
    geometry = run.raster_geometry_for(fs, ss, RASTER)
    return sc, path, reference, corners, fs, ss, geometry


def _mark(corners, **extra):
    return ElementMark.model_validate({"class": "window", "mounting": "embedded",
                                       "edge_type": "sharp_wall_edge",
                                       "corners_px": corners, **extra})


def test_schema_version_is_1_2(scene):
    *_, fs, ss, geometry = scene
    model = run.elements_stage(fs, ss, geometry, None)
    assert model.schema_version == "1.2" == schema.SCHEMA_VERSION


def test_each_mark_keeps_its_own_sigma(scene):
    _sc, _path, _ref, corners, fs, ss, geometry = scene
    fine = _mark(corners, id="fine", sigma_px=1.0)
    coarse = _mark(corners, id="coarse", sigma_px=3.0)
    model = run.elements_stage(fs, ss, geometry, [fine, coarse])
    a, b = model.elements
    assert a.contour_px[0]["sigma_px"] == 1.0
    assert b.contour_px[0]["sigma_px"] == 3.0
    assert b.size_mm.sigma_width > a.size_mm.sigma_width + 1.0
    # Грубый проём не портит σ соседа: её значение — как у одиночного точного.
    alone = run.elements_stage(fs, ss, geometry, [fine]).elements[0]
    assert a.size_mm.sigma_width == alone.size_mm.sigma_width


def test_mark_without_sigma_takes_the_call_default(scene):
    _sc, _path, _ref, corners, fs, ss, geometry = scene
    model = run.elements_stage(fs, ss, geometry, [_mark(corners)], mark_sigma_px=2.5)
    assert model.elements[0].contour_px[0]["sigma_px"] == 2.5


def test_element_ids_survive_deleting_an_earlier_mark(scene):
    _sc, _path, _ref, corners, fs, ss, geometry = scene
    marks = [_mark(corners, id=name) for name in ("w_a", "w_b", "w_c")]
    before = [e.id for e in run.elements_stage(fs, ss, geometry, marks).elements]
    after = [e.id for e in run.elements_stage(fs, ss, geometry, marks[1:]).elements]
    assert before == ["w_a", "w_b", "w_c"]
    assert after == ["w_b", "w_c"]


def test_duplicate_element_ids_are_refused_by_name(scene):
    _sc, _path, _ref, corners, fs, ss, geometry = scene
    with pytest.raises(ValueError, match="w_dup"):
        run.elements_stage(fs, ss, geometry,
                           [_mark(corners, id="w_dup"), _mark(corners, id="w_dup")])
    # Явный id, совпавший с присвоенным по номеру, — тоже повтор, а не перезапись.
    with pytest.raises(ValueError, match="w_001"):
        run.elements_stage(fs, ss, geometry, [_mark(corners, id="w_001"), _mark(corners)])


def test_non_positive_mark_sigma_is_refused():
    with pytest.raises(ValueError, match="sigma_px"):
        ElementMark.model_validate({"class": "window", "mounting": "embedded",
                                    "edge_type": "sharp_wall_edge",
                                    "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                    "sigma_px": 0.0})


def test_base_ends_with_different_sigma_combine_in_quadrature(scene):
    """σ базы — √((σ1·g1)² + (σ2·g2)²) / L, а не √2·σ·g / L с одной σ на оба конца."""
    _sc, _path, reference, _corners, fs, _ss, _geometry = scene
    equal = run.scale_stage(fs, reference)
    ends = run.OperatorReference(origin_px=reference.origin_px, span_px=reference.span_px,
                                 span_mm=reference.span_mm,
                                 end_sigma_px=(reference.sigma_px, reference.sigma_px))
    assert run.scale_stage(fs, ends).sigma_rel == pytest.approx(equal.sigma_rel, rel=1e-12)

    uneven = run.OperatorReference(origin_px=reference.origin_px,
                                   span_px=reference.span_px, span_mm=reference.span_mm,
                                   end_sigma_px=(1.0, 6.0))
    got = run.scale_stage(fs, uneven)
    g1, g2 = (run._gsd_at(got.fields.gsd, fs.image_size, p) for p in reference.span_px)
    expected = math.hypot(1.0 * g1, 6.0 * g2) / reference.span_mm
    assert got.sigma_rel == pytest.approx(expected, rel=1e-12)


def test_reveal_mark_carries_its_own_sigma():
    raw = [{"class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
            "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "reveal": {"side": "left", "inner_edge_px": [[0, 0], [0, 1]], "sigma_px": 1.7}}]
    assert parse_marks(raw)[0].reveal.sigma_px == 1.7


def test_cli_marks_json_carries_sigma_and_id(scene, tmp_path):
    _sc, path, reference, corners, *_ = scene
    marks = tmp_path / "marks.json"
    marks.write_text(json.dumps([{"class": "window", "mounting": "embedded",
                                  "edge_type": "sharp_wall_edge", "corners_px": corners,
                                  "id": "окно_1", "sigma_px": 2.0}], ensure_ascii=False),
                     encoding="utf-8")
    (x0, y0), (x1, y1) = reference.span_px
    code = run.main([str(path), "--raster-mm-per-px", str(RASTER),
                     "--origin-px", str(x0), str(y0),
                     "--span-px", str(x0), str(y0), str(x1), str(y1),
                     "--span-mm", str(FACADE_W), "--marks", str(marks),
                     "--out-dir", str(tmp_path)])
    assert code == 0
    out = json.loads((tmp_path / "facade.json").read_text(encoding="utf-8"))
    element = out["elements"][0]
    assert element["id"] == "окно_1"
    assert element["contour_px"][0]["sigma_px"] == 2.0
    assert out["schema_version"] == "1.2"


def test_new_fields_of_later_tasks_are_null_until_filled(scene):
    """Поля поздних задач плана существуют; пусты, пока их нечем заполнить."""
    _sc, _path, _ref, corners, fs, ss, geometry = scene
    model = run.elements_stage(fs, ss, geometry, [_mark(corners)])
    element = model.elements[0]
    assert element.position_sigma_mm > 0                 # задача 21: сводит assemble
    assert element.relative_position_sigma_mm is None    # один элемент — соседа нет
    assert element.operator is None                      # задача 22
    assert model.facade.scale.span_sigma_mm is None      # задача 11


def test_schema_1_1_files_are_refused_or_migrated_explicitly(scene):
    *_, fs, ss, geometry = scene
    payload = json.loads(run.elements_stage(fs, ss, geometry, None).model_dump_json())
    payload["schema_version"] = "1.1"
    with pytest.raises(ValueError, match="schema_version"):
        schema.FacadeModel.model_validate(payload)
    migrated = schema.migrate_1_1_to_1_2(payload)
    assert schema.FacadeModel.model_validate(migrated).schema_version == "1.2"
    assert payload["schema_version"] == "1.1"          # исходник не тронут
    with pytest.raises(ValueError, match="1.1"):
        schema.migrate_1_1_to_1_2({"schema_version": "1.0"})
