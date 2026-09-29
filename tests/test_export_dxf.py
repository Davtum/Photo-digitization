"""Экспорт DXF. План 3, задача 18."""
import shlex

import ezdxf
import pytest

from tests import test_session_file as _sf
from tests.test_session_file import _session, _sigma

scene = _sf.scene


@pytest.fixture(scope="module")
def exported(scene, tmp_path_factory):
    from facade_digitizer.schema import FacadeModel
    from facade_digitizer.ui.session_file import export

    out = export(_session(scene), tmp_path_factory.mktemp("dxf"), _sigma)
    model = FacadeModel.model_validate_json(out.json_path.read_text(encoding="utf-8"))
    return out, model, ezdxf.readfile(out.dxf_path)


def _outlines(doc):
    from facade_digitizer.pipeline.export_dxf import read_xdata

    return {read_xdata(e)["id"]: e
            for e in doc.modelspace().query("LWPOLYLINE")
            if e.dxf.layer in ("WINDOW", "DOOR", "FACADE_BOUNDARY")}


def test_dxf_reopens_and_contains_every_element(exported):
    _out, model, doc = exported
    assert doc.header["$INSUNITS"] == 4                     # миллиметры
    outlines = _outlines(doc)
    assert set(outlines) == {e.id for e in model.elements}
    for element in model.elements:
        assert outlines[element.id].dxf.layer == element.class_name.upper()
    assert len(doc.modelspace().query('LWPOLYLINE[layer=="FACADE_EXTENT"]')) == 1


def test_dxf_coordinates_match_contour_mm(exported):
    _out, model, doc = exported
    outlines = _outlines(doc)
    for element in model.elements:
        entity = outlines[element.id]
        assert entity.closed
        points = [(x, y) for x, y, *_ in entity.get_points()]
        assert points == pytest.approx([tuple(p) for p in element.contour_mm], abs=1e-9)
        assert entity.dxf.elevation == 0.0


def test_dxf_keeps_sigma_as_xdata(exported):
    from facade_digitizer.pipeline.export_dxf import read_xdata

    _out, model, doc = exported
    outlines = _outlines(doc)
    for element in model.elements:
        x = read_xdata(outlines[element.id])
        assert x["sigma_width_mm"] == element.size_mm.sigma_width
        assert x["sigma_height_mm"] == element.size_mm.sigma_height
        assert x["scale_sigma_rel"] == model.facade.scale.sigma_rel
        assert x["origin"] == element.origin
        # Неизмеренное не пишется нулём.
        assert ("sigma_recess_mm" in x) == (element.recess is not None
                                            and element.recess.sigma_mm is not None)
        assert ("position_sigma_mm" in x) == (element.position_sigma_mm is not None)


def test_reveal_inner_contour_lies_at_the_recess_depth(exported):
    from facade_digitizer.pipeline.export_dxf import read_xdata

    _out, model, doc = exported
    with_recess = [e for e in model.elements if e.recess and e.recess.value_mm is not None]
    assert with_recess, "сцена обязана давать хотя бы один измеренный откос"
    reveals = {read_xdata(e)["id"]: e
               for e in doc.modelspace().query('LWPOLYLINE[layer=="REVEAL"]')}
    assert set(reveals) == {e.id for e in with_recess}
    for element in with_recess:
        inner = reveals[element.id]
        assert inner.dxf.elevation == pytest.approx(-element.recess.value_mm)
        assert read_xdata(inner)["sigma_recess_mm"] == element.recess.sigma_mm


def test_cli_dxf_flag_writes_the_same_drawing(exported, tmp_path, monkeypatch):
    from facade_digitizer.pipeline import run

    out, model, _doc = exported
    args = shlex.split(out.cli_line)[1:]
    assert "--dxf" in args
    args[args.index("--out-dir") + 1] = str(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert run.main(args) == 0
    cli_doc = ezdxf.readfile(tmp_path / out.dxf_path.name)
    cli = _outlines(cli_doc)
    assert set(cli) == {e.id for e in model.elements}
    for element in model.elements:
        points = [(x, y) for x, y, *_ in cli[element.id].get_points()]
        assert points == pytest.approx([tuple(p) for p in element.contour_mm], abs=1e-9)
