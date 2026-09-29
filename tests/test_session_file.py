"""Файл сессии и экспорт JSON. План 3, задача 17."""
import json
import shlex
import shutil
import sys

import numpy as np
import pytest

from tests.test_ui_marks import VIEW


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sc = make_scene(**VIEW)
    path = tmp_path_factory.mktemp("сессия") / "фасад.png"
    save_image(path, sc.render())
    sys.path.insert(0, "scripts")
    from demo_synthetic import operator_base

    base_px, span_mm, _origin = operator_base(sc)
    op = sc.openings[0]
    corners = sc.project(op.corners_mm())
    side, _h = visible_reveal_side(sc.camera_on_plane(), op.x, op.x + op.width,
                                   op.y, op.y + op.height)
    x = op.x if side == "left" else op.x + op.width
    inner = sc.project(np.array([[x, op.y], [x, op.y + op.height]]), depth=op.depth)
    return sc, path, base_px, span_mm, corners, side, inner


def _sigma(point):
    from facade_digitizer.ui.zoom import click_sigma

    return click_sigma(point)


def _session(scene, *, manual=False):
    from facade_digitizer.ui.session import ClickedPoint, OperatorSession

    sc, path, base_px, span_mm, corners, side, inner = scene
    s = OperatorSession()
    s.open_image(path)
    if manual:
        op = sc.openings[0]
        for x, y in corners:
            s.add_plane_point(ClickedPoint(x, y, 3.0))
        s.set_plane_override(s.manual_plane(aspect_ratio=op.width / op.height))
    assert s.compute_frame() is not None, s.error
    for (x, y), scale in zip(base_px, (1.0, 0.5)):
        s.add_reference_end(ClickedPoint(x, y, scale))
    s.set_span_mm(span_mm)
    s.set_base_options(span_sigma_mm=4.0, scale_source="operator_reference",
                       origin_is_facade_corner=False)
    first = s.new_mark()
    for (x, y), scale in zip(corners, (2.0, 3.0, 4.0, 2.0)):
        s.add_corner(first, ClickedPoint(x, y, scale))
    for x, y in inner:
        s.add_reveal_point(first, side, ClickedPoint(x, y, 4.0))
    dropped = s.new_mark("door")
    s.delete_mark(dropped.id)
    second = s.new_mark()
    second.mounting, second.edge_type = "flush", "surround"
    for x, y in corners:
        s.add_corner(second, ClickedPoint(x + 25.0, y - 10.0, 2.0))
    s.timing = {"marking_s": 42.5}
    return s


def test_session_round_trip_preserves_clicks_scales_and_plane(tmp_path, scene):
    from facade_digitizer.ui.session_file import load_session, save_session

    s = _session(scene, manual=True)
    loaded = load_session(save_session(s, tmp_path / "работа.session.json"))
    assert loaded.image_path == s.image_path and loaded.image_hash == s.image_hash
    assert loaded.reference == s.reference
    assert loaded.marks == s.marks
    assert loaded.plane_points == s.plane_points
    assert np.array_equal(loaded.plane_override.image_pts, s.plane_override.image_pts)
    assert loaded.plane_override.aspect_ratio == s.plane_override.aspect_ratio
    assert loaded.timing == {"marking_s": 42.5}
    # И считается загруженная сессия в ту же модель.
    assert loaded.compute_frame() is not None
    assert loaded.compute_elements(_sigma) == s.compute_elements(_sigma)
    # Номер удалённой разметки не переиспользуется и после загрузки.
    assert loaded.new_mark().id not in {m.id for m in s.marks} | {"d_001"}


def test_session_with_other_image_or_profile_is_refused_by_name(tmp_path, scene):
    from facade_digitizer.ui.session_file import load_session, save_session

    _sc, path, *_ = scene
    image = tmp_path / "снимок.png"
    shutil.copy(path, image)
    s = _session(scene)
    s.image_path = image.resolve()
    saved = save_session(s, tmp_path / "a.session.json")
    image.write_bytes(image.read_bytes() + b"\0")          # тот же путь, другой файл
    with pytest.raises(ValueError, match="снимок снимок.png не тот"):
        load_session(saved)

    raw = json.loads(saved.read_text(encoding="utf-8"))
    raw["image"] = {"path": str(path), "sha256": s.image_hash}
    profile = tmp_path / "профиль.json"
    profile.write_text("{}", encoding="utf-8")
    raw["profile"] = {"path": str(profile), "sha256": "0" * 64}
    (tmp_path / "b.session.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="профиль калибровки профиль.json не тот"):
        load_session(tmp_path / "b.session.json")


def test_export_is_a_valid_schema_1_2_model(tmp_path, scene):
    from facade_digitizer.schema import SCHEMA_VERSION, FacadeModel
    from facade_digitizer.ui.session_file import export

    s = _session(scene)
    out = export(s, tmp_path / "выход", _sigma)
    model = FacadeModel.model_validate_json(out.json_path.read_text(encoding="utf-8"))
    assert model.schema_version == SCHEMA_VERSION == "1.2"
    assert [e.id for e in model.elements] == [m.id for m in s.marks]
    marks = json.loads(out.marks_path.read_text(encoding="utf-8"))
    assert [m["id"] for m in marks] == [m.id for m in s.marks]
    assert all(m["sigma_px"] > 0 for m in marks)
    assert marks[0]["reveal"]["sigma_px"] > 0
    assert (tmp_path / "выход" / "фасад.command.txt").read_text(
        encoding="utf-8").strip() == out.cli_line


@pytest.mark.parametrize("manual", [False, True], ids=["авто", "ручная_плоскость"])
def test_export_and_cli_produce_the_same_json(tmp_path, scene, monkeypatch, manual):
    from facade_digitizer.pipeline import run
    from facade_digitizer.ui.session_file import export

    s = _session(scene, manual=manual)
    s.reference.origin_is_facade_corner = manual       # обе ветви начала отсчёта
    out = export(s, tmp_path / "из_окна", _sigma)
    args = shlex.split(out.cli_line)[1:]
    assert args == out.cli_args
    cli_dir = tmp_path / "из_cli"
    args[args.index("--out-dir") + 1] = str(cli_dir)
    # CLI запускается из другого каталога: пути в строке абсолютные.
    monkeypatch.chdir(tmp_path)
    assert run.main(args) == 0
    assert (cli_dir / out.json_path.name).read_bytes() == out.json_path.read_bytes()


def test_session_module_does_not_import_qt():
    import subprocess

    code = ("import sys, facade_digitizer.ui.session_file; "
            "sys.exit(any(m.startswith('PySide6') for m in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_relative_image_path_is_resolved(tmp_path, scene, monkeypatch):
    """`images[].path` пишется как задан: относительный путь из окна против
    абсолютного у CLI дал бы разный JSON."""
    from facade_digitizer.ui.session import OperatorSession

    _sc, path, *_ = scene
    monkeypatch.chdir(path.parent)
    s = OperatorSession()
    s.open_image(path.name)
    assert s.image_path == path.resolve()
