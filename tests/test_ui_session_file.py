"""Сессия и экспорт из окна. План 3, задача 17."""
import pytest

pytest.importorskip("PySide6")

from tests import test_session_file as _sf
from tests.test_session_file import _session, _sigma
from tests.test_ui_marks import _wait

scene = _sf.scene


def test_window_restores_a_saved_session_and_exports_the_same_json(qapp, tmp_path, scene):
    from facade_digitizer.ui.session_file import export, save_session
    from facade_digitizer.ui.window import MainWindow

    s = _session(scene, manual=True)
    reference = export(s, tmp_path / "эталон", _sigma)
    saved = save_session(s, tmp_path / "работа.session.json")

    w = MainWindow()
    w.show()
    w.open_session(saved)
    assert _wait(qapp, lambda: w.session.model is not None), w.status_label.text()
    assert w.session.frame.plane.method == "manual_four_point"
    assert w.current_mark == s.marks[-1].id
    assert w.side.result.table.rowCount() == len(s.marks)
    assert w.canvas.overlays.get(f"mark:{s.marks[0].id}")
    out = w.export(tmp_path / "из_окна")
    w.close()
    assert out.json_path.read_bytes() == reference.json_path.read_bytes()
    assert out.marks_path.read_bytes() == reference.marks_path.read_bytes()
    # Строки CLI различаются только каталогом вывода.
    assert [a.replace("из_окна", "эталон") for a in out.cli_args] == reference.cli_args


def test_window_refuses_a_session_of_another_image(qapp, tmp_path, scene):
    import json

    from facade_digitizer.ui.session_file import save_session
    from facade_digitizer.ui.window import MainWindow

    saved = save_session(_session(scene), tmp_path / "a.session.json")
    raw = json.loads(saved.read_text(encoding="utf-8"))
    raw["image"]["sha256"] = "f" * 64
    saved.write_text(json.dumps(raw), encoding="utf-8")
    w = MainWindow()
    w.open_session(saved)
    assert "не тот" in w.status_label.text() and w.session.image_path is None
    w.close()
