"""Метрика трудозатрат оператора. План 3, задача 22."""
import dataclasses
import re

import pytest

from facade_digitizer.ui.labour import IDLE_CAP_S, LabourLog, checked_label, opaque_label
from facade_digitizer.ui.session import ClickedPoint, OperatorSession
from tests import test_session_file as _sf
from tests.test_session_file import _session, _sigma

scene = _sf.scene


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def _session_with_clock():
    clock = FakeClock()
    s = OperatorSession()
    s.labour = LabourLog(clock=clock)
    s.labour.start()
    return s, clock


def test_edits_and_undos_are_counted_per_element():
    s, clock = _session_with_clock()
    mark = s.new_mark()
    for i in range(4):
        clock.tick(2.0)
        s.add_corner(mark, ClickedPoint(10.0 * i, 5.0, 2.0))
    clock.tick(3.0)
    s.undo_last_point(mark)
    clock.tick(1.0)
    s.add_corner(mark, ClickedPoint(30.0, 5.0, 2.0))
    for _ in range(5):                                   # движение мыши — не правка
        s.move_point(f"{mark.id}:corner:0", ClickedPoint(1.0, 5.0, 2.0), final=False)
    clock.tick(4.0)
    s.move_point(f"{mark.id}:corner:0", ClickedPoint(1.5, 5.0, 2.0))
    record = s.labour.operator_record(mark.id)
    assert record["edits"] == 2                          # один перенос и одна отмена
    assert record["seconds"] == pytest.approx(16.0)
    summary = s.labour.summary()
    assert summary["edits"] == 1 and summary["undos"] == 1


def test_time_on_reject_frames_is_kept_apart(scene):
    s = _session(scene)
    assert s.compute_scale(_sigma) is not None, s.error
    clock = FakeClock()
    s.labour = LabourLog(clock=clock)
    s.labour.start()
    mark = s.new_mark()
    clock.tick(5.0)
    s.add_corner(mark, ClickedPoint(100.0, 100.0, 2.0))
    assert s.labour.reject_s == 0.0
    s.scale = dataclasses.replace(
        s.scale, quality=s.scale.quality.model_copy(update={"verdict": "reject"}))
    clock.tick(7.0)
    s.add_corner(mark, ClickedPoint(200.0, 100.0, 2.0))
    assert s.labour.reject_s == pytest.approx(7.0)
    assert s.labour.total_s == pytest.approx(12.0)


def test_time_at_coarse_scale_is_measured():
    s, clock = _session_with_clock()
    mark = s.new_mark()
    clock.tick(3.0)
    s.add_corner(mark, ClickedPoint(1.0, 1.0, 0.5))      # грубее 2:1
    clock.tick(5.0)
    s.add_corner(mark, ClickedPoint(2.0, 1.0, 3.0))
    summary = s.labour.summary()
    assert summary["coarse_scale_s"] == pytest.approx(3.0)
    assert summary["coarse_scale_fraction"] == pytest.approx(3.0 / 8.0)


def test_operator_label_is_opaque():
    assert re.fullmatch(r"op-[0-9a-f]{8}", opaque_label())
    assert opaque_label() != opaque_label()
    for bad in ("Иван Петров", "ivan", "op-IVAN1234", "op-1234", ""):
        with pytest.raises(ValueError, match="непрозрачна"):
            checked_label(bad)
    with pytest.raises(ValueError):
        LabourLog(operator="petrov")


def test_intervals_are_non_negative_when_the_system_clock_is_changed(monkeypatch):
    """Часы монотонные: перевод системного времени не трогает интервалы."""
    import time

    log = LabourLog()
    assert log.clock is time.monotonic
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() - 3600.0)    # перевели на час назад
    log.start()
    log.event("click", "w_000")
    assert log.total_s >= 0.0 and log.elements["w_000"].seconds >= 0.0
    # И если бы часы всё-таки пошли назад, интервал не стал бы отрицательным.
    clock = FakeClock()
    log = LabourLog(clock=clock)
    log.start()
    clock.tick(-50.0)
    log.event("click", "w_000")
    assert log.elements["w_000"].seconds == 0.0


def test_idle_time_is_capped():
    s, clock = _session_with_clock()
    mark = s.new_mark()
    clock.tick(3 * 3600.0)                                # ушёл на обед
    s.add_corner(mark, ClickedPoint(1.0, 1.0, 2.0))
    assert s.labour.operator_record(mark.id)["seconds"] == pytest.approx(IDLE_CAP_S)


def test_operator_record_reaches_the_output_and_the_cli(scene, tmp_path):
    import json
    import shlex

    from facade_digitizer.pipeline import run
    from facade_digitizer.ui.session_file import export

    s = _session(scene)
    out = export(s, tmp_path / "окно", _sigma, dxf=False)
    model = json.loads(out.json_path.read_text(encoding="utf-8"))
    for element in model["elements"]:
        assert re.fullmatch(r"op-[0-9a-f]{8}", element["operator"]["id"])
        assert element["operator"]["seconds"] >= 0.0
    args = shlex.split(out.cli_line)[1:]
    args[args.index("--out-dir") + 1] = str(tmp_path / "cli")
    assert run.main(args) == 0
    assert (tmp_path / "cli" / out.json_path.name).read_bytes() == out.json_path.read_bytes()


def test_ui_entry_refuses_a_name_as_operator_label(capsys):
    from facade_digitizer.web.__main__ import main

    assert main(["--operator", "Petrov"]) == 2
    assert "непрозрачна" in capsys.readouterr().err
