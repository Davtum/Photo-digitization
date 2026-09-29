"""Профиль калибровки из окна. План 3, задача 19 (пункт «чужой профиль» чек-листа)."""
import pytest

pytest.importorskip("PySide6")

from tests import test_ui_marks as _marks
from tests.test_ui_marks import _wait

scene = _marks.scene


def _profile(tmp_path, sc, model="unknown"):
    """Профиль под синтетический кадр: без EXIF модель камеры читается как «unknown»."""
    from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile

    path = tmp_path / f"профиль{model}.json"
    save_profile(CalibrationProfile(model=model, K=sc.K.tolist(), dist=[0.0] * 5,
                                    rms_px=0.2, image_size=tuple(sc.image_size)), path)
    return path


@pytest.fixture
def window(qapp, scene):
    from facade_digitizer.ui.window import MainWindow

    w = MainWindow()
    w.show()
    w.open_image(scene[1])
    assert _wait(qapp, lambda: w.session.frame is not None)
    yield w
    w.close()


def test_profile_recomputes_the_frame_with_target_calibration(qapp, window, scene, tmp_path):
    assert window.session.frame.camera_record.calibration != "target"
    assert window.set_profile(_profile(tmp_path, scene[0]))
    assert _wait(qapp, lambda: window.session.frame is not None
                 and window.session.frame.camera_record.calibration == "target")
    assert window.set_profile(None, discard_clicks=True)
    assert _wait(qapp, lambda: window.session.frame is not None
                 and window.session.frame.camera_record.calibration != "target")


def test_profile_change_with_clicks_needs_confirmation(window, scene, tmp_path):
    _sc, _path, base_px, *_ = scene
    window.set_mode("base")
    window.handle_click(*base_px[0], 1.0)
    assert not window.set_profile(_profile(tmp_path, scene[0]))
    assert "подтвердите сброс" in window.status_label.text()
    assert len(window.session.reference.ends) == 1            # точки не тронуты
    assert window.set_profile(_profile(tmp_path, scene[0]), discard_clicks=True)
    assert window.session.reference.ends == []


def test_foreign_profile_is_refused_by_camera_name(qapp, window, scene, tmp_path):
    window.set_profile(_profile(tmp_path, scene[0], model="FC3411"))
    assert _wait(qapp, lambda: "камере «FC3411»" in window.status_label.text())
    assert "Кадр не обработан" in window.status_label.text()
    assert window.session.frame is None
