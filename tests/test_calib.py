import numpy as np
import pytest

from facade_digitizer.pipeline.calib import intrinsics_from_meta, undistort
from facade_digitizer.pipeline.io import CameraMeta


def test_intrinsics_from_focal_and_sensor_width():
    meta = CameraMeta(model="M3E", focal_mm=24.0, sensor_width_mm=17.3,
                      image_size=(5280, 3956), captured_at=None, gnss=None)
    K, source = intrinsics_from_meta(meta)
    assert source == "exif"
    assert K[0, 0] == pytest.approx(5280 * 24.0 / 17.3, rel=1e-6)
    assert K[0, 2] == pytest.approx(2640.0)
    assert K[1, 2] == pytest.approx(1978.0)


def test_intrinsics_fall_back_to_default_fov():
    """Без EXIF принимается типичное поле зрения 84° по диагонали."""
    meta = CameraMeta(model="unknown", focal_mm=None, sensor_width_mm=None,
                      image_size=(4000, 3000), captured_at=None, gnss=None)
    K, source = intrinsics_from_meta(meta)
    assert source == "database"
    assert K[0, 0] > 0


def test_undistort_is_identity_for_zero_coefficients():
    img = np.random.default_rng(0).integers(0, 255, (200, 300), dtype=np.uint8)
    K = np.array([[300.0, 0, 150.0], [0, 300.0, 100.0], [0, 0, 1.0]])
    out = undistort(img, K, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(out, img)


def test_calibration_profile_roundtrip(tmp_path):
    """Профиль камеры сохраняется и читается — калибровка по мишени делается однажды."""
    from facade_digitizer.pipeline.calib import CalibrationProfile, load_profile, save_profile

    prof = CalibrationProfile(
        model="M3E", K=[[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]],
        dist=[-0.21, 0.09, 0.0, 0.0, -0.02], rms_px=0.34, image_size=(5280, 3956),
    )
    path = tmp_path / "m3e.json"
    save_profile(prof, path)
    assert load_profile(path).rms_px == pytest.approx(0.34)


def test_intrinsics_prefer_profile_over_exif(tmp_path):
    from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile

    prof = CalibrationProfile(
        model="M3E", K=[[3333.0, 0, 2640.0], [0, 3333.0, 1978.0], [0, 0, 1.0]],
        dist=[0.0] * 5, rms_px=0.3, image_size=(5280, 3956),
    )
    path = tmp_path / "m3e.json"
    save_profile(prof, path)

    meta = CameraMeta(model="M3E", focal_mm=24.0, sensor_width_mm=17.3,
                      image_size=(5280, 3956), captured_at=None, gnss=None)
    K, source = intrinsics_from_meta(meta, profile_path=path)
    assert source == "target"
    assert K[0, 0] == pytest.approx(3333.0)


def test_calibrate_from_chessboard_rejects_too_few_views():
    from facade_digitizer.pipeline.calib import calibrate_from_chessboard

    with pytest.raises(ValueError):
        calibrate_from_chessboard([], pattern=(9, 6), square_mm=25.0,
                                  image_size=(640, 480), model="test")


# --- тесты сверх брифа: закрывают мутации, пережившие исходный набор ---


def test_undistort_skips_resampling_for_zero_coefficients():
    """Нулевые коэффициенты возвращают тот же объект, а не пересчитанную копию.

    cv2.undistort с нулевыми коэффициентами даёт побитово тот же массив, поэтому
    сравнение значений не отличает «коэффициенты не применены» от «применены
    нулевые». Отличает тождество объекта.
    """
    img = np.random.default_rng(1).integers(0, 255, (200, 300), dtype=np.uint8)
    K = np.array([[300.0, 0, 150.0], [0, 300.0, 100.0], [0, 0, 1.0]])
    assert undistort(img, K, [0.0, 0.0, 0.0, 0.0, 0.0]) is img


def test_calibrate_from_chessboard_separates_its_two_refusals():
    """Отказ по числу поданных снимков отличается от отказа по нераспознанной мишени."""
    from facade_digitizer.pipeline.calib import calibrate_from_chessboard

    kwargs = {"pattern": (9, 6), "square_mm": 25.0, "image_size": (640, 480), "model": "test"}

    with pytest.raises(ValueError, match="не менее 5 снимков") as too_few:
        calibrate_from_chessboard([], **kwargs)

    blank = [np.zeros((480, 640), dtype=np.uint8) for _ in range(5)]
    with pytest.raises(ValueError, match="мишень распознана лишь") as not_found:
        calibrate_from_chessboard(blank, **kwargs)

    # Подстроки взаимно исключительны: ни одна не подходит к чужому сообщению.
    assert "мишень распознана лишь" not in str(too_few.value)
    assert "не менее 5 снимков" not in str(not_found.value)
