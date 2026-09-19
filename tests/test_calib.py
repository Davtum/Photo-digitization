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


# --- калибровка по синтетической мишени с известными параметрами ---

CHESSBOARD = (9, 6)          # внутренние углы: 9 по горизонтали, 6 по вертикали
SQUARE_MM = 25.0
SYNTH_IMAGE_SIZE = (640, 480)
SYNTH_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
_TEMPLATE_SCALE = 3          # пикселей шаблона на миллиметр мишени
_TEMPLATE_MARGIN_MM = 25.0   # белое поле вокруг мишени: без него углы не находятся


def _chessboard_template():
    """Мишень в собственных координатах и матрица «объектные мм → пиксели шаблона»."""
    cols, rows = CHESSBOARD
    side = int(SQUARE_MM * _TEMPLATE_SCALE)
    squares = np.indices((rows + 1, cols + 1)).sum(axis=0) % 2
    board = np.kron(squares, np.ones((side, side), np.uint8)) * 255

    margin = int(_TEMPLATE_MARGIN_MM * _TEMPLATE_SCALE)
    canvas = np.full((board.shape[0] + 2 * margin, board.shape[1] + 2 * margin), 255, np.uint8)
    canvas[margin:margin + board.shape[0], margin:margin + board.shape[1]] = board

    shift = (SQUARE_MM + _TEMPLATE_MARGIN_MM) * _TEMPLATE_SCALE
    to_template = np.array([[_TEMPLATE_SCALE, 0.0, shift],
                            [0.0, _TEMPLATE_SCALE, shift],
                            [0.0, 0.0, 1.0]])
    return canvas, to_template


def _render_view(template, to_template, rvec, tvec):
    """Снимок плоской мишени при заданной позе: гомография точна для плоскости."""
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    plane = SYNTH_K @ np.column_stack([rotation[:, 0], rotation[:, 1],
                                       np.asarray(tvec, dtype=float)])
    return cv2.warpPerspective(template, plane @ np.linalg.inv(to_template),
                               SYNTH_IMAGE_SIZE, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


_SYNTH_POSES = [
    ((0.00, 0.00, 0.00), (-100, -62, 420)),
    ((0.25, 0.00, 0.00), (-100, -50, 430)),
    ((-0.25, 0.00, 0.00), (-100, -70, 430)),
    ((0.00, 0.28, 0.00), (-90, -62, 430)),
    ((0.00, -0.28, 0.00), (-110, -62, 430)),
    ((0.20, 0.20, 0.10), (-95, -58, 450)),
    ((-0.20, 0.22, -0.10), (-105, -66, 450)),
    ((0.18, -0.24, 0.15), (-98, -60, 400)),
    ((-0.22, -0.20, -0.12), (-102, -64, 470)),
]


def synthetic_chessboard_views():
    """Девять видов мишени, снятых камерой с матрицей SYNTH_K."""
    template, to_template = _chessboard_template()
    return [_render_view(template, to_template, rvec, tvec) for rvec, tvec in _SYNTH_POSES]


def test_calibrate_from_chessboard_recovers_known_intrinsics():
    """Калибровка по синтетической мишени возвращает близкие к истине параметры."""
    from facade_digitizer.pipeline.calib import calibrate_from_chessboard

    views = synthetic_chessboard_views()
    profile = calibrate_from_chessboard(views, pattern=CHESSBOARD, square_mm=SQUARE_MM,
                                        image_size=SYNTH_IMAGE_SIZE, model="synthetic")

    K = np.array(profile.K)
    assert K[0, 0] == pytest.approx(800.0, rel=0.01, abs=8.0)
    assert K[1, 1] == pytest.approx(800.0, rel=0.01, abs=8.0)
    assert K[0, 2] == pytest.approx(320.0, rel=0.02, abs=6.4)
    assert K[1, 2] == pytest.approx(240.0, rel=0.02, abs=4.8)
    assert 0.0 < profile.rms_px < 1.0
    assert profile.model == "synthetic"
    assert profile.image_size == SYNTH_IMAGE_SIZE
    assert len(profile.dist) >= 5


def test_calibrated_profile_outranks_exif_end_to_end(tmp_path):
    """Профиль, полученный калибровкой, действительно перебивает EXIF того же снимка."""
    from facade_digitizer.pipeline.calib import calibrate_from_chessboard, save_profile
    from facade_digitizer.pipeline.io import CameraMeta

    profile = calibrate_from_chessboard(synthetic_chessboard_views(), pattern=CHESSBOARD,
                                        square_mm=SQUARE_MM, image_size=SYNTH_IMAGE_SIZE,
                                        model="synthetic")
    path = tmp_path / "synthetic.json"
    save_profile(profile, path)

    meta = CameraMeta(model="synthetic", focal_mm=24.0, sensor_width_mm=17.3,
                      image_size=SYNTH_IMAGE_SIZE, captured_at=None, gnss=None,
                      sensor_width_source="crop_factor")
    K, source = intrinsics_from_meta(meta, profile_path=path)

    assert source == "target"
    assert K[0, 0] == pytest.approx(np.array(profile.K)[0, 0], rel=1e-9, abs=1e-9)
    # EXIF дал бы заметно иное фокусное — значит, проверка не вырождена.
    assert abs(K[0, 0] - 640 * 24.0 / 17.3) > 10.0
