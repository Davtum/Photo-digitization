"""Единый кадр для конвейера и для глаза. План 3, задача 3.

Конвейер измеряет по кадру, повёрнутому по EXIF и исправленному от дисторсии, а все
координаты разметки лежат в его пикселях. Просмотрщик, показывающий файл как есть,
отправлял бы клики не туда без единой ошибки: рецензия плана 3 измерила 13–232 px
при k1 = −0.08 и перевёрнутые оси при EXIF Orientation = 6 (QPixmap не поворачивает,
`cv2.imdecode` поворачивает).

Опорные значения берутся из декодера и из растрового `cv2.undistort`, а не из формул
модуля: проверка, повторяющая формулу, прошла бы и при ошибке в ней.
"""
import cv2
import numpy as np
import piexif
import pytest

from facade_digitizer.pipeline import run as run_module
from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile
from facade_digitizer.pipeline.frame import load_frame, raw_points_to_frame
from facade_digitizer.pipeline.io import save_image

RAW_W, RAW_H = 600, 300
MARK_RAW = (451.0, 83.0)          # точка на СЫРОМ растре файла


def _jpeg_with_orientation(path, orientation: int, marker=MARK_RAW):
    """JPEG 600×300 с яркой меткой в `marker` и тегом Orientation."""
    img = np.full((RAW_H, RAW_W, 3), 40, np.uint8)
    img[:, :, 2] = np.linspace(0, 200, RAW_W, dtype=np.uint8)[None, :]   # асимметрия
    cv2.circle(img, (int(marker[0]), int(marker[1])), 6, (255, 255, 255), -1)
    save_image(path, img)
    piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}}),
                  str(path))
    return path


def _bright_centroid(gray: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(gray > 230)
    return float(xs.mean()), float(ys.mean())


def test_exif_orientation_6_is_applied_once(tmp_path):
    frame = load_frame(_jpeg_with_orientation(tmp_path / "кадр.jpg", 6))
    assert frame.gray.shape == (RAW_W, RAW_H)            # (h, w) после поворота
    assert frame.color.shape[:2] == (RAW_W, RAW_H)
    assert frame.size == (RAW_H, RAW_W)
    assert frame.orientation == 6
    assert frame.meta.image_size == frame.size


def test_color_and_gray_frames_are_pixel_aligned(tmp_path):
    frame = load_frame(_jpeg_with_orientation(tmp_path / "кадр.jpg", 6))
    assert frame.color.shape[:2] == frame.gray.shape
    gc = _bright_centroid(frame.gray)
    cc = _bright_centroid(cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY))
    assert gc == pytest.approx(cc, abs=0.25)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_raw_file_marks_follow_the_decoder_for_every_orientation(tmp_path, orientation):
    """Перевод точки файла в кадр сверяется с тем, куда метку положил ДЕКОДЕР."""
    frame = load_frame(_jpeg_with_orientation(tmp_path / f"o{orientation}.jpg", orientation))
    decoded = _bright_centroid(frame.gray)
    converted = raw_points_to_frame([MARK_RAW], frame)[0]
    assert converted == pytest.approx(decoded, abs=0.6)


def _undistort_scene(tmp_path, k1: float):
    """Снимок с меткой и профиль с дисторсией k1 для той же камеры."""
    w, h = 1200, 900
    img = np.full((h, w), 60, np.uint8)
    marker = (1050.0, 780.0)                  # у края кадра: дисторсия там заметна
    cv2.circle(img, (int(marker[0]), int(marker[1])), 7, 255, -1)
    path = tmp_path / "снимок.png"
    save_image(path, img)
    K = [[900.0, 0.0, 600.0], [0.0, 900.0, 450.0], [0.0, 0.0, 1.0]]
    profile = tmp_path / "профиль.json"
    save_profile(CalibrationProfile(model="unknown", K=K, dist=[k1, 0.0, 0.0, 0.0, 0.0],
                                    rms_px=0.2, image_size=(w, h)), profile)
    return path, profile, marker


def test_raw_file_marks_follow_the_undistorted_raster(tmp_path):
    """Точка файла переводится туда, куда `cv2.undistort` переносит сам растр."""
    path, profile, marker = _undistort_scene(tmp_path, k1=-0.08)
    frame = load_frame(path, profile_path=profile)
    moved_to = _bright_centroid(frame.gray)
    assert np.hypot(moved_to[0] - marker[0], moved_to[1] - marker[1]) > 5.0   # сдвиг есть
    converted = raw_points_to_frame([marker], frame)[0]
    assert converted == pytest.approx(moved_to, abs=0.6)


def test_without_profile_the_frame_is_the_decoded_file(tmp_path):
    path, _profile, _marker = _undistort_scene(tmp_path, k1=-0.08)
    frame = load_frame(path)
    assert frame.dist == [0.0] * 5
    decoded = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(frame.gray, decoded)


def test_undistorted_frame_matches_the_pipeline_array(tmp_path, monkeypatch):
    """Массив, по которому считает `process()`, — побайтно тот же, что даёт `load_frame`.

    Именно `Frame` показывает просмотрщик (задача 9): совпадение с массивом конвейера
    и есть гарантия, что клик оператора ложится туда, где конвейер его измерит.
    """
    path, profile, _marker = _undistort_scene(tmp_path, k1=-0.08)
    captured = {}

    class _Stop(Exception):
        pass

    def spy(image, *args, **kwargs):
        captured["image"] = image
        raise _Stop

    monkeypatch.setattr(run_module, "estimate_plane", spy)
    ref = run_module.OperatorReference(origin_px=(100.0, 800.0),
                                       span_px=((100.0, 800.0), (1100.0, 800.0)),
                                       span_mm=10000.0)
    with pytest.raises(_Stop):
        run_module.process(path, operator_reference=ref, raster_mm_per_px=10.0,
                           profile_path=profile)
    frame = load_frame(path, profile_path=profile)
    assert captured["image"].shape == frame.gray.shape
    assert np.array_equal(captured["image"], frame.gray)


def test_process_does_not_decode_colour(tmp_path, monkeypatch):
    """Конвейер цвета не читает: лишнее декодирование 20 Мп — время без пользы."""
    path, _profile, _marker = _undistort_scene(tmp_path, k1=0.0)
    assert load_frame(path, with_color=False).color is None

    calls = []
    real = run_module.load_frame

    def spy(*args, **kwargs):
        calls.append(kwargs.get("with_color", True))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "load_frame", spy)
    ref = run_module.OperatorReference(origin_px=(100.0, 800.0),
                                       span_px=((100.0, 800.0), (1100.0, 800.0)),
                                       span_mm=10000.0)
    try:
        run_module.process(path, operator_reference=ref, raster_mm_per_px=10.0)
    except ValueError:
        pass                  # плоскость на пустом кадре не оценивается — здесь не важно
    assert calls == [False]


def _jpeg_with_lens(path, orientation: int):
    """Тот же снимок той же камерой, отличается только тегом Orientation."""
    img = np.full((RAW_H, RAW_W), 90, np.uint8)
    save_image(path, img)
    exif = {"0th": {piexif.ImageIFD.Orientation: orientation,
                    piexif.ImageIFD.Model: b"Test Camera"},
            "Exif": {piexif.ExifIFD.FocalLength: (12, 1),
                     piexif.ExifIFD.FocalLengthIn35mmFilm: 24}}
    piexif.insert(piexif.dump(exif), str(path))
    return path


@pytest.mark.parametrize("orientation", [6, 8])
def test_focal_length_in_pixels_does_not_depend_on_exif_rotation(tmp_path, orientation):
    """Поворот по EXIF не меняет объектив: fx в пикселях обязан остаться прежним.

    Ширина матрицы из кроп-фактора — длинная сторона сенсора, а после поворота на
    90° ширина КАДРА — короткая. Прежде `intrinsics_from_meta` делила ширину кадра
    после поворота на ширину сенсора и занижала fx в отношении сторон (на кадре
    5280×3956 — на 25 %), причём только у снимков, снятых «портретом».
    """
    upright = load_frame(_jpeg_with_lens(tmp_path / "upright.jpg", 1))
    rotated = load_frame(_jpeg_with_lens(tmp_path / "rotated.jpg", orientation))
    assert upright.source == rotated.source == "exif:crop_factor"
    assert rotated.K[0, 0] == pytest.approx(upright.K[0, 0], rel=1e-9)
    assert rotated.K[1, 1] == pytest.approx(upright.K[1, 1], rel=1e-9)
    # Главная точка — центр ПОВЁРНУТОГО кадра.
    assert rotated.K[0, 2] == pytest.approx(RAW_H / 2.0)
    assert rotated.K[1, 2] == pytest.approx(RAW_W / 2.0)
