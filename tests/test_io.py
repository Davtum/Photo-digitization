"""Загрузка снимка и разбор EXIF.

Непроверенный разбор EXIF — источник молчаливо неверных внутренних параметров,
то есть систематической ошибки во всех измерениях проекта. Поэтому синтетический
JPEG с заданными полями, а не доверие к формату на слово.
"""
from pathlib import Path

import numpy as np
import piexif
import pytest

from facade_digitizer.pipeline.calib import intrinsics_from_meta
from facade_digitizer.pipeline.io import _read_exif, load_image, save_image

WIDTH, HEIGHT = 400, 300


def write_jpeg(path, zeroth=None, exif=None, gps=None, size=(WIDTH, HEIGHT)):
    """JPEG заданного размера с указанными полями EXIF."""
    w, h = size
    # Плавный градиент, а не шум: JPEG сжимает с потерями, и на шуме сравнивать нечего.
    col = np.linspace(0, 200, w, dtype=np.float64)
    row = np.linspace(0, 40, h, dtype=np.float64)
    pixels = (col[None, :] + row[:, None]).round().astype(np.uint8)
    save_image(path, pixels)
    blob = piexif.dump({"0th": dict(zeroth or {}), "Exif": dict(exif or {}),
                        "GPS": dict(gps or {}), "1st": {}, "Interop": {},
                        "thumbnail": None})
    piexif.insert(blob, str(path))
    return pixels


def test_load_image_returns_pixels_and_width_height_order(tmp_path):
    path = tmp_path / "shot.jpg"
    pixels = write_jpeg(path, zeroth={piexif.ImageIFD.Model: b"M3E"},
                        exif={piexif.ExifIFD.FocalLength: (240, 10),
                              piexif.ExifIFD.DateTimeOriginal: b"2026:09:17 12:00:00"})

    img, meta = load_image(path)

    assert img.shape == (HEIGHT, WIDTH)
    # Пиксели те же с точностью до потерь JPEG — снимок прочитан, а не подменён.
    assert np.abs(img.astype(int) - pixels.astype(int)).max() <= 3
    assert meta.image_size == (WIDTH, HEIGHT)   # именно (ширина, высота)
    assert meta.model == "M3E"
    assert meta.focal_mm == pytest.approx(24.0, rel=1e-9, abs=1e-9)
    assert meta.captured_at == "2026:09:17 12:00:00"


def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="не найдено"):
        load_image(tmp_path / "нет-такого.jpg")


def test_load_image_distinguishes_corrupt_file_from_missing_file(tmp_path):
    """Файл есть, но не декодируется, — другая причина отказа, чем «файла нет».

    Раньше оба случая поднимали `FileNotFoundError` с одним и тем же текстом:
    оператор не мог по сообщению отличить опечатку в пути от битого снимка.
    """
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not an image, just some bytes")

    with pytest.raises(ValueError, match="не удалось прочитать"):
        load_image(path)


def test_missing_and_corrupt_file_raise_different_exception_types(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"garbage, not a real image")

    missing_error = None
    corrupt_error = None
    try:
        load_image(tmp_path / "нет.jpg")
    except Exception as error:  # noqa: BLE001 — тип и текст сравниваются ниже
        missing_error = error
    try:
        load_image(corrupt)
    except Exception as error:  # noqa: BLE001
        corrupt_error = error

    assert isinstance(missing_error, FileNotFoundError)
    assert isinstance(corrupt_error, ValueError)
    assert not isinstance(corrupt_error, FileNotFoundError)
    assert str(missing_error) != str(corrupt_error)


def test_image_without_exif_gives_empty_meta(tmp_path):
    path = tmp_path / "plain.png"
    save_image(path, np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

    _, meta = load_image(path)

    assert meta.model == "unknown"
    assert meta.focal_mm is None
    assert meta.sensor_width_mm is None
    assert meta.sensor_width_source is None
    assert meta.gnss is None


def test_missing_exif_fields_stay_none_not_zero(tmp_path):
    """Отсутствующее поле — None. Нуль здесь означал бы «измерено и равно нулю»."""
    path = tmp_path / "bare.jpg"
    write_jpeg(path, exif={piexif.ExifIFD.ExifVersion: b"0230"})

    _, meta = load_image(path)

    assert meta.focal_mm is None
    assert meta.captured_at is None
    assert meta.sensor_width_mm is None
    assert meta.sensor_width_source is None
    # Модель не указана — «unknown», а не имя какого-нибудь аппарата из таблицы.
    assert meta.model == "unknown"
    assert intrinsics_from_meta(meta)[1] == "database"


def test_fields_are_not_confused_with_each_other(tmp_path):
    """Фокусное, 35-мм эквивалент, модель и момент съёмки не путаются местами."""
    path = tmp_path / "distinct.jpg"
    write_jpeg(path, zeroth={piexif.ImageIFD.Model: b"M3E",
                             piexif.ImageIFD.Make: b"DJI"},
               exif={piexif.ExifIFD.FocalLength: (1230, 100),      # 12.30 мм
                     piexif.ExifIFD.FocalLengthIn35mmFilm: 24,
                     piexif.ExifIFD.DateTimeOriginal: b"2026:09:17 12:00:00"})

    _, meta = load_image(path)

    assert meta.focal_mm == pytest.approx(12.30, rel=1e-9, abs=1e-9)
    assert meta.model == "M3E"                  # не "DJI" и не дата
    assert meta.captured_at == "2026:09:17 12:00:00"
    # 36 · 12.30 / 24, а не 36 · 24 / 12.30.
    assert meta.sensor_width_mm == pytest.approx(18.45, rel=1e-9, abs=1e-9)


def test_sensor_width_from_crop_factor(tmp_path):
    """Ширина матрицы выводится из 35-мм эквивалента — без всякой внешней базы."""
    path = tmp_path / "crop.jpg"
    # Модели нет в таблице — ширина обязана взяться из кроп-фактора.
    write_jpeg(path, zeroth={piexif.ImageIFD.Model: "неизвестный аппарат".encode()},
               exif={piexif.ExifIFD.FocalLength: (240, 10),
                     piexif.ExifIFD.FocalLengthIn35mmFilm: 50})

    _, meta = load_image(path)

    assert meta.sensor_width_mm == pytest.approx(36.0 * 24.0 / 50.0, rel=1e-9, abs=1e-9)
    assert meta.sensor_width_source == "crop_factor"
    K, source = intrinsics_from_meta(meta)
    assert source == "exif:crop_factor"
    assert K[0, 0] == pytest.approx(WIDTH * 24.0 / 17.28, rel=1e-9, abs=1e-9)


def test_explicit_sensor_width_wins_over_crop_factor(tmp_path):
    """Измеренная ширина матрицы надёжнее выведенной, поэтому идёт первой."""
    path = tmp_path / "plane.jpg"
    # FocalPlaneXResolution в пикселях на дюйм: 400 px / 17.3 мм · 25.4 мм/дюйм.
    x_res = WIDTH * 25.4 / 17.3
    write_jpeg(path, exif={piexif.ExifIFD.FocalLength: (240, 10),
                           piexif.ExifIFD.FocalLengthIn35mmFilm: 50,   # дал бы 17.28
                           piexif.ExifIFD.FocalPlaneXResolution: (round(x_res * 1000), 1000),
                           piexif.ExifIFD.FocalPlaneResolutionUnit: 2})

    _, meta = load_image(path)

    assert meta.sensor_width_source == "focal_plane"
    assert meta.sensor_width_mm == pytest.approx(17.3, rel=1e-5, abs=1e-5)
    assert intrinsics_from_meta(meta)[1] == "exif:focal_plane"


def test_sensor_width_from_model_table_is_the_last_resort(tmp_path):
    """Ни явного значения, ни эквивалента — остаётся таблица по моделям парка."""
    path = tmp_path / "table.jpg"
    write_jpeg(path, zeroth={piexif.ImageIFD.Model: b"M3E"},
               exif={piexif.ExifIFD.FocalLength: (240, 10)})

    _, meta = load_image(path)

    assert meta.sensor_width_mm == pytest.approx(17.3, rel=1e-9, abs=1e-9)
    assert meta.sensor_width_source == "model_table"
    assert intrinsics_from_meta(meta)[1] == "exif:model_table"


def test_source_distinguishes_how_sensor_width_was_obtained():
    """Три пути к ширине матрицы различаются по надёжности — значит, и в выходе."""
    from facade_digitizer.pipeline.io import CameraMeta

    def source_for(sensor_width_source):
        meta = CameraMeta(model="M3E", focal_mm=24.0, sensor_width_mm=17.3,
                          image_size=(5280, 3956), captured_at=None, gnss=None,
                          sensor_width_source=sensor_width_source)
        return intrinsics_from_meta(meta)[1]

    # Поимённо, а не «четыре различных значения»: различимость не есть соответствие,
    # и перестановка двух целей местами прошла бы проверку на различимость.
    assert {s: source_for(s) for s in (None, "focal_plane", "crop_factor", "model_table")} == {
        None: "exif",
        "focal_plane": "exif:focal_plane",
        "crop_factor": "exif:crop_factor",
        "model_table": "exif:model_table",
    }


def test_exif_branch_is_reachable_on_a_real_photograph(tmp_path):
    """Ключевой случай: снимок с дрона даёт источник exif, а не типовое поле зрения.

    Без достижимой ветви EXIF ablation «EXIF против калибровки» (спецификация,
    п. 12.4) провести нельзя вообще.
    """
    path = tmp_path / "drone.jpg"
    write_jpeg(path, zeroth={piexif.ImageIFD.Make: b"DJI", piexif.ImageIFD.Model: b"M3E"},
               exif={piexif.ExifIFD.FocalLength: (1230, 100),
                     piexif.ExifIFD.FocalLengthIn35mmFilm: 24,
                     piexif.ExifIFD.DateTimeOriginal: b"2026:09:17 12:00:00"})

    _, meta = load_image(path)
    K, source = intrinsics_from_meta(meta)

    assert source.startswith("exif")
    assert source != "database"
    assert K[0, 0] == pytest.approx(WIDTH * meta.focal_mm / meta.sensor_width_mm,
                                    rel=1e-9, abs=1e-9)


def test_gnss_is_read_with_hemisphere_signs(tmp_path):
    path = tmp_path / "gnss.jpg"
    write_jpeg(path, gps={
        piexif.GPSIFD.GPSLatitude: ((12, 1), (30, 1), (3600, 100)),   # 12.51
        piexif.GPSIFD.GPSLatitudeRef: b"S",
        piexif.GPSIFD.GPSLongitude: ((45, 1), (15, 1), (1800, 100)),  # 45.255
        piexif.GPSIFD.GPSLongitudeRef: b"W",
        piexif.GPSIFD.GPSAltitude: (12345, 100),
        piexif.GPSIFD.GPSAltitudeRef: 0,
    })

    _, meta = load_image(path)

    assert meta.gnss is not None
    assert meta.gnss["lat"] == pytest.approx(-12.51, rel=1e-9, abs=1e-9)
    assert meta.gnss["lon"] == pytest.approx(-45.255, rel=1e-9, abs=1e-9)
    assert meta.gnss["alt_m"] == pytest.approx(123.45, rel=1e-9, abs=1e-9)


def test_gnss_absent_without_coordinates(tmp_path):
    path = tmp_path / "no_gnss.jpg"
    write_jpeg(path, gps={piexif.GPSIFD.GPSAltitude: (12345, 100)})

    assert load_image(path)[1].gnss is None


def test_unreadable_exif_falls_back_instead_of_raising(tmp_path):
    """Битый EXIF — не отказ, а метаданные с пустыми полями."""
    path = tmp_path / "broken.bin"
    path.write_bytes(b"\xff\xd8\xff\xe1\x00\x08Exif\x00\x00\x01\x02\x03\x04")

    meta = _read_exif(path, (WIDTH, HEIGHT))

    assert meta.model == "unknown"
    assert meta.focal_mm is None
    assert meta.image_size == (WIDTH, HEIGHT)


# --- не-ASCII пути ---------------------------------------------------------
#
# На Windows `cv2.imread`/`cv2.imwrite` переводят путь в ANSI-кодовую страницу
# процесса и отказывают на путях с символами вне неё. Каталог этого проекта
# называется «Оцифровка фото», поэтому речь не о экзотике, а о рабочем случае.
# `tmp_path` у pytest всегда ASCII, значит кириллицу в путь ниже вносим сами —
# иначе тест проверял бы что угодно, только не этот дефект.

NONASCII_PIXELS = np.full((HEIGHT, WIDTH), 77, dtype=np.uint8)


def _write_raw_png(path, pixels):
    """PNG на диске в обход `cv2.imwrite` — байтами, через `imencode` + `tofile`.

    Тестам ниже, что проверяют ЧТЕНИЕ, нужен файл, о существовании которого
    известно заранее и независимо от того, чинён ли ещё дефект записи.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", pixels)
    assert ok
    buf.tofile(str(path))


def _read_back(path):
    """Тот же путь, что `imdecode`-чтение внутри починенного `load_image`."""
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def test_load_image_absolute_path_with_cyrillic_directory(tmp_path):
    """Абсолютный путь, кириллица в каталоге. По таблице дефекта — СБОЙ."""
    path = tmp_path / "растр_каталог" / "снимок.png"
    _write_raw_png(path, NONASCII_PIXELS)

    img, _ = load_image(path)

    assert np.array_equal(img, NONASCII_PIXELS)


def test_load_image_cyrillic_filename_in_ascii_directory(tmp_path):
    """ASCII-каталог, кириллица в имени файла. По таблице дефекта — СБОЙ."""
    path = tmp_path / "ascii_dir" / "растр.png"
    _write_raw_png(path, NONASCII_PIXELS)

    img, _ = load_image(path)

    assert np.array_equal(img, NONASCII_PIXELS)


def test_load_image_relative_path_with_cyrillic_cwd(tmp_path, monkeypatch):
    """Относительный путь, кириллица ТОЛЬКО в текущем каталоге, имя файла — ASCII.

    По таблице дефекта этот случай и без починки был ОК — регрессия здесь так же
    недопустима, как и не-починка двух случаев выше. Кириллица в самом имени файла
    — уже другой случай (см. тест на ASCII-каталог выше), здесь его нарочно нет.
    """
    directory = tmp_path / "каталог_относительный"
    directory.mkdir()
    _write_raw_png(directory / "plain.png", NONASCII_PIXELS)
    monkeypatch.chdir(directory)

    img, _ = load_image(Path("plain.png"))

    assert np.array_equal(img, NONASCII_PIXELS)


def test_load_image_fully_ascii_path_still_works(tmp_path):
    """Путь целиком ASCII. Контрольный случай — должен работать и до, и после."""
    path = tmp_path / "ascii_dir" / "plain.png"
    _write_raw_png(path, NONASCII_PIXELS)

    img, _ = load_image(path)

    assert np.array_equal(img, NONASCII_PIXELS)


def test_save_image_absolute_path_with_cyrillic_directory(tmp_path):
    path = tmp_path / "растр_каталог" / "снимок.png"
    path.parent.mkdir()

    result = save_image(path, NONASCII_PIXELS)

    # Не то, что функция вернула что-то похожее на успех, а то, что файл
    # действительно появился на диске и читается обратно с теми же пикселями —
    # ровно то, чем `cv2.imwrite` на этом самом случае молчаливо соврал бы.
    assert path.exists()
    assert np.array_equal(_read_back(path), NONASCII_PIXELS)
    assert Path(result) == path


def test_save_image_cyrillic_filename_in_ascii_directory(tmp_path):
    path = tmp_path / "ascii_dir" / "растр.png"
    path.parent.mkdir()

    save_image(path, NONASCII_PIXELS)

    assert path.exists()
    assert np.array_equal(_read_back(path), NONASCII_PIXELS)


def test_save_image_relative_path_with_cyrillic_cwd(tmp_path, monkeypatch):
    """Относительный путь, кириллица только в текущем каталоге, имя файла — ASCII
    (см. пояснение в тесте на чтение того же случая)."""
    directory = tmp_path / "каталог_относительный"
    directory.mkdir()
    monkeypatch.chdir(directory)

    save_image(Path("plain.png"), NONASCII_PIXELS)

    assert Path("plain.png").exists()
    assert np.array_equal(_read_back("plain.png"), NONASCII_PIXELS)


def test_save_image_fully_ascii_path(tmp_path):
    path = tmp_path / "ascii_dir" / "plain.png"
    path.parent.mkdir()

    save_image(path, NONASCII_PIXELS)

    assert np.array_equal(_read_back(path), NONASCII_PIXELS)


def test_save_image_supports_jpeg_extension(tmp_path):
    """Формат берётся из расширения пути — тем же способом, каким `write_jpeg`
    в этом файле пишет JPEG для тестов EXIF."""
    path = tmp_path / "ascii_dir" / "снимок.jpg"
    path.parent.mkdir()

    save_image(path, NONASCII_PIXELS)

    img = _read_back(path)
    assert img is not None
    # JPEG сжимает с потерями — сравнение с допуском, как и в write_jpeg выше.
    assert np.abs(img.astype(int) - NONASCII_PIXELS.astype(int)).max() <= 3


def test_save_image_failure_is_a_named_error_not_a_silent_success(tmp_path):
    """Каталога нет — запись обязана лопнуть с названной причиной.

    Именно так, тихим `True` без файла, вела бы себя здесь прежняя запись через
    `cv2.imwrite` на кириллическом пути (см. спецификацию дефекта). Здесь причина
    другая — отсутствующий каталог, а не кодовая страница, — но опасность та же:
    отчёт об успехе не должен расходиться с файловой системой.
    """
    path = tmp_path / "нет-такого-каталога" / "снимок.png"

    with pytest.raises(OSError):
        save_image(path, NONASCII_PIXELS)

    assert not path.exists()
