"""Загрузка снимка и разбор EXIF."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Ширина кадра 35-мм плёнки. По ней EXIF-поле FocalLengthIn35mmFilm пересчитывается
#: в ширину матрицы: sensor_width_mm = 36 · FocalLength / FocalLengthIn35mmFilm.
FULL_FRAME_WIDTH_MM = 36.0

#: Ширина матрицы по модели камеры — последнее средство, когда ни явного значения,
#: ни 35-мм эквивалента в EXIF нет. Только аппараты собственного парка.
#: 4/3 CMOS — 17.3 мм, полный кадр Zenmuse P1 — 35.9 мм.
SENSOR_WIDTH_MM_BY_MODEL = {
    "M3E": 17.3,
    "M3T": 17.3,
    "M3M": 17.3,
    "ZENMUSE L2": 17.3,
    "ZENMUSE P1": 35.9,
}

#: Множитель перевода единицы FocalPlaneResolutionUnit в миллиметры.
_RESOLUTION_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0}

#: Значения EXIF Orientation, при которых оси кадра переставлены относительно осей
#: матрицы (поворот на 90° или 270°, с отражением или без).
AXES_SWAPPED = frozenset({5, 6, 7, 8})


@dataclass(frozen=True)
class CameraMeta:
    """Метаданные снимка, нужные для восстановления внутренних параметров.

    `sensor_width_source` называет, откуда взялась ширина матрицы: измеренная из
    тега FocalPlane, выведенная из 35-мм эквивалента и взятая из таблицы моделей —
    пути разной надёжности.

    **Докуда это различие доходит на деле.** До `calib.intrinsics_from_meta`,
    которая возвращает по нему уточнённый источник K (`"exif:focal_plane"`,
    `"exif:crop_factor"`, `"exif:model_table"`). ДАЛЬШЕ ОНО НЕ ИДЁТ:
    `camera.calibration` схемы (раздел 10) различает ровно три происхождения —
    `target`, `exif`, `database`, — и уточнение сознательно схлопывается в
    `"exif"` в `pipeline.run._camera_record`. Поля под него в схеме нет.

    Прежде здесь стояло «различие обязано дойти до выходного файла», и это было
    обещание, которого код не выполняет: заявленное обязательство без исполнения
    неотличимо от забытой строки. Довести различие до выхода означало бы добавить
    поле в раздел 10 спецификации; пока такого поля там нет, здесь описывается
    то, что есть.
    """

    model: str
    focal_mm: float | None
    sensor_width_mm: float | None
    image_size: tuple[int, int]      # (ширина, высота) в пикселях
    captured_at: str | None
    gnss: dict | None
    sensor_width_source: str | None = None   # "focal_plane" | "crop_factor" | "model_table"
    #: Тег EXIF Orientation, УЖЕ применённый декодером (`cv2.imdecode` поворачивает
    #: кадр сам). 1 — поворота нет либо тега нет. Нужен затем, что при 5–8 оси кадра
    #: переставлены относительно осей матрицы, и ширину сенсора в пикселях надо брать
    #: по сырому, неповёрнутому кадру (план 3, задача 3).
    orientation: int = 1


def load_image(path: str | Path):
    """Снимок в градациях серого плюс метаданные камеры.

    Читает байты сам (`np.fromfile`) и декодирует их (`cv2.imdecode`), а не
    `cv2.imread(str(path))`. На Windows `cv2.imread` переводит путь в ANSI-кодовую
    страницу процесса и не открывает файл, если путь содержит символы вне неё —
    а каталог этого проекта называется «Оцифровка фото». `np.fromfile` открывает
    файл через собственный, не ANSI, слой Python и такого ограничения не имеет.

    Два разных отказа различаются по типу исключения, а не сливаются в один:
    файла нет — `FileNotFoundError`; файл есть, но не декодируется (испорчен или
    это не изображение) — `ValueError`. Раньше оба давали одинаковый
    `FileNotFoundError` с одним и тем же текстом, и оператор не мог по сообщению
    отличить опечатку в пути от битого снимка.
    """
    path = Path(path)
    raw = read_bytes(path)
    img = decode(raw, cv2.IMREAD_GRAYSCALE, path)
    h, w = img.shape[:2]
    meta = _read_exif(path, (w, h))
    return img, meta


def read_bytes(path: Path) -> np.ndarray:
    """Байты файла в обход кодовой страницы Windows (см. `load_image`)."""
    try:
        return np.fromfile(str(path), dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(f"изображение не найдено: {path}") from error


def decode(raw: np.ndarray, flag: int, path: Path) -> np.ndarray:
    """Декодирование с поворотом по EXIF (так делает `cv2.imdecode` без
    `IMREAD_IGNORE_ORIENTATION`). Пустой или испорченный файл — `ValueError`."""
    img = cv2.imdecode(raw, flag) if raw.size else None
    if img is None:
        raise ValueError("не удалось прочитать изображение (файл испорчен либо "
                         f"формат не распознан): {path}")
    return img


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Пишет изображение на диск в обход кодовой страницы Windows.

    `cv2.imwrite` страдает тем же дефектом, что и `cv2.imread` (см. `load_image`),
    но хуже: на не-ASCII пути он не только не пишет файл, но и **молча возвращает
    `True`** — то есть сообщает об успехе, ничего не записав. Проверено исполнением
    ("imwrite растр.png: вернул True, файл существует: False"). Тихий `True` без
    файла здесь недопустим ни при каких обстоятельствах.

    Кодируем в память (`cv2.imencode`, формат — по расширению пути) и пишем байты
    сами (`ndarray.tofile`) — путь при этом ни разу не проходит через C++-слой
    OpenCV. Любой отказ — исключение с названной причиной: неверный формат или
    нечего кодировать — `ValueError`; файловая система отказала (нет каталога, нет
    прав) — исключение `tofile`/`OSError`, не перехватывается и не глушится.
    """
    path = Path(path)
    suffix = path.suffix
    if not suffix:
        raise ValueError(f"у пути нет расширения, формат кодирования не определить: {path}")

    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"не удалось закодировать изображение в формат {suffix!r}: {path}")

    buf.tofile(str(path))
    return path


def _rational(tag_dict, tag):
    """Значение рационального тега EXIF. Отсутствие поля — None, а не нуль."""
    v = tag_dict.get(tag)
    if isinstance(v, tuple) and len(v) == 2 and v[1]:
        return v[0] / v[1]
    return None


def _decode(value):
    """Текстовый тег EXIF в str. Отсутствие поля — None, а не пустая строка."""
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip("\x00").strip()
    if value is None:
        return None
    return str(value)


def _sensor_width(exif_ifd, model, focal_mm, piexif, raw_width_px):
    """Ширина матрицы и происхождение этого значения.

    Порядок: явное значение из EXIF (FocalPlaneXResolution); затем кроп-фактор по
    35-мм эквиваленту; затем таблица по моделям. Ничего не нашлось — (None, None),
    и вызывающий уйдёт на типовое поле зрения.

    Ширина в пикселях — СЫРОГО, неповёрнутого кадра: `FocalPlaneXResolution`
    отсчитывается вдоль оси X матрицы, а не вдоль оси X кадра после поворота по EXIF.
    """
    width_px = raw_width_px

    x_res = _rational(exif_ifd, piexif.ExifIFD.FocalPlaneXResolution)
    unit = exif_ifd.get(piexif.ExifIFD.FocalPlaneResolutionUnit)
    unit_mm = _RESOLUTION_UNIT_MM.get(unit)
    if x_res and unit_mm and width_px:
        return width_px / x_res * unit_mm, "focal_plane"

    equiv_35mm = exif_ifd.get(piexif.ExifIFD.FocalLengthIn35mmFilm)
    if focal_mm and equiv_35mm:
        return FULL_FRAME_WIDTH_MM * focal_mm / equiv_35mm, "crop_factor"

    if model:
        tabulated = SENSOR_WIDTH_MM_BY_MODEL.get(model.upper())
        if tabulated:
            return tabulated, "model_table"

    return None, None


def _read_gnss(gps_ifd, piexif):
    """Телеметрия GNSS из EXIF. Без пары широта-долгота возвращается None."""
    lat = _degrees(gps_ifd.get(piexif.GPSIFD.GPSLatitude),
                   _decode(gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)), "S")
    lon = _degrees(gps_ifd.get(piexif.GPSIFD.GPSLongitude),
                   _decode(gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)), "W")
    if lat is None or lon is None:
        return None

    alt = _rational(gps_ifd, piexif.GPSIFD.GPSAltitude)
    if alt is not None and gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef) == 1:
        alt = -alt
    return {"lat": lat, "lon": lon, "alt_m": alt}


def _degrees(dms, ref, negative_ref):
    """Тройка градус-минута-секунда в градусы со знаком."""
    if not isinstance(dms, tuple) or len(dms) != 3:
        return None
    parts = []
    for pair in dms:
        if not (isinstance(pair, tuple) and len(pair) == 2 and pair[1]):
            return None
        parts.append(pair[0] / pair[1])
    value = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    return -value if ref == negative_ref else value


def _read_exif(path: Path, image_size: tuple[int, int]) -> CameraMeta:
    """Разбор EXIF. Снимок без пригодного EXIF даёт метаданные с пустыми полями."""
    try:
        import piexif
        exif = piexif.load(str(path))
    # Любая беда с EXIF — отсутствие piexif, битый или пустой блок — значит одно:
    # метаданных нет, параметры придётся брать из другого источника.
    except Exception:  # noqa: BLE001
        return CameraMeta("unknown", None, None, image_size, None, None)

    zeroth = exif.get("0th", {})
    exif_ifd = exif.get("Exif", {})
    gps_ifd = exif.get("GPS", {})

    model = _decode(zeroth.get(piexif.ImageIFD.Model))
    if not model:
        model = "unknown"
    focal = _rational(exif_ifd, piexif.ExifIFD.FocalLength)
    captured = _decode(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal))

    orientation = zeroth.get(piexif.ImageIFD.Orientation, 1)
    if orientation not in range(1, 9):
        orientation = 1
    raw_width = image_size[1] if orientation in AXES_SWAPPED else image_size[0]
    sensor_width, sensor_width_source = _sensor_width(
        exif_ifd, model, focal, piexif, raw_width
    )
    gnss = _read_gnss(gps_ifd, piexif)

    return CameraMeta(model, focal, sensor_width, image_size, captured, gnss,
                      sensor_width_source, orientation)
