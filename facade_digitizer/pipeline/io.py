"""Загрузка снимка и разбор EXIF."""
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CameraMeta:
    """Метаданные снимка, нужные для восстановления внутренних параметров."""

    model: str
    focal_mm: float | None
    sensor_width_mm: float | None
    image_size: tuple[int, int]      # (ширина, высота) в пикселях
    captured_at: str | None
    gnss: dict | None


def load_image(path: str | Path):
    """Снимок в градациях серого плюс метаданные камеры."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"не удалось прочитать изображение: {path}")

    h, w = img.shape[:2]
    meta = _read_exif(path, (w, h))
    return img, meta


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

    def rational(tag_dict, tag):
        v = tag_dict.get(tag)
        if isinstance(v, tuple) and len(v) == 2 and v[1]:
            return v[0] / v[1]
        return None

    model = zeroth.get(piexif.ImageIFD.Model, b"unknown")
    model = model.decode(errors="replace") if isinstance(model, bytes) else str(model)
    focal = rational(exif_ifd, piexif.ExifIFD.FocalLength)
    captured = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
    captured = captured.decode(errors="replace") if isinstance(captured, bytes) else None

    return CameraMeta(model, focal, None, image_size, captured, None)
