"""Кодирование изображений для страницы. Веб-интерфейс, задача 2.

**Кадр — PNG без потерь.** Оператор целится в кромку при 4:1, и JPEG дал бы там звон
на блоках 8×8 и смазанную цветность (4:2:0): замер σ кромки (задача 25 плана 3) шёл
бы по другой картинке, чем та, по которой считает конвейер. Сжатие 1 — быстрое:
кадр кодируется один раз, в фоне, вместе с фазой кадра. Выровненный вид и
миниатюры — JPEG: по ним не кликают.

Пути читаются байтами (`np.fromfile`), а не `cv2.imread`: OpenCV на Windows не
открывает путь с кириллицей («Оцифровка фото»), см. `pipeline.io.load_image`.
"""
from pathlib import Path

import cv2
import numpy as np

#: Затемнение непригодной по углу части кадра (п. 2.4) — альфа окна Qt.
UNUSABLE_ALPHA = 110


def _encode(ext: str, image: np.ndarray, params=()) -> bytes:
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(image), list(params))
    if not ok:
        raise ValueError(f"изображение не кодируется в {ext}")
    return buf.tobytes()


def png(image: np.ndarray, compression: int = 1) -> bytes:
    return _encode(".png", image, (cv2.IMWRITE_PNG_COMPRESSION, compression))


def jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    return _encode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, quality))


#: Наибольшая сторона маски пригодности на странице: достаточно для плавной границы.
USABLE_MAX_SIDE = 1024


def usable_png(mask: np.ndarray, *, theta_deg=None, visible=None, limit_deg=None,
               frame_size=None) -> bytes:
    """Маска пригодности → RGBA: непригодное затемнено, пригодное прозрачно.

    С полем углов (`theta_deg`, `visible` — на той же сетке узлов, что и `mask`)
    граница строится по растянутому ПОЛЮ, а не по растянутой маске: угол непрерывен,
    и порог по бикубически растянутому полю даёт плавную границу, а растянутая маска
    узлов 64×64 — лесенку из клеток, которой в поле нет. В узлах сетки обе совпадают.
    """
    if theta_deg is not None and frame_size is not None:
        fw, fh = frame_size
        k = USABLE_MAX_SIDE / max(fw, fh)
        size = (max(1, round(fw * k)), max(1, round(fh * k)))
        seen = np.asarray(visible, dtype=bool)
        theta = np.where(seen, np.asarray(theta_deg, dtype=float), 180.0).astype(np.float32)
        up = cv2.resize(theta, size, interpolation=cv2.INTER_CUBIC)
        seen_up = cv2.resize(seen.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        mask = (up <= limit_deg) & (seen_up >= 0.5)
    m = np.asarray(mask, dtype=bool)
    rgba = np.zeros(m.shape + (4,), np.uint8)
    rgba[~m] = (0, 0, 0, UNUSABLE_ALPHA)
    return png(rgba)


def invalid_png(valid_mask: np.ndarray) -> bytes:
    """Непокрытая снимком часть растра — явная штриховка, а не серый фон: серое поле
    читается как измеренная часть фасада, которой там нет (окно Qt, задача 14)."""
    mask = np.asarray(valid_mask, dtype=bool)
    stripes = ((np.add.outer(np.arange(mask.shape[0]), np.arange(mask.shape[1])) // 6)
               % 2).astype(bool)
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    # OpenCV пишет BGRA: красная полоса — (60, 60, 200).
    rgba[~mask & stripes] = (60, 60, 200, 150)
    rgba[~mask & ~stripes] = (255, 255, 255, 120)
    return png(rgba)


def thumbnail(path, width: int = 360) -> bytes:
    """Миниатюра снимка (JPEG). Уменьшенное декодирование — в разы быстрее полного."""
    raw = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_REDUCED_COLOR_4) if raw.size else None
    if image is None or image.shape[1] < width:
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if image is None:
        raise ValueError(f"{Path(path).name}: не читается как снимок")
    h, w = image.shape[:2]
    height = max(1, round(h * width / w))
    return jpeg(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), 82)
