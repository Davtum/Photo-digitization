"""Внутренние параметры камеры и снятие дисторсии. Спецификация, п. 4.2."""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import CameraMeta

DEFAULT_DIAGONAL_FOV_DEG = 84.0  # типично для DJI Zenmuse L2 и Mavic 3

MIN_CALIBRATION_VIEWS = 5


@dataclass(frozen=True)
class CalibrationProfile:
    """Результат разовой калибровки камеры по мишени. Спецификация, п. 4.2."""

    model: str
    K: list[list[float]]
    dist: list[float]
    rms_px: float
    image_size: tuple[int, int]


def save_profile(profile: CalibrationProfile, path) -> None:
    """Запись профиля камеры в JSON."""
    Path(path).write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_profile(path) -> CalibrationProfile:
    """Чтение профиля камеры из JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["image_size"] = tuple(data["image_size"])
    return CalibrationProfile(**data)


def calibrate_from_chessboard(
    images: list[np.ndarray],
    pattern: tuple[int, int],
    square_mm: float,
    image_size: tuple[int, int],
    model: str,
) -> CalibrationProfile:
    """Калибровка по снимкам шахматной мишени. Выполняется один раз на аппарат.

    Ошибка фокусного расстояния при наклоне искажает отношение сторон как ε·sin²θ:
    при ε = 2 % и θ = 30 % это 7.5 мм на полутораметровом окне. Поэтому камеры
    собственного парка калибруются по мишени, а не по EXIF. Спецификация, п. 4.2.

    Число поданных снимков проверяется до всякой обработки: меньше
    MIN_CALIBRATION_VIEWS видов — отказ с отдельным сообщением, не смешиваемым
    с сообщением о нераспознанной мишени.
    """
    if len(images) < MIN_CALIBRATION_VIEWS:
        raise ValueError("для устойчивой калибровки нужно не менее 5 снимков мишени")

    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    obj_points, img_points = [], []
    for img in images:
        found, corners = cv2.findChessboardCorners(img, pattern, None)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            img, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < MIN_CALIBRATION_VIEWS:
        raise ValueError(f"мишень распознана лишь на {len(obj_points)} снимках из {len(images)}")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    return CalibrationProfile(model=model, K=K.tolist(), dist=dist.ravel().tolist(),
                              rms_px=float(rms), image_size=image_size)


def intrinsics_from_meta(meta: CameraMeta, profile_path=None) -> tuple[np.ndarray, str]:
    """Матрица K и источник её происхождения.

    Приоритет: профиль калибровки по мишени; затем фокусное и ширина матрицы из
    EXIF; затем типичное поле зрения. Источник обязан попасть в выходной файл —
    без него нельзя оценить вклад ошибки фокусного в бюджет (спецификация, п. 6.1).

    Возвращаемый источник называет то, откуда параметры взяты на деле:
    "target" — профиль по мишени, "exif" — фокусное и ширина матрицы из снимка,
    "database" — типовое поле зрения.
    """
    w, h = meta.image_size

    if profile_path is not None and Path(profile_path).exists():
        prof = load_profile(profile_path)
        if tuple(prof.image_size) == (w, h):
            return np.array(prof.K, dtype=float), "target"

    if meta.focal_mm and meta.sensor_width_mm:
        fx = fy = w * meta.focal_mm / meta.sensor_width_mm
        source = "exif"
    else:
        diag_px = math.hypot(w, h)
        fx = fy = diag_px / (2.0 * math.tan(math.radians(DEFAULT_DIAGONAL_FOV_DEG) / 2.0))
        source = "database"

    K = np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    return K, source


def undistort(image: np.ndarray, K: np.ndarray, dist: list[float]) -> np.ndarray:
    """Снятие дисторсии. При нулевых коэффициентах возвращается исходное изображение.

    Исходный массив возвращается тем же объектом: нулевые коэффициенты не должны
    приводить ни к передискретизации, ни к копированию.
    """
    coeffs = np.asarray(dist, dtype=float)
    if not np.any(coeffs):
        return image
    return cv2.undistort(image, K, coeffs)
