"""Отрезки и точки схода. Спецификация, п. 4.2."""
import cv2
import numpy as np


def detect_segments(gray: np.ndarray, min_length_px: float = 40.0) -> np.ndarray:
    """Отрезки на изображении, отфильтрованные по длине.

    Требует OpenCV >= 4.8.0: в 4.1–4.7 LSD был изъят из-за патента.
    """
    if gray.ndim != 2:
        raise ValueError("ожидается одноканальное изображение")

    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        return np.empty((0, 4), dtype=float)

    segs = lines.reshape(-1, 4).astype(float)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= min_length_px]
