"""Отрезки и точки схода. Спецификация, п. 4.2."""
import math
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class VanishingPoint:
    """Точка схода: однородные координаты, число поддержавших отрезков, невязка."""

    point: np.ndarray       # однородные координаты (3,)
    support: int
    residual_px: float


@dataclass(frozen=True)
class PlaneConfidence:
    """Мера доверия к восстановленной плоскости фасада. Спецификация, п. 4.2.

    Не диагностика, а часть контракта: ниже порога система обязана не гадать,
    а передать управление оператору. Список `reasons` называет, что именно
    помешало довериться результату.
    """

    value: float
    support_h: int
    support_v: int
    residual_px: float
    orthogonality_deg: float
    coverage: float
    reasons: list[str] = field(default_factory=list)


def _segment_lines(segs: np.ndarray) -> np.ndarray:
    """Прямые в однородных координатах через векторное произведение концов."""
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    return np.cross(p1, p2)


def _fit_vanishing_point(lines: np.ndarray, threshold: float, rng):
    """RANSAC по парам прямых. Возвращает точку и маску поддержки."""
    best_point, best_mask = None, np.zeros(len(lines), dtype=bool)
    if len(lines) < 2:
        return np.array([0.0, 0.0, 1.0]), best_mask, float("inf")

    for _ in range(200):
        i, j = rng.choice(len(lines), size=2, replace=False)
        v = np.cross(lines[i], lines[j])
        norm = np.linalg.norm(v[:2])
        if norm < 1e-9:
            continue
        dist = np.abs(lines @ v) / (np.linalg.norm(lines[:, :2], axis=1) * np.linalg.norm(v) + 1e-12)
        mask = dist < threshold
        if mask.sum() > best_mask.sum():
            best_point, best_mask = v, mask

    if best_point is None:
        return np.array([0.0, 0.0, 1.0]), best_mask, float("inf")

    v = best_point
    dist = np.abs(lines @ v) / (np.linalg.norm(lines[:, :2], axis=1) * np.linalg.norm(v) + 1e-12)
    resid = float(dist[best_mask].mean()) if best_mask.any() else float("inf")
    return best_point, best_mask, resid


def estimate_vanishing_points(
    segments: np.ndarray,
    image_size: tuple[int, int],
    K: np.ndarray,
    threshold: float = 0.02,
    seed: int = 0,
) -> tuple[VanishingPoint, VanishingPoint, PlaneConfidence]:
    """Две ортогональные точки схода плюс мера доверия к плоскости."""
    reasons: list[str] = []
    if len(segments) < 8:
        reasons.append("слишком мало отрезков")
        empty = VanishingPoint(np.array([1.0, 0.0, 0.0]), 0, float("inf"))
        return empty, empty, PlaneConfidence(0.0, 0, 0, float("inf"), 0.0, 0.0, reasons)

    angles = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1],
                                   segments[:, 2] - segments[:, 0])) % 180.0
    horiz = segments[(angles < 45.0) | (angles > 135.0)]
    vert = segments[(angles >= 45.0) & (angles <= 135.0)]

    rng = np.random.default_rng(seed)
    vh_pt, vh_mask, vh_res = (_fit_vanishing_point(_segment_lines(horiz), threshold, rng)
                              if len(horiz) >= 2 else
                              (np.array([1.0, 0.0, 0.0]), np.zeros(0, dtype=bool), float("inf")))
    vv_pt, vv_mask, vv_res = (_fit_vanishing_point(_segment_lines(vert), threshold, rng)
                              if len(vert) >= 2 else
                              (np.array([0.0, 1.0, 0.0]), np.zeros(0, dtype=bool), float("inf")))

    Kinv = np.linalg.inv(K)
    dh, dv = Kinv @ vh_pt, Kinv @ vv_pt
    dh, dv = dh / (np.linalg.norm(dh) + 1e-12), dv / (np.linalg.norm(dv) + 1e-12)
    orthogonality = math.degrees(math.acos(min(1.0, abs(float(dh @ dv)))))

    support_h, support_v = int(vh_mask.sum()), int(vv_mask.sum())
    coverage = (support_h + support_v) / max(len(segments), 1)
    # Настоящая невязка: средняя нормированная дистанция инлайеров, переведённая
    # в пиксели через размер кадра. В редакции 1 здесь стояла константа
    # threshold * max(image_size), не зависевшая ни от сцены, ни от качества подгонки.
    finite = [r for r in (vh_res, vv_res) if np.isfinite(r)]
    residual = float(np.mean(finite) * max(image_size)) if finite else float("inf")

    if support_h < 2 or support_v < 2:
        reasons.append("недостаточная поддержка одной из точек схода")
    if abs(orthogonality - 90.0) > 15.0:
        reasons.append(f"направления не ортогональны: {orthogonality:.1f}°")
    if coverage < 0.2:
        reasons.append("отрезки покрывают малую долю кадра")
    if not np.isfinite(residual):
        reasons.append("невязка не определена")
    if support_h + support_v < 10:
        reasons.append(f"мало поддерживающих отрезков: {support_h + support_v}")

    value = 0.0 if reasons else min(1.0, 0.3 + 0.7 * coverage)

    return (
        VanishingPoint(vh_pt, support_h, residual),
        VanishingPoint(vv_pt, support_v, residual),
        PlaneConfidence(value, support_h, support_v, residual, orthogonality, coverage, reasons),
    )
