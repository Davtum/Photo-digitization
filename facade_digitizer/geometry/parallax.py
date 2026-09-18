"""Параллакс точек, не лежащих в плоскости фасада. Спецификация, раздел 5.

Точка на глубине d за плоскостью после ректификации оказывается смещённой:

    δ = (C_xy − P_xy) · d / (C_z + d)

Для заглублённых точек (d > 0) смещение направлено К опорной точке F,
для выступающих (d < 0) — ОТ неё.
"""
from .camera import CameraOnPlane


def apparent_position(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Где точка (x, y, −depth) окажется после ректификации."""
    k = cam.cz / (cam.cz + depth)
    return (cam.cx + k * (x - cam.cx), cam.cy + k * (y - cam.cy))


def parallax_offset(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Вектор смещения δ = P′ − P_xy."""
    ax, ay = apparent_position(cam, x, y, depth)
    return (ax - x, ay - y)


def correct_for_depth(
    cam: CameraOnPlane, xa: float, ya: float, depth: float
) -> tuple[float, float]:
    """Обратная операция: из наблюдаемого положения в истинное."""
    k = (cam.cz + depth) / cam.cz
    return (cam.cx + k * (xa - cam.cx), cam.cy + k * (ya - cam.cy))
