"""Параллакс точек, не лежащих в плоскости фасада. Спецификация, раздел 5.

Точка на глубине d за плоскостью после ректификации оказывается смещённой:

    δ = (C_xy − P_xy) · d / (C_z + d)

Для заглублённых точек (d > 0) смещение направлено К опорной точке F,
для выступающих (d < 0) — ОТ неё.
"""
import math

from .camera import CameraOnPlane


def apparent_position(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Где точка (x, y, −depth) окажется после ректификации."""
    denom = cam.cz + depth
    if denom <= 0:
        raise ValueError("точка оказалась за камерой: cz + depth <= 0")
    k = cam.cz / denom
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
    """Обратная операция: из наблюдаемого положения в истинное.

    Область определения совпадает с `apparent_position`: функции объявлены взаимно
    обратными, поэтому охрана здесь та же.
    """
    denom = cam.cz + depth
    if denom <= 0:
        raise ValueError("точка оказалась за камерой: cz + depth <= 0")
    k = denom / cam.cz
    return (cam.cx + k * (xa - cam.cx), cam.cy + k * (ya - cam.cy))


def reveal_depth(
    cam: CameraOnPlane,
    edge_x: float,
    edge_y: float,
    reveal_width_mm: float,
    edge_normal: tuple[float, float],
) -> float:
    """Глубина заглубления по видимой ширине грани откоса.

    Ширина откоса, измеренная ПОПЕРЁК его ребра, равна не полному смещению, а его
    компоненте вдоль нормали к ребру. Замкнутое решение:

        w = d · |u| / (C_z + d)   ⟹   d = w · C_z / (|u| − w)

    где u — проекция вектора от опорной точки к кромке на edge_normal.
    Спецификация, п. 5.4.
    """
    nx, ny = edge_normal
    norm = math.hypot(nx, ny)
    if norm == 0:
        raise ValueError("нормаль к ребру откоса не может быть нулевой")
    nx, ny = nx / norm, ny / norm

    u = abs((edge_x - cam.cx) * nx + (edge_y - cam.cy) * ny)
    if reveal_width_mm <= 0:
        raise ValueError("ширина откоса должна быть положительной")
    if u <= reveal_width_mm:
        raise ValueError(
            "вырожденная геометрия: ширина откоса не меньше расстояния до опорной точки; "
            "угол визирования слишком мал либо ширина измерена неверно"
        )
    return reveal_width_mm * cam.cz / (u - reveal_width_mm)


def reveal_depth_sigma(
    depth_mm: float,
    reveal_width_mm: float,
    sigma_width_mm: float,
    tan_perp: float,
    sigma_theta_rad: float,
) -> float:
    """σ глубины: вклад ошибки ширины и вклад ошибки позы.

    От ширины:  σ_d = σ_w / |t·n̂|
    От позы:    σ_d / d = 2 σ_θ / sin 2θ
    Спецификация, п. 6.2.
    """
    if tan_perp <= 0:
        raise ValueError("компонента угла должна быть положительной")
    from_width = sigma_width_mm / tan_perp

    theta = math.atan(tan_perp)
    from_pose = depth_mm * 2.0 * sigma_theta_rad / math.sin(2.0 * theta)

    return math.hypot(from_width, from_pose)


def visible_reveal_side(
    cam: CameraOnPlane,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> tuple[str, str]:
    """Какие грани откоса видны: дальние от опорной точки.

    Возвращает (вертикальная, горизонтальная). Если опорная точка попадает внутрь
    диапазона проёма, соответствующая грань называется по знаку, но её угол мал —
    вызывающий код обязан проверить порог θ_min.
    Спецификация, п. 5.5.
    """
    vertical = "right" if cam.cx < (x_left + x_right) / 2 else "left"
    horizontal = "top" if cam.cy < (y_bottom + y_top) / 2 else "bottom"
    return (vertical, horizontal)
