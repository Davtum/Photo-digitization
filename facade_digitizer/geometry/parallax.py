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
    theta_min_deg: float = 15.0,
) -> float:
    """Глубина заглубления по видимой ширине грани откоса.

    Ширина откоса, измеренная ПОПЕРЁК его ребра, равна не полному смещению, а его
    компоненте вдоль нормали к ребру. Замкнутое решение:

        w = d · |u| / (C_z + d)   ⟹   d = w · C_z / (|u| − w)

    где u — проекция вектора от опорной точки к кромке на edge_normal.
    Ориентация нормали безразлична: проекция берётся по модулю, так что
    `(1, 0)` и `(−1, 0)` дают одну и ту же глубину.

    `theta_min_deg` — порог компоненты угла визирования θ_⊥ = arctg(|u| / (C_z + d)).
    Ниже порога знаменатель |u| − w вырождается и глубина уходит в бесконечность:
    при |u| = 1000 и w = 999.9999 замкнутая форма даёт 1.0e11 мм без всякого признака
    неисправности. Спецификация требует θ_⊥ ≥ 15°; порог перекрываем, поскольку на
    реальных данных он может понадобиться мягче.
    Спецификация, пп. 5.4, 6.2.
    """
    nx, ny = edge_normal
    norm = math.hypot(nx, ny)
    if norm == 0:
        raise ValueError("нормаль к ребру откоса не может быть нулевой")
    nx, ny = nx / norm, ny / norm

    u = abs((edge_x - cam.cx) * nx + (edge_y - cam.cy) * ny)
    if reveal_width_mm <= 0:
        raise ValueError("ширина откоса должна быть строго положительной")
    if u <= reveal_width_mm:
        raise ValueError(
            "вырожденная геометрия: ширина откоса не меньше расстояния до опорной точки; "
            "угол визирования слишком мал либо ширина измерена неверно"
        )

    depth = reveal_width_mm * cam.cz / (u - reveal_width_mm)

    theta_deg = math.degrees(math.atan(u / (cam.cz + depth)))
    if theta_deg < theta_min_deg:
        raise ValueError(
            f"компонента угла визирования {theta_deg:.2f}° ниже порога "
            f"{theta_min_deg:.2f}°: оценка глубины неустойчива, "
            "грань откоса непригодна для измерения"
        )
    return depth


def reveal_depth_sigma(
    reveal_width_mm: float,
    u_mm: float,
    cz_mm: float,
    sigma_width_mm: float,
    sigma_u_mm: float,
    sigma_cz_mm: float = 0.0,
) -> float:
    """σ глубины: точные частные производные замкнутой формы, сложение по RSS.

    Замкнутая форма дифференцируется по своим ДЕЙСТВИТЕЛЬНО независимым переменным —
    измеренной ширине w, проекции |u| и высоте камеры C_z:

        d = w · C_z / (|u| − w)

        ∂d/∂w   =  C_z · |u| / (|u| − w)²
        ∂d/∂|u| = −w · C_z  / (|u| − w)²
        ∂d/∂C_z =  w        / (|u| − w)

        σ_d = √( (∂d/∂w · σ_w)² + (∂d/∂|u| · σ_u)² + (∂d/∂C_z · σ_Cz)² )

    Вклады складываются квадратично: источники погрешности независимы, линейная
    сумма завышает σ (на типовых числах — до 35 %).

    Прежняя малоугловая запись через угол оставлена как приближение:

        от ширины:  σ_d = σ_w / tg θ_⊥          занижает РОВНО в 1 + d / C_z
        от позы:    σ_d / d = 2 σ_θ / sin 2θ_⊥  занижает в cos²θ_⊥ / cos²φ,
                                                где φ = arctg(|u| / C_z)

    Обе занижают чувствительность, то есть ошибаются В ОПАСНУЮ СТОРОНУ: заявленная
    точность выходит выше фактической. Причина — θ_⊥ = arctg(|u| / (C_z + d)) сама
    зависит от искомой d и независимой переменной не является. На C_z = 10 м,
    d = 150 мм, θ_⊥ = 30°: ∂d/∂w = 1.75803 против 1.73205, вклад позы 2.32685
    против 2.30940.
    Спецификация, п. 6.2.
    """
    if reveal_width_mm <= 0:
        raise ValueError("ширина откоса должна быть строго положительной")
    if cz_mm <= 0:
        raise ValueError("высота камеры над плоскостью фасада должна быть положительной")
    if u_mm <= reveal_width_mm:
        raise ValueError(
            "вырожденная геометрия: ширина откоса не меньше проекции |u|, "
            "производные не определены"
        )
    if sigma_width_mm < 0 or sigma_u_mm < 0 or sigma_cz_mm < 0:
        raise ValueError("погрешности не могут быть отрицательными")

    denom = (u_mm - reveal_width_mm) ** 2
    dd_dwidth = cz_mm * u_mm / denom
    dd_du = reveal_width_mm * cz_mm / denom
    dd_dcz = reveal_width_mm / (u_mm - reveal_width_mm)

    return math.hypot(
        dd_dwidth * sigma_width_mm,
        dd_du * sigma_u_mm,
        dd_dcz * sigma_cz_mm,
    )


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
