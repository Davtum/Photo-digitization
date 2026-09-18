"""Поза камеры относительно плоскости фасада.

Плоскость фасада Π: Z = 0. Камера в точке (cx, cy, cz), cz > 0.
Опорная точка F = (cx, cy) — проекция камеры на Π по нормали.
Спецификация, п. 5.1.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraOnPlane:
    cx: float
    cy: float
    cz: float

    def __post_init__(self) -> None:
        if self.cz <= 0:
            raise ValueError("камера должна находиться перед плоскостью фасада: cz > 0")

    def foot(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def tan_theta(self, x: float, y: float, depth: float = 0.0) -> tuple[float, float]:
        """Компоненты (tan θ_x, tan θ_y) луча на точку (x, y) при глубине depth.

        depth > 0 — точка заглублена за плоскость, depth < 0 — вынесена вперёд.
        """
        denom = self.cz + depth
        if denom <= 0:
            raise ValueError("точка оказалась за камерой: cz + depth <= 0")
        return ((x - self.cx) / denom, (y - self.cy) / denom)

    def theta_deg(self, x: float, y: float, depth: float = 0.0) -> float:
        tx, ty = self.tan_theta(x, y, depth)
        return math.degrees(math.atan(math.hypot(tx, ty)))
