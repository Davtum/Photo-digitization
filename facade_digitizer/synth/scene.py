"""Синтетические сцены с ПОЛНОЙ моделью камеры: перспектива и параллакс возникают сами."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import CameraOnPlane

WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Opening:
    x: float
    y: float
    width: float
    height: float
    depth: float = 0.0

    def corners_mm(self) -> np.ndarray:
        return np.array([
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + self.height],
            [self.x, self.y + self.height],
        ], dtype=float)


def look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Матрица поворота мир -> камера. Камера смотрит вдоль +Z, ось Y направлена вниз.

    Оба вырожденных случая отвергаются явно: без этого нормировка нулевого вектора
    даёт nan, сцена отрисовывается и тесты проходят, а матрица поворота — мусор.
    """
    f = target - center
    norm_f = np.linalg.norm(f)
    if norm_f == 0:
        raise ValueError("центр камеры совпадает с точкой наведения")
    f = f / norm_f
    r = np.cross(f, WORLD_UP)
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-9:
        raise ValueError(
            "направление взгляда коллинеарно мировой вертикали: "
            "ориентация камеры не определена"
        )
    r = r / norm_r
    d = np.cross(f, r)
    return np.array([r, d, f])


@dataclass
class SyntheticScene:
    width_mm: float
    height_mm: float
    openings: list
    camera_center: np.ndarray
    R_wc: np.ndarray
    K: np.ndarray
    image_size: tuple
    floor_band_step_mm: float = 3000.0
    _c: dict = field(default_factory=dict, repr=False)

    @classmethod
    def looking_at_centre(cls, width_mm, height_mm, openings, distance_mm,
                          offset_x_mm, offset_y_mm, K, image_size,
                          floor_band_step_mm=3000.0):
        target = np.array([width_mm / 2.0, height_mm / 2.0, 0.0])
        centre = target + np.array([offset_x_mm, offset_y_mm, distance_mm])
        return cls(width_mm, height_mm, openings, centre, look_at(centre, target),
                   K, image_size, floor_band_step_mm)

    def camera_on_plane(self) -> CameraOnPlane:
        """Истинная поза: опорная точка и расстояние до плоскости."""
        c = self.camera_center
        return CameraOnPlane(cx=float(c[0]), cy=float(c[1]), cz=float(c[2]))

    def project(self, points_mm: np.ndarray, depth: float = 0.0) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points_mm, dtype=float))
        Xw = np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), -depth)])
        Xc = (self.R_wc @ (Xw.T - self.camera_center[:, None]))
        if np.any(Xc[2] <= 0):
            raise ValueError("точка за камерой")
        uv = self.K @ Xc
        return (uv[:2] / uv[2]).T

    def render(self) -> np.ndarray:
        w_px, h_px = self.image_size
        img = np.full((h_px, w_px), 200, dtype=np.uint8)
        wall = self.project(np.array([
            [0.0, 0.0], [self.width_mm, 0.0],
            [self.width_mm, self.height_mm], [0.0, self.height_mm]]))
        cv2.fillPoly(img, [np.round(wall).astype(np.int32)], 150)
        # Межэтажные членения: без них на голой стене детектор отрезков находит
        # 8-10 штук, чего не хватает для устойчивой оценки точек схода. Реальные
        # фасады такие горизонтали имеют почти всегда.
        if self.floor_band_step_mm > 0:
            y = self.floor_band_step_mm
            while y < self.height_mm:
                pts = self.project(np.array([[0.0, y], [self.width_mm, y]]))
                cv2.line(img, tuple(np.round(pts[0]).astype(int)),
                         tuple(np.round(pts[1]).astype(int)), 115, 6)
                y += self.floor_band_step_mm

        for op in self.openings:
            cv2.fillPoly(img, [np.round(self.project(op.corners_mm())).astype(np.int32)], 90)
            cv2.fillPoly(img, [np.round(self.project(op.corners_mm(), depth=op.depth)).astype(np.int32)], 35)
        return img

    def reveal_width_px(self, index: int = 0) -> float:
        op = self.openings[index]
        outer = self.project(op.corners_mm())
        inner = self.project(op.corners_mm(), depth=op.depth)
        return float(np.max(np.abs(inner[:, 0] - outer[:, 0])))
