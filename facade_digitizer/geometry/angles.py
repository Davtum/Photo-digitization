"""Поле углов визирования по плоскости фасада и локальное разрешение."""
from dataclasses import dataclass

import numpy as np

from .camera import CameraOnPlane


@dataclass(frozen=True)
class AngleMap:
    theta_x: np.ndarray
    theta_y: np.ndarray
    theta_full: np.ndarray
    bounds_mm: tuple


def angle_map(cam: CameraOnPlane, bounds_mm: tuple, shape: tuple) -> AngleMap:
    """Поле углов луча на точки плоскости фасада, в градусах.

    Углы считаются от ОПОРНОЙ ТОЧКИ камеры (cx, cy) — проекции центра камеры на
    плоскость по нормали, — поэтому вычитание опорной точки обязательно: без него
    нуль поля уезжает в начало координат фасада, а это разные точки везде, кроме
    вырожденного случая камеры над углом.

    Компоненты `theta_x` и `theta_y` разведены и знаковые (глобальное ограничение,
    спецификация п. 2.1): `theta_x` растёт вдоль оси X фасада, `theta_y` — вдоль Y.
    Смешивать их между собой и с наклоном камеры `theta_cam` нельзя.
    """
    x0, y0, x1, y1 = bounds_mm
    rows, cols = shape
    gx, gy = np.meshgrid(np.linspace(x0, x1, cols), np.linspace(y0, y1, rows))
    tan_x = (gx - cam.cx) / cam.cz
    tan_y = (gy - cam.cy) / cam.cz
    return AngleMap(np.degrees(np.arctan(tan_x)), np.degrees(np.arctan(tan_y)),
                    np.degrees(np.arctan(np.hypot(tan_x, tan_y))), bounds_mm)


def usable_mask(am: AngleMap, theta_max_deg: float = 30.0) -> np.ndarray:
    """Маска пригодной области: полный угол визирования не превосходит порога."""
    return am.theta_full <= theta_max_deg


def local_gsd(H: np.ndarray, mm_per_rect_unit: float, image_size: tuple,
              shape: tuple = (64, 64), worst_direction: bool = True) -> np.ndarray:
    """Миллиметры фасада на один пиксель ИСХОДНОГО снимка, по полю кадра.

    Именно эта величина ограничивает точность измерения: ректифицированный растр
    равномерен по построению, неравномерно распределено исходное разрешение.

    Перспектива анизотропна: масштаб вдоль X и вдоль Y в одной точке различается.
    Величина sqrt(|det J|) даёт среднее геометрическое и занижает худший случай на
    единицы процентов. Поскольку требование спецификации (п. 2.2) относится к
    ХУДШЕМУ локальному разрешению, по умолчанию берётся наибольшее сингулярное
    число якобиана.
    """
    w_px, h_px = image_size
    rows, cols = shape
    gx, gy = np.meshgrid(np.linspace(0, w_px - 1, cols), np.linspace(0, h_px - 1, rows))

    w = H[2, 0] * gx + H[2, 1] * gy + H[2, 2]
    num_x = H[0, 0] * gx + H[0, 1] * gy + H[0, 2]
    num_y = H[1, 0] * gx + H[1, 1] * gy + H[1, 2]

    j00 = (H[0, 0] * w - num_x * H[2, 0]) / w ** 2
    j01 = (H[0, 1] * w - num_x * H[2, 1]) / w ** 2
    j10 = (H[1, 0] * w - num_y * H[2, 0]) / w ** 2
    j11 = (H[1, 1] * w - num_y * H[2, 1]) / w ** 2

    det = j00 * j11 - j01 * j10
    if not worst_direction:
        return mm_per_rect_unit * np.sqrt(np.abs(det))

    trace = j00 ** 2 + j01 ** 2 + j10 ** 2 + j11 ** 2
    disc = np.clip(trace ** 2 - 4.0 * det ** 2, 0.0, None)
    sigma_max = np.sqrt((trace + np.sqrt(disc)) / 2.0)
    return mm_per_rect_unit * sigma_max
