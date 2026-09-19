"""Ректификация с явной привязкой растра к миллиметрам."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectified:
    """Ректифицированный растр вместе с конвенцией, связывающей его с миллиметрами.

    `H` — преобразование «изображение -> ректифицированные ПИКСЕЛИ», уже с учётом
    масштаба и сдвига начала координат; именно её, а не исходную `H_units`, обязан
    применять потребитель, измеряющий в этом растре.
    """

    image: np.ndarray
    H: np.ndarray                 # изображение -> ректифицированные ПИКСЕЛИ
    valid_mask: np.ndarray
    mm_per_px: float
    origin_rect_px: tuple


MAX_SIDE_PX = 20000


def rectify(image, H_units, mm_per_rect_unit, mm_per_px, origin_rect_units=(0.0, 0.0)):
    """Гомография в единицах -> растр, где один пиксель равен mm_per_px миллиметрам.

    Конвенция обязательна: без неё выход гомографии из точек схода безразмерен и
    порядка 1/f, отчего снимок 20 Мп схлопывается в несколько пикселей.

    `valid_mask` — ОХВАТ ПРЕОБРАЗОВАНИЯ: истина там, куда легло реальное содержимое
    кадра, ложь в углах растра, оставшихся пустыми после варпа. Это НЕ «пригодная
    область», которой спецификация нормирует точность измерений: та задаётся двумя
    условиями — угол визирования на точку в пределах порога И знаменатель гомографии
    сохраняет знак (точка лежит перед камерой, а не за линией схода). Точка может
    входить в охват и при этом быть непригодной: содержимое туда легло, но легло под
    недопустимым углом. Обратного не бывает: вне охвата мерить нечего. Пригодность
    вычисляется отдельно при сборке конвейера и здесь сознательно не подменяется.
    """
    if mm_per_px <= 0:
        raise ValueError("mm_per_px должен быть положительным")

    h, w = image.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float) @ H_units.T
    corners = corners[:, :2] / corners[:, 2:3]

    scale = mm_per_rect_unit / mm_per_px          # единицы -> пиксели выхода
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    out_w = int(round((x1 - x0) * scale))
    out_h = int(round((y1 - y0) * scale))
    if not (1 <= out_w <= MAX_SIDE_PX and 1 <= out_h <= MAX_SIDE_PX):
        raise ValueError(
            f"невозможный размер ректифицированного растра {out_w}x{out_h}: "
            "проверьте mm_per_rect_unit и доверие к плоскости"
        )

    S = np.array([[scale, 0.0, -x0 * scale], [0.0, scale, -y0 * scale], [0.0, 0.0, 1.0]])
    H_total = S @ H_units

    warped = cv2.warpPerspective(image, H_total, (out_w, out_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(image, 255), H_total, (out_w, out_h),
                               flags=cv2.INTER_NEAREST) > 0
    ox, oy = origin_rect_units
    return Rectified(warped, H_total, mask, mm_per_px,
                     ((ox - x0) * scale, (oy - y0) * scale))
