"""Ректификация с явной привязкой растра к миллиметрам."""
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectified:
    """Ректифицированный растр вместе с конвенцией, связывающей его с миллиметрами.

    `H` — преобразование «изображение -> ректифицированные ПИКСЕЛИ», уже с учётом
    масштаба и сдвига начала координат; именно её, а не исходную `H_units`, обязан
    применять потребитель, измеряющий в этом растре.

    `origin_rect_px` — образ точки `origin_rect_units` в пикселях этого растра.
    Это НАЧАЛО ОТСЧЁТА, переданное вызывающим, а не какой-либо угол фасада сам по
    себе: `rectify` фасада не знает и знать не может.
    """

    image: np.ndarray
    H: np.ndarray                 # изображение -> ректифицированные ПИКСЕЛИ
    valid_mask: np.ndarray
    mm_per_px: float
    origin_rect_px: tuple


MAX_SIDE_PX = 20000


def _positive_finite(name, value):
    """Положительное конечное число, отвергаемое ПОИМЁННО.

    Отказ обязан называть виноватый аргумент. Без этого нечисловое или нулевое
    `mm_per_rect_unit` доезжало до проверки размера растра и сообщало о себе не там,
    где находится: `nan` проваливал `1 <= out_w` (сравнение с `nan` ложно), а
    вызывающий шёл искать неисправность в гомографии и в доверии к плоскости.

    Тип проверяется ДО значения: `math.isfinite` на строке поднимает TypeError, а
    `value <= 0` на `None` — TypeError из внутренностей, и отказ снова оказался бы
    про устройство функции вместо негодного аргумента. Род отсчётов проверяется по
    кортежу родов, а не вхождением в строку "uif": пустая строка — подстрока любой.
    Род `b` (булев) в набор не входит сознательно: `float(True)` дал бы молчаливую
    единицу измерения вместо отказа. Нульмерность обязательна — массив из двух
    чисел не есть масштаб.
    """
    probe = np.asarray(value)
    if probe.dtype.kind not in ("u", "i", "f") or probe.ndim != 0:
        raise ValueError(f"{name} должен быть положительным конечным числом, "
                         f"получено {type(value).__name__}")
    v = float(probe)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} должен быть положительным конечным числом, "
                         f"получено {v}")
    return v


def _grayscale(image):
    """Кадр ректификации: двумерный полутоновый, непустой, из чисел.

    Конвейер работает с одноканальным изображением, загрузка (задача 10) к нему и
    приводит. Трёхканальный кадр здесь не просто нежелателен: `np.full_like` дал бы
    трёхканальную маску охвата, и потребитель, ожидающий двумерную, получил бы её
    молча. Поэтому иной вход отвергается, а не приводится.
    """
    kind = image.dtype.kind if isinstance(image, np.ndarray) else ""
    if kind not in ("u", "i", "f"):
        raise ValueError(f"изображение: ожидался числовой массив numpy, "
                         f"получено {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(f"изображение: ожидался двумерный полутоновый кадр, "
                         f"получено измерений: {image.ndim}")
    if image.size == 0:
        raise ValueError("изображение: кадр пуст, ректифицировать нечего")
    return image


def _frame_bounds_units(image, H_units):
    """Углы растра в единицах: образ четырёх углов кадра, приведённый к минимуму и максимуму."""
    h, w = image.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float) @ H_units.T
    corners = corners[:, :2] / corners[:, 2:3]
    return corners.min(axis=0), corners.max(axis=0)


def attainable_mm_per_px(image, H_units, mm_per_rect_unit):
    """Границы `mm_per_px`, при которых растр получается возможного размера.

    Возвращает пару (наименьшее, наибольшее), ОБЕ границы достижимы: на нижней
    длинная сторона растра равна ровно `MAX_SIDE_PX`, на верхней короткая — ровно
    одному пикселю. Величины нужны не для украшения сообщения об отказе: без них
    вызывающий подбирает масштаб вслепую, а именно слепой подбор и породил растр
    16x16 в редакции 1.

    Вырожденный образ кадра (нулевой размах хотя бы по одной оси) даёт перевёрнутую
    пару — годного значения нет вовсе; вызывающий распознаёт это как `lo > hi`.
    """
    image = _grayscale(image)
    mm_per_rect_unit = _positive_finite("mm_per_rect_unit", mm_per_rect_unit)
    lo_xy, hi_xy = _frame_bounds_units(image, H_units)
    span = hi_xy - lo_xy
    return (float(span.max() * mm_per_rect_unit / MAX_SIDE_PX),
            float(span.min() * mm_per_rect_unit))


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

    `origin_rect_units` — точка РЕКТИФИЦИРОВАННОЙ системы координат (той, в которой
    работает `H_units`), а не угол фасада: по умолчанию это её нуль, лежащий там,
    куда его поставила гомография, и с началом фасада не совпадающий. Началом
    отсчёта служит опорная точка `foot_point_in_rectified` либо образ угла фасада —
    вызывающий обязан передать её сам. Перепутанное здесь начало сдвигает ВСЕ
    измеренные положения разом и на одну и ту же величину, то есть выглядит как
    правдоподобный результат.
    """
    image = _grayscale(image)
    mm_per_rect_unit = _positive_finite("mm_per_rect_unit", mm_per_rect_unit)
    mm_per_px = _positive_finite("mm_per_px", mm_per_px)

    (x0, y0), (x1, y1) = _frame_bounds_units(image, H_units)

    scale = mm_per_rect_unit / mm_per_px          # единицы -> пиксели выхода
    out_w = int(round((x1 - x0) * scale))
    out_h = int(round((y1 - y0) * scale))
    if not (1 <= out_w <= MAX_SIDE_PX and 1 <= out_h <= MAX_SIDE_PX):
        lo, hi = attainable_mm_per_px(image, H_units, mm_per_rect_unit)
        reachable = (f"годится mm_per_px от {lo:.4g} до {hi:.4g} мм"
                     if lo <= hi else "годного mm_per_px нет вовсе: образ кадра вырожден")
        raise ValueError(
            f"невозможный размер ректифицированного растра {out_w}x{out_h} "
            f"при mm_per_px={mm_per_px:.4g}: {reachable}; "
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
