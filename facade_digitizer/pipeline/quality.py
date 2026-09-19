"""Шлюз пригодности снимка. Отбраковывает, но не улучшает. Спецификация, п. 4.1.

Генеративное устранение смаза синтезирует границы, которых в кадре не было, а измерение
выполняется именно по границам: снимок стал бы визуально убедительным, а миллиметры —
уверенно неверными. Поэтому модуль только выносит вердикт.

Разделение ответственности. Негодный СНИМОК получает вердикт: это то, ради чего шлюз
существует. Негодный АРГУМЕНТ (нечисловое, бесконечное, отрицательное разрешение, доля
вне [0, 1]) — это поломка вызывающего кода, а не свойство снимка, и он поднимает
ValueError. Вердикт «брак» на такой вход был бы записан в выходной файл как свойство
кадра и стал бы неотличим от честной отбраковки плохого снимка; оператор переснял бы
кадр, а сломанный расчёт остался бы на месте. Спецификация, п. 6.3: значение без
происхождения в выход не попадает — выдуманная причина отбраковки тем более.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from facade_digitizer.schema import QualityReport


@dataclass(frozen=True)
class Thresholds:
    """Пороги шлюза. Все четыре подставляемы — калибровка не должна обойти ни один.

    Значение sharpness_min ПРЕДВАРИТЕЛЬНОЕ. Оно получено на синтетике (резкая сцена
    даёт около 1.2e-05, лёгкое размытие 8e-08) и подлежит замене после калибровки на
    реальных снимках — это отдельная задача этапа 1 спецификации. Абсолютный порог
    по вариации лапласиана ненадёжен и по другой причине: белый шум набирает по этой
    метрике больше, чем резкое изображение, поэтому шлюз не может отличить детали от
    шума и должен использоваться вместе с проверкой разрешения и угла.
    """
    sharpness_min: float = 1.0e-06
    gsd_max_mm_px: float = 5.0
    theta_p95_max_deg: float = 30.0
    usable_min_fraction: float = 0.5


DEFAULT = Thresholds()


def _number(name: str, value, low: float, high: float, low_inclusive: bool = True) -> float:
    """Аргумент-число: конечное и в осмысленном диапазоне. Иначе — поломка вызова.

    Проверка типа стоит до isfinite намеренно: math.isfinite(x) на нечисловом значении
    поднимает TypeError, а np.isfinite на строке — тоже TypeError, и без явной проверки
    сообщение было бы про внутренности шлюза, а не про негодный аргумент.
    """
    if value is None or isinstance(value, (str, bytes, bool, complex)):
        raise ValueError(f"{name}: ожидалось вещественное число, получено {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: ожидалось вещественное число, получено {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name}: ожидалось конечное число, получено {v}")
    if v > high or v < low or (v == low and not low_inclusive):
        edge = "(" if not low_inclusive else "["
        raise ValueError(f"{name}: значение {v} вне допустимого {edge}{low}, {high}]")
    return v


def _image(image) -> np.ndarray:
    """Кадр на входе шлюза: двумерный, непустой, из чисел и без нечисловых отсчётов.

    Пустой массив не бывает свойством снимка — это поломка вызывающего кода, и здесь
    действует то же разграничение, что для числовых аргументов. Формально кадр проверяет
    загрузка (задача 10), но шлюз стоит последним рубежом перед измерительным конвейером,
    и полагаться на безошибочность цепочки выше нет оснований.
    """
    # Проверка типа и вида отсчётов слита в одно условие сознательно: np.isfinite на
    # массиве объектов поднимает TypeError, а image.ndim на списке — AttributeError,
    # и отказ был бы про внутренности шлюза вместо негодного аргумента. Принадлежность
    # проверяется по кортежу, а не по строке "uif": пустая строка — её подстрока.
    kind = image.dtype.kind if isinstance(image, np.ndarray) else ""
    if kind not in ("u", "i", "f"):
        raise ValueError(f"изображение: ожидался числовой массив numpy, "
                         f"получено {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(f"изображение: ожидался двумерный полутоновый кадр, "
                         f"получено измерений: {image.ndim}")
    if image.size == 0:
        raise ValueError("изображение: кадр пуст, измерять нечего")
    if kind == "f" and not bool(np.isfinite(image).all()):
        raise ValueError("изображение: среди отсчётов есть нечисловые или бесконечные")
    return image


def sharpness(image: np.ndarray) -> float:
    """Вариация лапласиана, нормированная на дисперсию яркости.

    Нормировка делает метрику независимой от контраста и экспозиции: иначе тот же самый
    кадр, снятый темнее, получил бы меньшую резкость при неизменной геометрии границ.
    """
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return float(lap.var() / (float(image.std()) ** 2 + 1e-9))


def usable_fraction(theta_field_deg: np.ndarray, theta_max_deg: float = 30.0) -> float:
    """Доля кадра, удовлетворяющая угловому условию. Спецификация, п. 4.1."""
    return float(np.mean(np.asarray(theta_field_deg) <= theta_max_deg))


def assess(image: np.ndarray, gsd_min: float, gsd_max: float, theta_p95: float,
           usable: float = 1.0, thresholds: Thresholds | None = None) -> QualityReport:
    """Вердикт о пригодности снимка. Возвращает QualityReport (схема, раздел 10).

    Поднимает ValueError, если аргументы — числовые или сам кадр — не описывают реальный
    снимок: это поломка вызывающего кода, и маскировать её вердиктом нельзя. Сообщения
    об отказе не пересекаются между собой по подстрокам (см. тесты).
    """
    t = thresholds or DEFAULT

    gsd_min = _number("gsd_min", gsd_min, 0.0, 1.0e6, low_inclusive=False)
    gsd_max = _number("gsd_max", gsd_max, 0.0, 1.0e6, low_inclusive=False)
    theta_p95 = _number("theta_p95", theta_p95, 0.0, 90.0)
    usable = _number("usable", usable, 0.0, 1.0)
    if gsd_min > gsd_max:
        raise ValueError(f"границы разрешения переставлены: {gsd_min} > {gsd_max} мм/px")

    image = _image(image)

    reasons: list[str] = []
    sharp = sharpness(image)
    if not math.isfinite(sharp):
        raise ValueError(f"изображение: резкость неопределена ({sharp})")
    if sharp < t.sharpness_min:
        reasons.append(f"недостаточная резкость: {sharp:.2e} < {t.sharpness_min:.2e}")
    if gsd_max > t.gsd_max_mm_px:
        reasons.append(f"недостаточное разрешение: {gsd_max:.1f} мм/px > {t.gsd_max_mm_px}")
    if theta_p95 > t.theta_p95_max_deg:
        reasons.append(f"слишком крутой угол визирования: P95 = {theta_p95:.1f}°")
    if usable < t.usable_min_fraction:
        reasons.append(f"угловому условию удовлетворяет лишь {usable:.0%} кадра")

    if not reasons:
        verdict = "ok"
    elif any("разрешение" in r for r in reasons):
        verdict = "reject"
    else:
        verdict = "degraded"

    return QualityReport(
        verdict=verdict,
        sharpness=sharp,
        gsd_mm_px_min=gsd_min,
        gsd_mm_px_max=gsd_max,
        theta_field_deg_p95=theta_p95,
        reasons=reasons,
    )
