"""Шлюз пригодности снимка. Отбраковывает, но не улучшает. Спецификация, п. 4.1.

Генеративное устранение смаза синтезирует границы, которых в кадре не было, а измерение
выполняется именно по границам: снимок стал бы визуально убедительным, а миллиметры —
уверенно неверными. Поэтому модуль только выносит вердикт.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from facade_digitizer.schema import QualityReport


@dataclass(frozen=True)
class Thresholds:
    """Пороги шлюза.

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


DEFAULT = Thresholds()


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
    """Вердикт о пригодности снимка. Возвращает QualityReport (схема, раздел 10)."""
    t = thresholds or DEFAULT
    reasons: list[str] = []

    sharp = sharpness(image)
    if sharp < t.sharpness_min:
        reasons.append(f"недостаточная резкость: {sharp:.2e} < {t.sharpness_min:.2e}")
    if gsd_max > t.gsd_max_mm_px:
        reasons.append(f"недостаточное разрешение: {gsd_max:.1f} мм/px > {t.gsd_max_mm_px}")
    if theta_p95 > t.theta_p95_max_deg:
        reasons.append(f"слишком крутой угол визирования: P95 = {theta_p95:.1f}°")
    if usable < 0.5:
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
