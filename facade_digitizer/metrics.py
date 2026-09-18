"""Метрики точности и калибровки неопределённости. Спецификация, раздел 12."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ErrorStats:
    n: int
    rmse: float
    p95: float
    bias: float
    sigma: float


def _check(measured: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    measured = np.asarray(measured, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    if measured.shape != truth.shape:
        raise ValueError(
            f"длины не совпадают: измерено {measured.shape}, эталон {truth.shape}"
        )
    if measured.size == 0:
        raise ValueError("пустая выборка")
    return measured, truth


def error_stats(measured: np.ndarray, truth: np.ndarray) -> ErrorStats:
    """RMSE, P95, смещение и разброс. Смещение сообщается отдельно от разброса."""
    measured, truth = _check(measured, truth)
    err = measured - truth
    bias = float(err.mean())
    return ErrorStats(
        n=int(err.size),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        p95=float(np.percentile(np.abs(err), 95)),
        bias=bias,
        sigma=float(np.std(err, ddof=1)) if err.size > 1 else 0.0,
    )


def coverage(
    measured: np.ndarray, truth: np.ndarray, sigmas: np.ndarray, k: float = 1.0
) -> float:
    """Доля эталонных значений, попавших в заявленный интервал ±k·σ.

    Бессмысленна без mean_sharpness: завышение σ поднимает покрытие до единицы.
    """
    measured, truth = _check(measured, truth)
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if sigmas.shape != measured.shape:
        raise ValueError("массив σ не совпадает по длине с выборкой")
    if np.any(sigmas <= 0):
        raise ValueError("σ должна быть положительной")
    return float(np.mean(np.abs(measured - truth) <= k * sigmas))


def mean_sharpness(sigmas: np.ndarray) -> float:
    """Средняя заявленная σ — острота. Сообщается всегда рядом с покрытием."""
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if sigmas.size == 0:
        raise ValueError("пустой массив σ")
    return float(sigmas.mean())


def interval_score(
    measured: np.ndarray, truth: np.ndarray, sigmas: np.ndarray, alpha: float = 0.05
) -> float:
    """Interval score — proper scoring rule: штрафует и ширину, и промах.

    IS = (u − l) + (2/α)(l − y)·1{y<l} + (2/α)(y − u)·1{y>u}
    Меньше — лучше. Спецификация, п. 12.3.
    """
    measured, truth = _check(measured, truth)
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if sigmas.shape != measured.shape:
        raise ValueError("массив σ не совпадает по длине с выборкой")

    z = norm.ppf(1.0 - alpha / 2.0)
    lower, upper = measured - z * sigmas, measured + z * sigmas
    width = upper - lower
    below = (2.0 / alpha) * np.clip(lower - truth, 0.0, None)
    above = (2.0 / alpha) * np.clip(truth - upper, 0.0, None)
    return float(np.mean(width + below + above))
