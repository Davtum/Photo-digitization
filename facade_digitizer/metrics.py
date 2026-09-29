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


# --- Согласие разметчиков без эталона (план 3, задача 26; спецификация, п. 8.2) -----
#
# ICC по габаритам сюда не входит намеренно: размеры проёмов различаются на метры,
# межпроёмная дисперсия топит разметочную, и ICC ≈ 0.999 при любом качестве
# разметки. Нужны показатели в миллиметрах и по категориям.

@dataclass(frozen=True)
class BlandAltman:
    n: int
    bias: float        # среднее (a − b)
    sd: float          # разброс разностей
    lower: float       # bias − 1.96·sd
    upper: float       # bias + 1.96·sd


def bland_altman(a: np.ndarray, b: np.ndarray) -> BlandAltman:
    """Пределы согласия двух разметчиков по одной величине (кромке), мм."""
    a, b = _check(a, b)
    if a.size < 2:
        raise ValueError("для пределов согласия нужно не менее двух объектов")
    diff = a - b
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return BlandAltman(n=int(diff.size), bias=bias, sd=sd,
                       lower=bias - 1.96 * sd, upper=bias + 1.96 * sd)


@dataclass(frozen=True)
class MultiRaterAgreement:
    n_items: int
    n_raters: int
    sigma_within: float        # σ разметки внутри объекта (по всем разметчикам)
    loa_half_width: float      # 1.96·√2·σ_within: предел согласия ЛЮБЫХ двух
    rater_bias: np.ndarray     # смещение каждого разметчика от среднего по объекту


def limits_of_agreement(values: np.ndarray) -> MultiRaterAgreement:
    """Пределы согласия нескольких разметчиков: `values` — объекты × разметчики, мм.

    σ внутри объекта — корень средней по объектам несмещённой дисперсии; разность
    двух разметчиков тогда имеет разброс √2·σ, и предел согласия пары —
    ±1.96·√2·σ (Бланд — Альтман для повторных измерений). Смещение разметчика —
    его среднее отклонение от среднего по объекту: выбивающийся разметчик виден
    по нему, а не растворяется в общем разбросе.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] < 2:
        raise ValueError(f"нужна таблица объекты × разметчики не меньше 2×2, получено "
                         f"{v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("в таблице согласия есть неопределённые значения")
    item_mean = v.mean(axis=1, keepdims=True)
    sigma = float(np.sqrt(v.var(axis=1, ddof=1).mean()))
    return MultiRaterAgreement(n_items=int(v.shape[0]), n_raters=int(v.shape[1]),
                               sigma_within=sigma,
                               loa_half_width=1.96 * np.sqrt(2.0) * sigma,
                               rater_bias=(v - item_mean).mean(axis=0))


def fleiss_kappa_counts(counts: np.ndarray) -> float:
    """Каппа Флейса по таблице «объекты × категории» с числом отнесших разметчиков.

    Число разметчиков у всех объектов одинаково. Если все отнесли всё к одной
    категории, ожидаемое согласие равно единице и каппа не определена — отказ с
    причиной, а не 1 или NaN.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or c.shape[0] < 1:
        raise ValueError("нужна таблица объекты × категории")
    raters = c.sum(axis=1)
    if not np.allclose(raters, raters[0]) or raters[0] < 2:
        raise ValueError("у каждого объекта должно быть одно и то же число разметчиков, ≥ 2")
    n = raters[0]
    p_j = c.sum(axis=0) / c.sum()
    p_i = ((c * (c - 1)).sum(axis=1)) / (n * (n - 1))
    p_bar, p_e = float(p_i.mean()), float((p_j ** 2).sum())
    if np.isclose(p_e, 1.0):
        raise ValueError("все отнесли всё к одной категории: каппа не определена")
    return (p_bar - p_e) / (1.0 - p_e)


def fleiss_kappa(labels) -> float:
    """Каппа Флейса по меткам: `labels` — объекты × разметчики (любые метки)."""
    table = np.asarray(labels, dtype=object)
    if table.ndim != 2:
        raise ValueError("нужна таблица объекты × разметчики")
    categories = sorted({str(x) for x in table.ravel()})
    index = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((table.shape[0], len(categories)))
    for row, item in enumerate(table):
        for label in item:
            counts[row, index[str(label)]] += 1
    return fleiss_kappa_counts(counts)


def disagreement_fraction(labels) -> float:
    """Доля объектов, по которым разметчики НЕ единодушны (например, выбор кромки:
    рама против кромки стены)."""
    table = np.asarray(labels, dtype=object)
    if table.ndim != 2 or table.shape[0] == 0:
        raise ValueError("нужна непустая таблица объекты × разметчики")
    return float(np.mean([len({str(x) for x in row}) > 1 for row in table]))
