import numpy as np
import pytest

from facade_digitizer.metrics import (
    coverage,
    error_stats,
    interval_score,
    mean_sharpness,
)

# Квантили нормального распределения заданы константами, а не вызовом scipy: ожидаемое
# значение в тесте не должно вычисляться тем же кодом, что и проверяемая величина.
Z_975 = 1.959963984540054   # norm.ppf(0.975), alpha = 0.05
Z_900 = 1.2815515655446004  # norm.ppf(0.900), alpha = 0.20


def test_error_stats_on_known_arrays():
    truth = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    measured = truth + np.array([10.0, -10.0, 10.0, -10.0])
    st = error_stats(measured, truth)
    assert st.n == 4
    assert st.rmse == pytest.approx(10.0)
    assert st.bias == pytest.approx(0.0)
    assert st.sigma == pytest.approx(11.547, rel=1e-3)


def test_bias_is_separated_from_spread():
    """Систематическое смещение сообщается отдельно. Спецификация, п. 12.1."""
    truth = np.array([1000.0, 2000.0, 3000.0])
    measured = truth + 25.0
    st = error_stats(measured, truth)
    assert st.bias == pytest.approx(25.0)
    assert st.sigma == pytest.approx(0.0, abs=1e-9)
    assert st.rmse == pytest.approx(25.0)


def test_p95_is_reported():
    truth = np.zeros(100)
    measured = np.linspace(0.0, 100.0, 100)
    st = error_stats(measured, truth)
    assert 90.0 < st.p95 <= 100.0


def test_coverage_matches_nominal_for_correct_sigma():
    rng = np.random.default_rng(0)
    truth = np.zeros(20000)
    sigma = 10.0
    measured = rng.normal(0.0, sigma, 20000)
    got = coverage(measured, truth, np.full(20000, sigma), k=1.0)
    assert got == pytest.approx(0.683, abs=0.02)


def test_coverage_can_be_gamed_by_inflating_sigma():
    """Именно поэтому острота обязательна рядом с покрытием. Спецификация, п. 12.3."""
    rng = np.random.default_rng(1)
    truth = np.zeros(5000)
    measured = rng.normal(0.0, 10.0, 5000)
    inflated = np.full(5000, 200.0)
    assert coverage(measured, truth, inflated, k=1.0) > 0.99
    assert mean_sharpness(inflated) == pytest.approx(200.0)


def test_interval_score_penalises_both_width_and_miss():
    """IS = (u−l) + (2/α)(l−y)·1{y<l} + (2/α)(y−u)·1{y>u}. Спецификация, п. 12.3."""
    alpha, sigma = 0.05, 10.0
    z = Z_975

    # Попадание: штрафа нет, остаётся только ширина.
    hit = interval_score(np.array([0.0]), np.array([0.0]), np.array([sigma]), alpha)
    assert hit == pytest.approx(2.0 * z * sigma)

    # Промах: ширина плюс штраф (2/alpha) * (y - u).
    y = 500.0
    miss = interval_score(np.array([0.0]), np.array([y]), np.array([sigma]), alpha)
    expected = 2.0 * z * sigma + (2.0 / alpha) * (y - z * sigma)
    assert miss == pytest.approx(expected)
    assert miss > 100 * hit

    # Ширина наказывается отдельно, при одинаковом попадании.
    wide = interval_score(np.array([0.0]), np.array([0.0]), np.array([100.0]), alpha)
    assert wide > hit


def test_interval_score_penalises_underestimate_too():
    """Штраф считается двумя ветками; выше проверена только верхняя."""
    alpha, sigma = 0.05, 10.0
    z = Z_975
    y = -500.0
    got = interval_score(np.array([0.0]), np.array([y]), np.array([sigma]), alpha)
    expected = 2.0 * z * sigma + (2.0 / alpha) * (-z * sigma - y)
    assert got == pytest.approx(expected)


def test_interval_score_respects_alpha():
    """Ширина интервала и вес штрафа заданы alpha, а не зашиты на 0.05."""
    alpha, sigma = 0.2, 10.0
    z = Z_900

    hit = interval_score(np.array([0.0]), np.array([0.0]), np.array([sigma]), alpha)
    assert hit == pytest.approx(2.0 * z * sigma)

    y = 100.0
    miss = interval_score(np.array([0.0]), np.array([y]), np.array([sigma]), alpha)
    assert miss == pytest.approx(2.0 * z * sigma + (2.0 / alpha) * (y - z * sigma))

    # При том же промахе более узкий номинальный интервал (alpha=0.2) даёт другой счёт,
    # чем стандартный (alpha=0.05) — значит alpha действительно участвует.
    assert miss != pytest.approx(
        interval_score(np.array([0.0]), np.array([y]), np.array([sigma]), 0.05)
    )


def test_mean_sharpness_is_the_mean_not_an_extremum():
    sigmas = np.array([1.0, 2.0, 30.0])      # mean 11, median 2, min 1, max 30
    assert mean_sharpness(sigmas) == pytest.approx(11.0)


def test_coverage_widens_with_k():
    rng = np.random.default_rng(7)
    truth = np.zeros(20000)
    measured = rng.normal(0.0, 10.0, 20000)
    sigmas = np.full(20000, 10.0)
    assert coverage(measured, truth, sigmas, k=1.0) == pytest.approx(0.683, abs=0.02)
    assert coverage(measured, truth, sigmas, k=2.0) == pytest.approx(0.954, abs=0.01)


def test_rmse_differs_from_mean_absolute_error():
    truth = np.zeros(4)
    measured = np.array([1.0, 1.0, 1.0, 9.0])   # MAE 3.0, RMSE sqrt(84/4) = sqrt(21)
    st = error_stats(measured, truth)
    assert st.rmse == pytest.approx(4.582576, rel=1e-5)
    assert st.rmse > 1.5 * float(np.mean(np.abs(measured - truth)))


def test_p95_has_exact_analytic_value():
    """На 0..100 с шагом 1 линейная интерполяция numpy даёт ровно 95.0."""
    truth = np.zeros(101)
    measured = np.arange(101.0)   # |ошибки| = 0..100, индекс 0.95*(101-1) = 95
    st = error_stats(measured, truth)
    assert st.p95 == pytest.approx(95.0)


def test_single_sample_reports_zero_spread_not_nan():
    """При n=1 дисперсия по ddof=1 не определена; модуль обязан вернуть 0.0, а не nan."""
    st = error_stats(np.array([5.0]), np.array([0.0]))
    assert st.n == 1
    assert st.bias == pytest.approx(5.0)
    assert st.rmse == pytest.approx(5.0)
    assert st.sigma == 0.0
    assert not np.isnan(st.sigma)


def test_empty_sample_is_rejected():
    with pytest.raises(ValueError):
        error_stats(np.array([]), np.array([]))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        error_stats(np.array([1.0, 2.0]), np.array([1.0]))
