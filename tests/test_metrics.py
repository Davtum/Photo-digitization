import numpy as np
import pytest

from facade_digitizer.metrics import (
    coverage,
    error_stats,
    interval_score,
    mean_sharpness,
)


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
    from scipy.stats import norm

    alpha, sigma = 0.05, 10.0
    z = norm.ppf(1.0 - alpha / 2.0)

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


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        error_stats(np.array([1.0, 2.0]), np.array([1.0]))
