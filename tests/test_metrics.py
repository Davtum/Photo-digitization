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
    truth = np.array([0.0])
    tight_correct = interval_score(np.array([0.0]), truth, np.array([10.0]))
    wide_correct = interval_score(np.array([0.0]), truth, np.array([100.0]))
    tight_miss = interval_score(np.array([500.0]), truth, np.array([10.0]))
    assert tight_correct < wide_correct
    assert tight_correct < tight_miss


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        error_stats(np.array([1.0, 2.0]), np.array([1.0]))
