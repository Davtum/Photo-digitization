"""Показатели согласия разметчиков. План 3, задача 26, шаг 1."""
import numpy as np
import pytest

from facade_digitizer.metrics import (
    bland_altman,
    disagreement_fraction,
    fleiss_kappa,
    fleiss_kappa_counts,
    limits_of_agreement,
)

#: Пример Флейса (Fleiss, 1971; воспроизведён в учебниках): 10 объектов,
#: 14 разметчиков, 5 категорий; κ = 0.210.
FLEISS_EXAMPLE = np.array([
    [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])


def test_full_agreement_gives_zero_limits_and_kappa_one():
    sizes = np.array([1460.0, 900.0, 2100.0, 1200.0])
    values = np.column_stack([sizes] * 3)
    ba = bland_altman(values[:, 0], values[:, 1])
    assert ba.bias == 0.0 and ba.sd == 0.0 and ba.lower == ba.upper == 0.0
    agree = limits_of_agreement(values)
    assert agree.sigma_within == 0.0 and np.all(agree.rater_bias == 0.0)
    labels = [["sharp_wall_edge", "surround"][i % 2] for i in range(4)]
    assert fleiss_kappa([[x] * 3 for x in labels]) == pytest.approx(1.0)
    assert disagreement_fraction([[x] * 3 for x in labels]) == 0.0


def test_known_shift_and_noise_are_recovered():
    rng = np.random.default_rng(1)
    truth = rng.uniform(800, 2400, 4000)
    a = truth + rng.normal(0, 3.0, truth.size)
    b = truth + 5.0 + rng.normal(0, 3.0, truth.size)
    ba = bland_altman(a, b)
    assert ba.bias == pytest.approx(-5.0, abs=0.2)
    assert ba.sd == pytest.approx(3.0 * np.sqrt(2), rel=0.05)
    assert ba.upper - ba.lower == pytest.approx(2 * 1.96 * ba.sd)
    agree = limits_of_agreement(np.column_stack([a, a * 0 + truth + rng.normal(0, 3, a.size),
                                                 b]))
    # σ внутри объекта: три разметчика с σ = 3 и один сдвинут на 5 мм.
    assert agree.sigma_within == pytest.approx(np.sqrt(9 + 25 / 3), rel=0.05)
    assert agree.loa_half_width == pytest.approx(1.96 * np.sqrt(2) * agree.sigma_within)


def test_one_outlying_rater_is_named_by_its_bias():
    rng = np.random.default_rng(2)
    truth = rng.uniform(800, 2400, 200)
    values = np.column_stack([truth + rng.normal(0, 2, truth.size) for _ in range(4)])
    values[:, 2] += 12.0
    bias = limits_of_agreement(values).rater_bias
    assert int(np.argmax(np.abs(bias))) == 2
    assert bias[2] == pytest.approx(9.0, abs=0.6)     # 12 · (1 − 1/4)
    labels = [["sharp_wall_edge"] * 3 + ["surround"] for _ in range(10)]
    assert disagreement_fraction(labels) == 1.0
    assert fleiss_kappa(labels) < 0.0                  # согласие хуже случайного


def test_fleiss_kappa_matches_the_published_example():
    assert fleiss_kappa_counts(FLEISS_EXAMPLE) == pytest.approx(0.210, abs=5e-4)


def test_degenerate_inputs_are_refused_by_name():
    with pytest.raises(ValueError, match="не определена"):
        fleiss_kappa([["a", "a"], ["a", "a"]])
    with pytest.raises(ValueError, match="одно и то же число"):
        fleiss_kappa_counts(np.array([[2, 1], [1, 1]]))
    with pytest.raises(ValueError, match="2×2"):
        limits_of_agreement(np.ones((1, 3)))
    with pytest.raises(ValueError, match="не менее двух"):
        bland_altman([1.0], [1.0])
