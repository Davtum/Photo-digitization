"""Масштаб просмотра → σ клика. План 3, задача 8; открытие 1 плана.

σ = √((σ_экрана / s)² + σ_кромки²): промах руки на экране и размытость кромки —
независимые источники, и складываются они квадратично, а не максимумом.
Следствие, которое тесты фиксируют, а не прячут: при 1:1 σ = 1.92 px, больше
константы плана 1 `DEFAULT_MARK_SIGMA_PX = 1.0`, которая есть предел СИЛЬНОГО
увеличения.
"""
import math

import pytest

from facade_digitizer.pipeline.run import DEFAULT_MARK_SIGMA_PX, DEFAULT_OPERATOR_SIGMA_PX
from facade_digitizer.ui.session import ClickedPoint
from facade_digitizer.ui.zoom import (
    RECOMMENDED_MIN_SCALE,
    SIGMA_EDGE_PX,
    SIGMA_SCREEN_PX,
    click_sigma,
    localisation_cost_mm,
    sigma_image_px,
)


def test_sigma_grows_as_the_view_gets_coarser():
    scales = [4.0, 2.0, 1.0, 0.5, 0.25]
    sigmas = [sigma_image_px(s) for s in scales]
    assert sigmas == sorted(sigmas)
    assert sigmas[-1] > 3 * sigmas[0]


def test_sigma_approaches_the_edge_term_under_strong_magnification():
    assert sigma_image_px(16.0) == pytest.approx(SIGMA_EDGE_PX, rel=0.01)
    for s in (1.0, 2.0, 8.0, 64.0, 1e6):
        assert sigma_image_px(s) >= SIGMA_EDGE_PX


def test_one_to_one_is_worse_than_the_plan_1_constant():
    """Следствие открытия 1: константа плана 1 — предел сильного увеличения."""
    assert sigma_image_px(1.0) == pytest.approx(math.hypot(SIGMA_SCREEN_PX, SIGMA_EDGE_PX))
    assert sigma_image_px(1.0) == pytest.approx(1.92, abs=0.005)
    assert sigma_image_px(1.0) > DEFAULT_MARK_SIGMA_PX


def test_documented_reduction_reproduces_the_reference_constant():
    """При уменьшении 2.75× (кадр 5280 px в окне 1920) — константа концов базы."""
    assert sigma_image_px(1 / 2.75) == pytest.approx(DEFAULT_OPERATOR_SIGMA_PX, rel=0.03)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_is_refused(bad):
    with pytest.raises(ValueError, match="масштаб"):
        sigma_image_px(bad)


def test_click_sigma_reads_the_scale_of_the_click():
    assert click_sigma(ClickedPoint(0.0, 0.0, view_scale=2.0)) == sigma_image_px(2.0)


def test_localisation_cost_matches_the_budget_term():
    """Цена в мм — одно слагаемое бюджета п. 6.1: √2·σ·GSD (две независимые кромки)."""
    assert localisation_cost_mm(1.0, gsd_mm_px=5.0) == pytest.approx(13.6, abs=0.1)
    assert localisation_cost_mm(4.0, gsd_mm_px=5.0) == pytest.approx(7.6, abs=0.1)


def test_recommended_magnification_brings_the_term_near_the_edge_limit():
    """Рекомендованное увеличение у углов — там, где σ уже близка к пределу кромки."""
    assert RECOMMENDED_MIN_SCALE == 2.0
    assert sigma_image_px(RECOMMENDED_MIN_SCALE) < 1.35 * SIGMA_EDGE_PX
