"""Анализ замера σ клика. План 3, задача 25."""
import numpy as np
import pytest

from facade_digitizer.click_study import (
    click_errors,
    fit_sigma_model,
    per_scale,
    repeatability,
)

NOMINAL = (0.5, 1.0, 2.0, 4.0)


def _session(points):
    return {"marks": [{"corners": [{"x": x, "y": y, "view_scale": s} for x, y, s in points]}]}


def _simulated(sigma_screen, sigma_edge, bias=(0.0, 0.0), n=400, seed=0):
    """Клики, порождённые моделью `ui.zoom`, по сетке истинных углов."""
    rng = np.random.default_rng(seed)
    truth = np.array([[200.0 + 300 * i, 300.0 + 200 * j] for i in range(10) for j in range(10)])
    points = []
    for s in NOMINAL:
        sigma = np.hypot(sigma_screen / s, sigma_edge)
        for k in range(n):
            tx, ty = truth[k % len(truth)]
            points.append((tx + bias[0] + rng.normal(0, sigma),
                           ty + bias[1] + rng.normal(0, sigma), s * 1.03))
    return _session(points), truth


def test_model_parameters_are_recovered():
    raw, truth = _simulated(1.64, 1.0)
    rows = per_scale(click_errors(raw, truth), NOMINAL)
    assert [r.scale for r in rows] == list(NOMINAL)          # 1.03·s → номинал
    fit = fit_sigma_model(rows)
    assert fit.sigma_screen_px == pytest.approx(1.64, rel=0.08)
    assert fit.sigma_edge_px == pytest.approx(1.0, rel=0.15)
    assert fit.residual_rel < 0.1


def test_bias_is_reported_apart_from_spread():
    raw, truth = _simulated(1.64, 1.0, bias=(0.5, -0.3))
    rows = per_scale(click_errors(raw, truth), NOMINAL)
    for r in rows:
        assert r.bias_x == pytest.approx(0.5, abs=0.25)
        assert r.bias_y == pytest.approx(-0.3, abs=0.25)
    unbiased = per_scale(click_errors(_simulated(1.64, 1.0)[0], truth), NOMINAL)
    for a, b in zip(rows, unbiased, strict=True):
        assert a.sigma == pytest.approx(b.sigma, rel=0.02)   # смещение не раздувает σ


def test_shape_is_not_checkable_without_magnification():
    raw, truth = _simulated(1.64, 1.0)
    rows = [r for r in per_scale(click_errors(raw, truth), NOMINAL) if r.scale <= 2.0]
    with pytest.raises(ValueError, match="не проверяема"):
        fit_sigma_model(rows)


def test_wrong_shape_shows_in_the_residual():
    """Если σ от масштаба не зависит вовсе, модель это покажет: σ_экрана ≈ 0."""
    raw, truth = _simulated(0.0, 1.5)
    fit = fit_sigma_model(per_scale(click_errors(raw, truth), NOMINAL))
    assert fit.sigma_screen_px < 0.3 and fit.sigma_edge_px == pytest.approx(1.5, rel=0.1)


def test_repeatability_within_operator():
    rng = np.random.default_rng(3)
    corners = [(500.0 + 100 * i, 400.0) for i in range(8)]
    sessions = [_session([(x + rng.normal(0, 0.8), y + rng.normal(0, 0.8), 2.0)
                          for x, y in corners]) for _ in range(30)]
    sigma, n = repeatability(sessions)
    assert n == 8 and sigma == pytest.approx(0.8, rel=0.12)
    with pytest.raises(ValueError, match="не менее двух"):
        repeatability(sessions[:1])


@pytest.mark.parametrize("blur", [0.8, 1.5, 3.0])
def test_edge_width_measures_blur_even_under_noise(blur):
    import cv2

    from facade_digitizer.click_study import EDGE_WIDTH_PER_SIGMA, edge_width_px

    img = np.full((200, 200), 60.0, np.float32)
    img[:, 100:] = 190.0
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img += np.random.default_rng(0).normal(0, 4.0, img.shape).astype(np.float32)
    width = edge_width_px(np.clip(img, 0, 255).astype(np.uint8), (99.5, 40), (99.5, 160))
    assert width == pytest.approx(EDGE_WIDTH_PER_SIGMA * blur, rel=0.2)


def test_edge_width_refuses_a_flat_region():
    from facade_digitizer.click_study import edge_width_px

    with pytest.raises(ValueError, match="не кромка"):
        edge_width_px(np.full((100, 100), 128, np.uint8), (50, 20), (50, 80))


def test_study_script_round_trip(tmp_path, monkeypatch, capsys):
    """`make` → смоделированные сессии оператора → `analyze` печатает подгонку."""
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "click_study_script", Path(__file__).resolve().parents[1] / "scripts" / "click_study.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "BLUR_LEVELS", (1.5,))
    assert script.main(["make", "--out-dir", str(tmp_path), "--blocks", "1"]) == 0
    set_dir = tmp_path / "размытие_1.5" / "набор00"
    rng = np.random.default_rng(5)
    for record in json.loads((set_dir / "истина.json").read_text(encoding="utf-8")):
        corners = script._truth_corners(record)
        points = [{"x": x + rng.normal(0, np.hypot(1.64 / s, 1.0)),
                   "y": y + rng.normal(0, np.hypot(1.64 / s, 1.0)), "view_scale": s}
                  for s in script.NOMINAL_SCALES for x, y in corners for _ in range(3)]
        raw = {"image": {"path": str(set_dir / f"{record['name']}.png")},
               "marks": [{"corners": points}]}
        (set_dir / f"{record['name']}.session.json").write_text(json.dumps(raw),
                                                                encoding="utf-8")
    assert script.main(["analyze", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "σ_экрана" in out and "ширина кромки" in out
