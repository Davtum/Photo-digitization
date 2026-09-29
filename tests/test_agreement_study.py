"""Сведение экспортов разных разметчиков. План 3, задача 26."""
import numpy as np
import pytest

from facade_digitizer.agreement_study import agreement, edges_mm, match_elements

MM_PER_PX = 2.0          # эталон: единичная гомография, 2 мм на пиксель


def _element(i, corners, edge_type="sharp_wall_edge", mounting="embedded"):
    return {"id": f"w_{i:03d}", "edge_type": edge_type, "mounting": mounting,
            "contour_px": [{"image_id": "img_0", "points": corners.tolist(), "sigma_px": 1.0}]}


def _model(elements):
    return {"images": [{"homography": {"H": np.eye(3).tolist(), "origin_rect_px": [0.0, 0.0]}}],
            "facade": {"mm_per_rectified_px": MM_PER_PX}, "elements": elements}


TRUTH = [np.array([[100.0 + 150 * (i % 6), 900.0 - 200 * (i // 6)],
                   [180.0 + 150 * (i % 6), 900.0 - 200 * (i // 6)],
                   [180.0 + 150 * (i % 6), 780.0 - 200 * (i // 6)],
                   [100.0 + 150 * (i % 6), 780.0 - 200 * (i // 6)]]) for i in range(30)]


def _raters(noise_px=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return [[c + rng.normal(0, noise_px, c.shape) for c in TRUTH] for _ in range(3)]


def test_edges_are_converted_with_the_reference_geometry():
    e = edges_mm(TRUTH[0], _model([]))
    assert e == pytest.approx({"left": 200.0, "right": 360.0, "bottom": -1800.0,
                               "top": -1560.0})


def test_edge_limits_follow_the_click_noise():
    raters = _raters()
    report = agreement([_model([_element(i, c) for i, c in enumerate(r)]) for r in raters])
    assert report.items == 30 and report.missing == 0
    for stats in report.edges.values():
        # Кромка — среднее двух углов: σ = 1 px · 2 мм / √2.
        assert stats.sigma_within == pytest.approx(MM_PER_PX / np.sqrt(2), rel=0.2)
    assert report.choice_disagreement == 0.0
    assert report.kappa_edge_type is None          # все согласны на одной категории


def test_frame_choice_and_missing_openings_are_reported_apart():
    raters = _raters()
    for i in range(5):                              # третий кликнул раму: +30 px внутрь
        raters[2][i][[0, 3], 0] += 30.0
    models = [_model([_element(i, c) for i, c in enumerate(r)]) for r in raters]
    models[1]["elements"].pop(10)                   # второй пропустил проём
    report = agreement(models)
    assert report.missing == 1 and report.items == 29
    assert report.choice_disagreement == pytest.approx(5 / 29)
    assert int(np.argmax(np.abs(report.edges["left"].rater_bias))) == 2


def test_categorical_disagreement_gives_kappa():
    raters = _raters()
    labels = [["sharp_wall_edge", "surround"][i % 2] for i in range(30)]
    models = []
    for r, rater in enumerate(raters):
        elements = []
        for i, c in enumerate(rater):
            label = labels[i] if not (r == 2 and i < 6) else "unknown"
            elements.append(_element(i, c, edge_type=label))
        models.append(_model(elements))
    report = agreement(models)
    assert 0.0 < report.kappa_edge_type < 1.0
    assert report.edge_type_disagreement == pytest.approx(6 / 30)


def test_matching_needs_every_rater():
    models = [_model([_element(0, TRUTH[0])]), _model([_element(0, TRUTH[5])])]
    items, missing = match_elements(models)
    assert items == [] and missing == 1
    with pytest.raises(ValueError, match="не менее двух"):
        agreement(models[:1])


def test_several_images_are_pooled_with_their_own_geometry():
    from facade_digitizer.agreement_study import agreement_many

    raters = _raters()
    first = [_model([_element(i, c) for i, c in enumerate(r[:15])]) for r in raters]
    second = [_model([_element(i, c) for i, c in enumerate(r[15:])]) for r in raters]
    for m in second:                                  # другой масштаб эталона
        m["facade"]["mm_per_rectified_px"] = 4.0
    pooled = agreement_many([first, second])
    assert pooled.items == 30
    # σ — смесь 2 и 4 мм/px: между их значениями по отдельности.
    assert MM_PER_PX / np.sqrt(2) * 0.8 < pooled.edges["left"].sigma_within < 4.0 / np.sqrt(2)


def test_agreement_script_reads_rater_directories(tmp_path, capsys):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "agreement_script", Path(__file__).resolve().parents[1] / "scripts" / "agreement.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    dirs = []
    for k, rater in enumerate(_raters()):
        d = tmp_path / f"op-{k:08x}"
        d.mkdir()
        (d / "фасад.json").write_text(json.dumps(
            _model([_element(i, c) for i, c in enumerate(rater)])), encoding="utf-8")
        (d / "фасад.marks.json").write_text("[]", encoding="utf-8")   # не экспорт модели
        dirs.append(str(d))
    assert script.main(dirs) == 0
    out = capsys.readouterr().out
    assert "сопоставлено проёмов 30" in out and "предел согласия" in out
