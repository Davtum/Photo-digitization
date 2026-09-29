"""Опорная база и начало отсчёта. План 3, задача 11; спецификация, п. 6.3 и п. 7."""
import json
import math
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

FACADE_W = 20000.0


def _wait(qapp, predicate, seconds=60.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    return predicate()


@pytest.fixture(scope="module")
def scene_file(tmp_path_factory):
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    path = tmp_path_factory.mktemp("база") / "facade.png"
    save_image(path, sc.render())
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    return sc, path, base


@pytest.fixture
def window(qapp, scene_file):
    from facade_digitizer.ui.window import MainWindow

    _sc, path, _base = scene_file
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w.open_image(path)
    assert _wait(qapp, lambda: w.session.frame is not None)
    yield w
    w.close()


def _base(w, base, scales=(0.4, 0.4), **values):
    w.set_mode("base")
    for pt, s in zip(base, scales):
        w.handle_click(*pt, s)
    w.side.base.set_values(span_mm=FACADE_W, **values)
    w.apply_base()


def test_base_sigma_uses_per_end_click_sigma(window, scene_file):
    from facade_digitizer.ui.zoom import click_sigma, sigma_image_px

    _sc, _path, base = scene_file
    _base(window, base, scales=(1.0, 0.25))
    ref = window.session.operator_reference(click_sigma)
    assert ref.end_sigma_px == (sigma_image_px(1.0), sigma_image_px(0.25))
    assert "грубее 1:1" in window.side.base.ends_label.text()


def test_span_length_uncertainty_enters_sigma_rel(window, scene_file):
    _sc, _path, base = scene_file
    _base(window, base)
    plain = window.session.scale.sigma_rel
    _base(window, base, span_sigma_mm=10.0)
    with_len = window.session.scale.sigma_rel
    assert with_len == pytest.approx(math.hypot(plain, 10.0 / FACADE_W), rel=1e-12)


def test_sigma_rel_shown_matches_the_output_file(window, scene_file):
    _sc, _path, base = scene_file
    _base(window, base, span_sigma_mm=5.0)
    shown = window.side.base.result.text()
    model = window.session.compute_elements(lambda p: 1.0)
    scale = model.facade.scale
    assert f"{scale.sigma_rel:.3%}" in shown
    assert scale.span_sigma_mm == 5.0


def test_needs_operator_branch_carries_the_span_uncertainty(tmp_path):
    """Ветвь отказа считает σ по разрешению вдоль базы (`gsd_along_base`) — и в ней
    погрешность длины тоже обязана участвовать."""
    from facade_digitizer.pipeline import run
    from facade_digitizer.pipeline.io import save_image

    rng = np.random.default_rng(0)
    path = tmp_path / "пусто.png"
    save_image(path, (128 + rng.normal(0, 3, (600, 800))).clip(0, 255).astype(np.uint8))
    common = {"origin_px": (10.0, 500.0), "span_px": ((10.0, 500.0), (790.0, 500.0)),
              "span_mm": 10000.0}
    plain = run.process(path, operator_reference=run.OperatorReference(**common),
                        raster_mm_per_px=10.0)
    with_len = run.process(path, raster_mm_per_px=10.0,
                           operator_reference=run.OperatorReference(**common,
                                                                    span_sigma_mm=20.0))
    assert with_len.images[0].rectification.needs_operator
    assert with_len.facade.scale.sigma_rel == pytest.approx(
        math.hypot(plain.facade.scale.sigma_rel, 20.0 / 10000.0), rel=1e-12)
    assert with_len.facade.scale.span_sigma_mm == 20.0


def test_bottom_left_origin_yields_full_coverage(window, scene_file):
    _sc, _path, base = scene_file
    _base(window, base, origin_is_facade_corner=True)
    model = window.session.compute_elements(lambda p: 1.0)
    assert model.facade.origin == "bottom_left" and model.coverage == "full"
    assert "угол фасада" in window.side.base.result.text()
    _base(window, base, origin_is_facade_corner=False)
    model = window.session.compute_elements(lambda p: 1.0)
    assert model.facade.origin == "operator_reference" and model.coverage == "partial"


def test_assumed_floor_height_does_not_write_a_span_uncertainty(window, scene_file):
    _sc, _path, base = scene_file
    _base(window, base, span_sigma_mm=5.0, scale_source="assumed_floor_height")
    model = window.session.compute_elements(lambda p: 1.0)
    assert model.facade.scale.source == "assumed_floor_height"
    assert model.facade.scale.span_sigma_mm is None


def test_changing_the_base_is_answered_in_milliseconds(window, scene_file, monkeypatch):
    from facade_digitizer.pipeline import run

    _sc, _path, base = scene_file
    calls = []
    real = run.estimate_plane
    monkeypatch.setattr(run, "estimate_plane", lambda *a, **k: calls.append(1) or real(*a, **k))
    _base(window, base)
    t0 = time.perf_counter()
    for span in (18000.0, 15000.0, 20000.0):
        window.side.base.set_values(span_mm=span)
        window.apply_base()
    per_change = (time.perf_counter() - t0) / 3
    assert calls == []
    assert per_change < 0.25, f"{per_change * 1000:.0f} мс на смену длины"


def test_cli_passes_nothing_new_by_default(tmp_path, scene_file):
    """CLI не получил новых обязательных аргументов: выход без них — прежний охват."""
    from facade_digitizer.pipeline import run

    _sc, path, base = scene_file
    (x0, y0), (x1, y1) = base
    code = run.main([str(path), "--raster-mm-per-px", "10", "--origin-px", str(x0), str(y0),
                     "--span-px", str(x0), str(y0), str(x1), str(y1),
                     "--span-mm", str(FACADE_W), "--out-dir", str(tmp_path)])
    assert code == 0
    out = json.loads((tmp_path / "facade.json").read_text(encoding="utf-8"))
    assert out["coverage"] == "partial" and out["facade"]["scale"]["span_sigma_mm"] is None
