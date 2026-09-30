"""Проверки ядра, пришедшие из тестов окна Qt плана 3 (задачи 11 и 12).

Окно Qt заменено веб-интерфейсом; эти проверки окна не касались — они о шлюзе
качества и о ветви отказа фазы масштаба, — поэтому сохранены отдельно.
"""
import json
import math

import numpy as np
import pytest

from facade_digitizer.pipeline import quality

FACADE_W = 20000.0


@pytest.fixture(scope="module")
def scene_file(tmp_path_factory):
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    path = tmp_path_factory.mktemp("база") / "facade.png"
    save_image(path, sc.render())
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    return sc, path, base


def test_preliminary_never_rejects_and_uses_the_same_strings():
    t = quality.DEFAULT
    pre = quality.assess_preliminary(t.sharpness_min / 10, usable=0.1)
    assert pre.verdict == "degraded"
    assert pre.reasons == [
        quality.SHARPNESS_REASON.format(value=t.sharpness_min / 10,
                                        threshold=t.sharpness_min),
        quality.USABLE_FRACTION_REASON.format(value=0.1)]
    assert quality.assess_preliminary(t.sharpness_min * 10, usable=0.9).verdict == "ok"


def test_preliminary_refuses_undefined_sharpness():
    with pytest.raises(ValueError, match="резкость"):
        quality.assess_preliminary(float("nan"), usable=0.5)


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
