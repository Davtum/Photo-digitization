"""Поле углов визирования до опорной базы. План 3, задачи 9 и 12.

Пригодную зону θ ≤ 30° (п. 2.4) оператор должен видеть ДО разметки, а опорная база
к тому времени может быть не указана. Углы от масштаба не зависят (проверено при
составлении плана 3: θ_cam и θ_p95 при длинах базы 20 000 и 15 000 мм совпадают до
шестого знака), поэтому поле считается при условном масштабе — и обязано совпасть
с полем окончательной фазы масштаба.
"""
import numpy as np
import pytest

from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.io import save_image
from facade_digitizer.pipeline.plane import ManualPlane
from tests.test_synth import make_scene

FACADE_W, FACADE_H = 20000.0, 15000.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("углы")
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    path = d / "facade.png"
    save_image(path, sc.render())
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    ref = run.OperatorReference(origin_px=tuple(base[0]),
                                span_px=(tuple(base[0]), tuple(base[1])), span_mm=FACADE_W)
    return sc, path, ref


def test_preview_matches_the_scale_stage_field(scene):
    _sc, path, ref = scene
    fs = run.frame_stage(path)
    preview = run.theta_preview(fs)
    ss = run.scale_stage(fs, ref)
    assert np.allclose(preview.theta_deg, ss.fields.theta_deg, equal_nan=True, atol=1e-9)
    assert preview.usable_fraction == pytest.approx(ss.fields.usable, abs=1e-12)
    assert preview.theta_p95 == pytest.approx(ss.fields.theta_field_deg["p95"], abs=1e-9)


def test_preview_works_for_a_manual_plane_without_vanishing_points(scene):
    sc, path, ref = scene
    bl, br, tr, tl = np.array([[0, 0], [FACADE_W, 0], [FACADE_W, FACADE_H], [0, FACADE_H]])
    quad = sc.project(np.array([tl, tr, br, bl]))
    fs = run.frame_stage(path, plane_override=ManualPlane(image_pts=quad,
                                                          size_mm=(FACADE_W, FACADE_H)))
    preview = run.theta_preview(fs)
    ss = run.scale_stage(fs, ref)
    assert np.allclose(preview.theta_deg, ss.fields.theta_deg, equal_nan=True, atol=1e-6)


def test_preview_refuses_a_frame_without_plane(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "пусто.png"
    save_image(path, (128 + rng.normal(0, 3, (600, 800))).clip(0, 255).astype(np.uint8))
    fs = run.frame_stage(path)
    with pytest.raises(ValueError, match="needs_operator"):
        run.theta_preview(fs)
