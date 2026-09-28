"""Состояние сессии оператора без Qt. План 3, задача 7.

Логика «чего не хватает, чтобы мерить» проверяется без окна: GUI-тесты медленны и
плохо объясняют, что сломалось. Модуль `ui.session` Qt не импортирует — это
проверяется отдельно.
"""
import subprocess
import sys

import numpy as np
import pytest

from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.io import save_image
from facade_digitizer.ui.session import ClickedPoint, OperatorSession, order_corners
from tests.test_synth import make_scene

VIEW = {"dx": 3000.0, "dy": -2000.0, "dist": 20000.0, "depth": 150.0}
FACADE_W = 20000.0


def _sigma_one(point: ClickedPoint) -> float:
    return 1.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("сессия")
    sc = make_scene(**VIEW)
    path = d / "facade.png"
    save_image(path, sc.render())
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    corners = sc.project(sc.openings[0].corners_mm())      # Н-Л, Н-П, В-П, В-Л
    return sc, path, base, corners


@pytest.fixture(scope="module")
def processed(scene):
    """Сессия с обработанным кадром — фаза кадра дорогая, считается раз на модуль."""
    _sc, path, _base, _corners = scene
    session = OperatorSession()
    session.open_image(path)
    session.compute_frame()
    return session


def test_session_module_does_not_import_qt():
    code = ("import sys, facade_digitizer.ui.session; "
            "print(any(m.startswith('PySide6') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_session_without_image_names_it():
    ready, reasons = OperatorSession().ready_to_measure()
    assert not ready and reasons == ["снимок не открыт"]


def test_session_names_each_missing_piece_by_its_own_reason(scene, processed):
    _sc, _path, base, corners = scene
    s = processed
    s.clear_reference()
    s.marks.clear()
    ready, reasons = s.ready_to_measure()
    assert not ready and "опорная база: указано концов 0 из 2" in reasons

    s.add_reference_end(ClickedPoint(*base[0], view_scale=0.4))
    s.add_reference_end(ClickedPoint(*base[1], view_scale=0.4))
    ready, reasons = s.ready_to_measure()
    assert "длина опорной базы не введена" in reasons

    s.set_span_mm(0.0)
    assert "длина опорной базы должна быть положительной" in s.ready_to_measure()[1]
    s.set_span_mm(FACADE_W)

    mark = s.new_mark()
    for x, y in corners[:3]:
        s.add_corner(mark, ClickedPoint(x, y, view_scale=2.0))
    ready, reasons = s.ready_to_measure()
    assert not ready and f"{mark.id}: углов 3 из 4" in reasons

    s.add_corner(mark, ClickedPoint(*corners[3], view_scale=2.0))
    assert s.ready_to_measure() == (True, [])


def test_frame_without_plane_names_it(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "пусто.png"
    save_image(path, (128 + rng.normal(0, 3, (600, 800))).clip(0, 255).astype(np.uint8))
    s = OperatorSession()
    s.open_image(path)
    s.compute_frame()
    reasons = s.ready_to_measure()[1]
    assert any("плоскость фасада не восстановлена" in r for r in reasons)


def test_each_click_keeps_its_view_scale(processed, scene):
    _sc, _path, _base, corners = scene
    s = processed
    mark = s.new_mark()
    for (x, y), scale in zip(corners, (1.0, 2.0, 3.0, 4.0)):
        s.add_corner(mark, ClickedPoint(x, y, view_scale=scale))
    assert sorted(p.view_scale for p in mark.corners) == [1.0, 2.0, 3.0, 4.0]
    s.delete_mark(mark.id)


def test_mark_drafts_validate_through_element_mark_in_any_click_order(processed, scene):
    """Порядок кликов не важен: сессия приводит углы к обходу `ElementMark`."""
    _sc, _path, _base, corners = scene
    s = processed
    mark = s.new_mark()
    for i in (2, 0, 3, 1):                              # вразброс
        s.add_corner(mark, ClickedPoint(*corners[i], view_scale=2.0))
    element_mark = s.element_marks(_sigma_one)[-1]
    assert np.allclose(element_mark.corners_px, corners)
    assert element_mark.id == mark.id and element_mark.sigma_px == 1.0
    s.delete_mark(mark.id)


def test_order_corners_handles_a_perspective_quad():
    bl, br, tr, tl = (100, 900), (700, 1000), (720, 300), (90, 150)
    shuffled = [tr, bl, tl, br]
    assert [tuple(p) for p in order_corners(shuffled)] == [bl, br, tr, tl]


def test_mark_ids_are_never_reused(processed):
    s = processed
    a, b = s.new_mark(), s.new_mark()
    s.delete_mark(a.id)
    c = s.new_mark()
    assert len({a.id, b.id, c.id}) == 3
    s.delete_mark(b.id)
    s.delete_mark(c.id)


def test_sigma_is_asked_per_point(processed, scene):
    """σ разметки — худшая по её точкам, и каждая точка отвечает своим масштабом."""
    _sc, _path, _base, corners = scene
    s = processed
    mark = s.new_mark()
    for (x, y), scale in zip(corners, (1.0, 1.0, 1.0, 0.25)):
        s.add_corner(mark, ClickedPoint(x, y, view_scale=scale))
    got = s.element_marks(lambda p: 1.0 / p.view_scale)[-1]
    assert got.sigma_px == 4.0
    s.delete_mark(mark.id)


def test_changing_the_profile_invalidates_clicks_explicitly(scene, tmp_path):
    from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile
    from tests.test_synth import SIZE, K

    _sc, path, base, _corners = scene
    s = OperatorSession()
    s.open_image(path)
    s.add_reference_end(ClickedPoint(*base[0], view_scale=0.4))
    profile = tmp_path / "p.json"
    save_profile(CalibrationProfile("unknown", K.tolist(), [0.0] * 5, 0.1, SIZE), profile)
    with pytest.raises(ValueError, match="сбросит"):
        s.set_profile(profile)
    assert len(s.reference.ends) == 1                  # ничего не потеряно молча
    s.set_profile(profile, discard_clicks=True)
    assert s.reference.ends == [] and s.frame is None
    assert s.profile_hash is not None


def test_session_measures_end_to_end(scene):
    sc, path, base, corners = scene
    s = OperatorSession()
    s.open_image(path)
    s.compute_frame()
    s.add_reference_end(ClickedPoint(*base[0], view_scale=0.4))
    s.add_reference_end(ClickedPoint(*base[1], view_scale=0.4))
    s.set_span_mm(FACADE_W)
    mark = s.new_mark()
    for x, y in corners:
        s.add_corner(mark, ClickedPoint(x, y, view_scale=2.0))
    model = s.compute_elements(_sigma_one)
    assert s.error is None
    size = model.elements[0].size_mm
    assert abs(size.width - sc.openings[0].width) <= size.sigma_width
    assert model.elements[0].id == mark.id


def test_errors_are_kept_as_text_not_raised(processed):
    s = processed
    s.clear_reference()
    s.add_reference_end(ClickedPoint(10.0, 10.0, view_scale=1.0))
    s.add_reference_end(ClickedPoint(10.0, 10.0, view_scale=1.0))   # нулевая база
    s.set_span_mm(1000.0)
    assert s.compute_scale(_sigma_one) is None
    assert s.error and "база" in s.error
    s.clear_reference()


def test_reference_uses_per_end_sigma(processed, scene):
    _sc, _path, base, _corners = scene
    s = processed
    s.clear_reference()
    s.add_reference_end(ClickedPoint(*base[0], view_scale=1.0))
    s.add_reference_end(ClickedPoint(*base[1], view_scale=0.5))
    s.set_span_mm(FACADE_W)
    ref = s.operator_reference(lambda p: 1.0 / p.view_scale)
    assert isinstance(ref, run.OperatorReference)
    assert ref.end_sigma_px == (1.0, 2.0)
    assert ref.origin_px == ref.span_px[0]
    s.clear_reference()
