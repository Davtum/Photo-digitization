"""Ручная плоскость в ядре. План 3, задача 5; спецификация, п. 4.2.

«Оператор указывает четыре точки И одно из: отношение сторон, два размера в
миллиметрах, либо подтверждение, что камера откалибрована и направления
ортогональны. Реализуются все три.» До этой задачи `estimate_plane_manual` не
вызывалась ниоткуда, не возвращала точек схода, и `camera_pose` без них отказывала:
путь `needs_operator` заканчивался тупиком.

Истина — синтетическая сцена с известной позой. Внутренние параметры подаются
профилем с ИСТИННОЙ матрицей сцены: иначе K бралась бы из типового поля зрения с
ошибкой около 1.8 %, и тест проверял бы её, а не ручную плоскость.
"""
import numpy as np
import pytest

from facade_digitizer.geometry.homography import camera_pose, pose_from_homography
from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile
from facade_digitizer.pipeline.elements import ElementMark
from facade_digitizer.pipeline.io import save_image
from facade_digitizer.pipeline.plane import ManualPlane, estimate_plane
from tests.test_synth import SIZE, K, make_scene

VIEW = {"dx": 3000.0, "dy": -2000.0, "dist": 20000.0, "depth": 150.0}
FACADE_W, FACADE_H = 20000.0, 15000.0
RASTER_MM_PER_PX = 10.0


def _operator_order(pts_mm_bl_br_tr_tl) -> np.ndarray:
    """Углы прямоугольника в порядке оператора: верхний левый, верхний правый,
    нижний правый, нижний левый (порядок `homography_from_four_points`)."""
    bl, br, tr, tl = np.asarray(pts_mm_bl_br_tr_tl, dtype=float)
    return np.array([tl, tr, br, bl])


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("ручная")
    sc = make_scene(**VIEW)
    path = d / "facade.png"
    save_image(path, sc.render())
    profile = d / "profile.json"
    save_profile(CalibrationProfile(model="unknown", K=K.tolist(), dist=[0.0] * 5,
                                    rms_px=0.1, image_size=SIZE), profile)
    facade_mm = np.array([[0, 0], [FACADE_W, 0], [FACADE_W, FACADE_H], [0, FACADE_H]])
    quad_px = sc.project(_operator_order(facade_mm))
    base = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0]]))
    reference = run.OperatorReference(origin_px=tuple(base[0]),
                                      span_px=(tuple(base[0]), tuple(base[1])),
                                      span_mm=FACADE_W)
    corners = [[float(u), float(v)] for u, v in sc.project(sc.openings[0].corners_mm())]
    mark = ElementMark.model_validate({"class": "window", "mounting": "embedded",
                                       "edge_type": "sharp_wall_edge",
                                       "corners_px": corners})
    return sc, path, profile, quad_px, reference, mark


CONSTRAINTS = {
    "calibrated": {"assume_calibrated": True},
    "aspect": {"aspect_ratio": FACADE_W / FACADE_H},
    "sizes": {"size_mm": (FACADE_W, FACADE_H)},
}


@pytest.mark.parametrize("name", list(CONSTRAINTS))
def test_manual_plane_recovers_the_pose_of_the_scene(setup, name):
    sc, path, profile, quad_px, reference, _mark = setup
    override = ManualPlane(image_pts=quad_px, **CONSTRAINTS[name])
    fs = run.frame_stage(path, profile_path=profile, plane_override=override)
    assert not fs.plane.needs_operator
    ss = run.scale_stage(fs, reference)
    truth = sc.camera_on_plane()
    got = ss.camera
    assert got.cx == pytest.approx(truth.cx, rel=5e-3)
    assert got.cy == pytest.approx(truth.cy, rel=5e-3)
    assert got.cz == pytest.approx(truth.cz, rel=5e-3)


@pytest.mark.parametrize("name", list(CONSTRAINTS))
def test_manual_plane_goes_end_to_end_to_millimetres(setup, name):
    sc, path, profile, quad_px, reference, mark = setup
    model = run.process(path, operator_reference=reference,
                        raster_mm_per_px=RASTER_MM_PER_PX, profile_path=profile,
                        plane_override=ManualPlane(image_pts=quad_px, **CONSTRAINTS[name]),
                        marks=[mark])
    assert model.images[0].rectification.method == "manual_four_point"
    size = model.elements[0].size_mm
    op = sc.openings[0]
    assert abs(size.width - op.width) <= size.sigma_width
    assert abs(size.height - op.height) <= size.sigma_height


def test_manual_plane_without_any_constraint_is_refused_by_name(setup):
    _sc, path, profile, quad_px, _reference, _mark = setup
    with pytest.raises(ValueError, match="четырёх точек недостаточно"):
        run.frame_stage(path, profile_path=profile,
                        plane_override=ManualPlane(image_pts=quad_px))


def test_manual_plane_points_outside_the_frame_are_refused(setup):
    _sc, path, profile, quad_px, _reference, _mark = setup
    bad = np.array(quad_px, dtype=float)
    bad[0] = (-50.0, 10.0)
    with pytest.raises(ValueError, match="вне кадра"):
        run.frame_stage(path, profile_path=profile,
                        plane_override=ManualPlane(image_pts=bad, aspect_ratio=1.0))


def test_operator_point_order_resolves_the_half_turn(setup):
    """Порядок углов, указанный оператором, называет верх фасада — поворот на
    полкруга, неразрешимый двумя точками схода, здесь разрешён, и файл это говорит."""
    _sc, path, profile, quad_px, reference, _mark = setup
    fs = run.frame_stage(path, profile_path=profile,
                         plane_override=ManualPlane(image_pts=quad_px,
                                                    size_mm=(FACADE_W, FACADE_H)))
    ss = run.scale_stage(fs, reference)
    assert ss.half_turn.resolved is True


def test_pose_from_homography_matches_camera_pose_on_the_automatic_path(setup):
    """Разложение гомографии — обобщение `camera_pose`, а не второй её вариант:
    на гомографии из точек схода оно обязано дать ту же позу."""
    _sc, path, profile, _quad, reference, _mark = setup
    fs = run.frame_stage(path, profile_path=profile)
    ss = run.scale_stage(fs, reference)
    plane = fs.plane
    expected = camera_pose(plane.H, plane.vh, plane.vv, fs.frame.K, ss.mm_per_unit,
                           ss.origin_rect).camera
    got = pose_from_homography(plane.H, fs.frame.K, ss.mm_per_unit, ss.origin_rect).camera
    assert got.cx == pytest.approx(expected.cx, abs=1e-6)
    assert got.cy == pytest.approx(expected.cy, abs=1e-6)
    assert got.cz == pytest.approx(expected.cz, abs=1e-6)


def test_manual_plane_rescues_a_frame_the_automatic_path_refuses(setup, tmp_path):
    """Кадр, на котором точки схода не оцениваются, становится измеримым.

    Сцена та же, но от неё оставлены только короткие кресты в углах фасада:
    отрезков длиннее порога детектора (40 px) в кадре нет, и автоматика отказывает
    (`needs_operator`). Углы при этом видны, и оператор их указывает.

    Сперва здесь был оставлен контур фасада целиком — и автоматика НЕ отказала:
    детектор дробит длинную сторону на 20–30 отрезков, и по ним поза восстановлена
    с ошибкой 0.1 %. То есть посылка теста была неверна, а не автоматика.
    """
    sc, _path, profile, quad_px, reference, mark = setup
    import cv2
    img = np.full((SIZE[1], SIZE[0]), 120, np.uint8)
    for x, y in np.round(quad_px).astype(int):
        cv2.line(img, (x - 12, y), (x + 12, y), 30, 3)
        cv2.line(img, (x, y - 12), (x, y + 12), 30, 3)
    path = tmp_path / "контур.png"
    save_image(path, img)
    assert run.frame_stage(path, profile_path=profile).plane.needs_operator

    model = run.process(path, operator_reference=reference,
                        raster_mm_per_px=RASTER_MM_PER_PX, profile_path=profile,
                        plane_override=ManualPlane(image_pts=quad_px,
                                                   size_mm=(FACADE_W, FACADE_H)),
                        marks=[mark])
    size = model.elements[0].size_mm
    assert abs(size.width - sc.openings[0].width) <= size.sigma_width
    assert abs(size.height - sc.openings[0].height) <= size.sigma_height


def test_roi_keeps_only_segments_inside(setup, monkeypatch):
    """ROI: точки схода оцениваются только по отрезкам внутри обведённой области.

    Нужна для снимка угла здания, где второй фасад даёт свои точки схода.
    """
    _sc, path, _profile, _quad, _reference, _mark = setup
    from facade_digitizer.pipeline import plane as plane_module
    from facade_digitizer.pipeline.io import load_image

    image, _meta = load_image(path)
    seen = {}
    real = plane_module.estimate_vanishing_points

    def spy(segments, image_size, K_, *args, **kwargs):
        seen["segments"] = segments
        return real(segments, image_size, K_, *args, **kwargs)

    monkeypatch.setattr(plane_module, "estimate_vanishing_points", spy)
    roi = np.array([[0, 0], [SIZE[0] / 2, 0], [SIZE[0] / 2, SIZE[1]], [0, SIZE[1]]], float)
    estimate_plane(image, K, roi=roi)
    segs = seen["segments"]
    assert len(segs) > 0
    assert np.all(segs[:, [0, 2]] <= SIZE[0] / 2 + 1e-6)


def test_roi_over_an_empty_area_sends_the_frame_to_the_operator(setup):
    _sc, path, _profile, _quad, _reference, _mark = setup
    from facade_digitizer.pipeline.io import load_image

    image, _meta = load_image(path)
    empty = np.array([[5, 5], [60, 5], [60, 60], [5, 60]], float)
    assert estimate_plane(image, K, roi=empty).needs_operator


def test_cli_manual_plane_reaches_the_output(setup, tmp_path):
    import json

    _sc, path, profile, quad_px, reference, _mark = setup
    (x0, y0), (x1, y1) = reference.span_px
    code = run.main([str(path), "--raster-mm-per-px", "10",
                     "--origin-px", str(x0), str(y0),
                     "--span-px", str(x0), str(y0), str(x1), str(y1),
                     "--span-mm", str(FACADE_W), "--profile", str(profile),
                     "--manual-plane", *[str(v) for v in np.asarray(quad_px).ravel()],
                     "--manual-size-mm", str(FACADE_W), str(FACADE_H),
                     "--out-dir", str(tmp_path)])
    assert code == 0
    model = json.loads((tmp_path / "facade.json").read_text(encoding="utf-8"))
    assert model["images"][0]["rectification"]["method"] == "manual_four_point"
    assert model["images"][0]["pose_to_facade"]["half_turn"]["resolved"] is True


@pytest.mark.parametrize("extra, message", [
    (["--manual-aspect", "1.3"], "без --manual-plane"),
    (["--manual-plane", *["1"] * 8, "--manual-aspect", "1.3", "--manual-calibrated"],
     "РОВНО одно"),
    (["--manual-plane", *["1"] * 8], "РОВНО одно"),
    (["--roi", "0", "0", "10", "10"], "не менее трёх"),
])
def test_cli_rejects_inconsistent_manual_plane_arguments(extra, message, capsys):
    with pytest.raises(SystemExit):
        run.main(["x.png", "--raster-mm-per-px", "10", "--origin-px", "0", "0",
                  "--span-px", "0", "0", "1", "1", "--span-mm", "1", *extra])
    assert message in capsys.readouterr().err
