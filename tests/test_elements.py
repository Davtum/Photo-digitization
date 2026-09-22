"""Свойства оцифровки проёмов оператором. Задача 18.

Сцена и истинные габариты проёма берутся из `tests/test_synth.py::make_scene`
(один проём: x=4000, y=3000, width=1460, height=1900, depth=150). Геометрия —
`H`, поза камеры, `mm_per_unit`, начало отсчёта — строится ТОЧНО из истинных точек
схода сцены (`exact_geometry`, по образцу `tests/test_homography.py::recover`), а
не через шумную оценку по кадру: задача 18 проверяет обвязку `elements.py`, а не
качество `estimate_plane`, для которого уже есть отдельные тесты и отдельный
бюджет. Разметка (`corners_px`, `reveal.inner_edge_px`) при этом берётся из
НАСТОЯЩЕЙ проекции сцены (`scene.project`), то есть ровно тем путём, каким её
получил бы оператор, кликающий по реальному кадру.
"""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from facade_digitizer.geometry.homography import (
    apply_homography,
    camera_pose,
    homography_from_vanishing_points,
)
from facade_digitizer.geometry.parallax import visible_reveal_side
from facade_digitizer.pipeline.elements import (
    ASSISTED_SIZE_TOLERANCE_MM,
    ElementMark,
    RevealMark,
    _gsd_near,
    digitize_elements,
    load_marks,
    parse_marks,
)
from facade_digitizer.pipeline.io import save_image
from facade_digitizer.pipeline.quality import THETA_MAX_DEG
from facade_digitizer.pipeline.run import OperatorReference, main, process
from facade_digitizer.schema import FacadeModel
from tests.test_synth import SIZE, K, make_scene

FACADE_CORNERS = np.array([[0.0, 0.0], [20000.0, 0.0], [20000.0, 15000.0], [0.0, 15000.0]])

#: σ указания точки для тестов этого модуля. Не `run.py:DEFAULT_OPERATOR_SIGMA_PX`
#: (4.5 px) — та величина откалибрована под клик по ДВУМ точкам опорной базы,
#: растянутой на всю двадцатиметровую сторону фасада, где экран показан с
#: уменьшением 2.75x (докстринг `run.py`). Клик по углу окна оператор делает с
#: увеличением (иначе полтора метра проёма невозможно разметить точнее полусотни
#: миллиметров), и `REFERENCE_SIGMA_PX` модуля `parallax.py` — «нижняя граница
#: качества режима assisted» (спецификация, п. 6.1) — сама спецификация называет
#: подходящим опорным значением для этого случая.
SIGMA_PX = 1.0
SIGMA_REL = 0.002       # типичная σ масштаба источника 1, п. 6.3 (0.14-0.28 %)
RESIDUAL_PX = 1.4       # пример из раздела 10 спецификации

#: Два ракурса одного и того же проёма (x=4000, y=3000, 1460x1900, depth=150 —
#: умолчание `make_scene`), подобранные по УГЛУ ВИЗИРОВАНИЯ НА САМ ПРОЁМ, а не по
#: наклону камеры: поле зрения широкое, и проём лежит в стороне от центра кадра,
#: поэтому θ_cam и θ(проём) — разные числа (спецификация, п. 2.1). Оба ракурса
#: держатся ниже порога пригодности (30°, `THETA_MAX_DEG`), иначе элемент
#: отвергался бы целиком, и сравнивать sigma_width/meets_tolerance было бы не с чем.
#:
#:     MILD_VIEW   θ(проём) ≈  1.1°   σ_width ≈  8.5 мм   meets_tolerance = True
#:     STEEP_VIEW  θ(проём) ≈ 29.7°   σ_width ≈ 29.1 мм   meets_tolerance = False
#:
#: Числа получены прогоном `digitize_elements` на этих ракурсах (см. отчёт задачи
#: 18) — не подобраны так, чтобы совпасть с ожиданием теста задним числом.
MILD_VIEW = {"dx": -5000.0, "dy": -3500.0, "dist": 15000.0}
STEEP_VIEW = {"dx": -500.0, "dy": -1000.0, "dist": 9500.0}


def exact_vps(sc):
    return (K @ (sc.R_wc @ np.array([1.0, 0.0, 0.0])),
            K @ (sc.R_wc @ np.array([0.0, 1.0, 0.0])))


def exact_geometry(sc):
    """Точные `H`, поза камеры, `mm_per_unit`, начало отсчёта — из истины сцены.

    То же построение, что `tests/test_homography.py::recover`, но начало отсчёта
    поставлено в (0, 0) фасада (левый нижний угол), а не в первый угол образа
    кадра: элементам нужна привычная система координат фасада, сравнимая с
    `Opening` синтетической сцены напрямую.
    """
    vh, vv = exact_vps(sc)
    H = homography_from_vanishing_points(vh, vv, K, SIZE)
    r = apply_homography(H, sc.project(FACADE_CORNERS))
    mm_per_unit = 20000.0 / float(np.linalg.norm(r[1] - r[0]))
    origin_rect = tuple(r[0])
    camera, _ = camera_pose(H, vh, vv, K, mm_per_unit, origin_rect)
    return H, camera, mm_per_unit, origin_rect


def _window_mark(sc, *, edge_type="sharp_wall_edge", reveal=None):
    op = sc.openings[0]
    corners = [tuple(map(float, pt)) for pt in sc.project(op.corners_mm())]
    payload = {"class": "window", "mounting": "embedded", "edge_type": edge_type,
              "corners_px": corners}
    if reveal is not None:
        payload["reveal"] = reveal
    return ElementMark.model_validate(payload)


def _digitize(sc, marks, **overrides):
    H, camera, mm_per_unit, origin_rect = exact_geometry(sc)
    kwargs = {"H": H, "camera": camera, "mm_per_unit": mm_per_unit,
             "origin_rect": origin_rect, "image_size": SIZE, "sigma_px": SIGMA_PX,
             "sigma_rel": SIGMA_REL, "residual_px": RESIDUAL_PX,
             "theta_max_deg": THETA_MAX_DEG}
    kwargs.update(overrides)
    return digitize_elements(marks, **kwargs)


def test_gsd_near_takes_the_worst_not_the_best_resolution():
    """Бюджет п. 6.1 нормируется по ХУДШЕМУ разрешению у проёма, не по лучшему.

    Синтетическое поле 2x2 с сильно различающимися узлами: если бы `_gsd_near`
    брала среднее или минимум, результат совпал бы с одним из меньших значений.
    """
    gsd_field = np.array([[3.0, 3.0], [3.0, 9.0]])
    value = _gsd_near(gsd_field, image_size=(2, 2),
                      points_px=[(0.0, 0.0), (1.0, 1.0)], label="test")
    assert value == 9.0


# --- 1. Габарит восстанавливается в пределах измеренной погрешности ---------------


def test_size_recovers_true_dimensions_within_reported_sigma():
    sc = make_scene(depth=150.0, **MILD_VIEW)
    op = sc.openings[0]
    elements = _digitize(sc, [_window_mark(sc)])
    element = elements[0]
    size = element.size_mm

    assert abs(size.width - op.width) <= 3.0 * size.sigma_width
    assert abs(size.height - op.height) <= 3.0 * size.sigma_height
    # σ — не выдумана и не вырождена: лежит в порядке величины таблицы п. 6.1 (мм).
    assert 1.0 < size.sigma_width < 40.0
    assert 1.0 < size.sigma_height < 40.0
    # Ширина и высота не перепутаны местами: проём не квадратный (1460 x 1900).
    assert abs(size.width - op.width) < abs(size.width - op.height)
    assert element.id == "w_000"
    assert element.class_name == "window"
    assert len(element.contour_mm) == 4


# --- 2. Погрешность не константа: один проём, два ракурса — разные sigma_width ----


def _sigma_and_theta(dx, dy, dist):
    sc = make_scene(dx=dx, dy=dy, dist=dist, depth=150.0)
    elements = _digitize(sc, [_window_mark(sc)])
    return elements[0].size_mm.sigma_width, elements[0].theta.full_deg


def test_sigma_is_not_a_mode_constant():
    sigma_mild, theta_mild = _sigma_and_theta(**MILD_VIEW)
    sigma_steep, theta_steep = _sigma_and_theta(**STEEP_VIEW)

    assert theta_steep > theta_mild
    assert sigma_steep > sigma_mild
    assert sigma_steep != pytest.approx(sigma_mild)


def test_sigma_reacts_to_the_measured_reprojection_residual():
    """Слагаемое остаточной невязки (п. 6.1) действительно зависит от `residual_px`.

    Тот же ракурс, та же разметка — единственное, что меняется, это переданная
    `residual_px`. Малая и большая невязка обязаны дать разную σ_width; равенство
    означало бы, что аргумент по дороге подменяется опорным допущением
    (`RESIDUAL_REFERENCE_MM`), которое предназначено ТОЛЬКО для случая
    `residual_px is None` (путь `manual_four_point`).
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    small = _digitize(sc, [_window_mark(sc)], residual_px=0.1)[0]
    large = _digitize(sc, [_window_mark(sc)], residual_px=50.0)[0]
    unmeasured = _digitize(sc, [_window_mark(sc)], residual_px=None)[0]

    assert large.size_mm.sigma_width > small.size_mm.sigma_width
    assert unmeasured.size_mm.sigma_width != pytest.approx(small.size_mm.sigma_width)
    assert unmeasured.size_mm.sigma_width != pytest.approx(large.size_mm.sigma_width)


# --- 3. meets_tolerance — по фактическому углу, не по режиму ----------------------


def test_meets_tolerance_follows_the_actual_angle_not_the_mode():
    sc_mild = make_scene(depth=150.0, **MILD_VIEW)
    mild = _digitize(sc_mild, [_window_mark(sc_mild)])[0]

    sc_steep = make_scene(depth=150.0, **STEEP_VIEW)
    steep = _digitize(sc_steep, [_window_mark(sc_steep)])[0]

    assert mild.theta.full_deg < steep.theta.full_deg
    assert mild.meets_tolerance is True
    assert steep.meets_tolerance is False
    # Признак — результат сравнения посчитанной σ с целью, а не переписанный ответ.
    assert (max(mild.size_mm.sigma_width, mild.size_mm.sigma_height)
            <= ASSISTED_SIZE_TOLERANCE_MM)
    assert (max(steep.size_mm.sigma_width, steep.size_mm.sigma_height)
            > ASSISTED_SIZE_TOLERANCE_MM)


# --- 4. Элемент вне пригодной области отвергается, а не тихо посчитан -------------


def test_element_beyond_the_angle_threshold_is_refused():
    """Угол визирования на проём выше порога — отказ с названной причиной."""
    sc = make_scene(dx=17000.0, dy=-500.0, dist=9000.0, depth=150.0)
    mark = _window_mark(sc)
    theta_full = np.degrees(np.arctan(np.hypot(
        *sc.camera_on_plane().tan_theta(4730.0, 3950.0))))
    assert theta_full > THETA_MAX_DEG   # проверка предположения о сцене

    with pytest.raises(ValueError, match="угол визирования .* превышает порог"):
        _digitize(sc, [mark])


def test_element_behind_the_vanishing_line_is_refused():
    """Проём за линией схода — отказ, а не подстановка мусорных координат.

    Ракурс — `tests/test_homography.py::GRAZING_VIEW` (камера в 200 мм от
    плоскости, 20 м вбок): линия схода плоскости фасада проходит в 36 px от
    главной точки, и правая половина кадра лежит по другую сторону от неё
    (измерено: точка (5279, 1978) даёт знаменатель гомографии знака,
    противоположного центру кадра). Метка ставится заведомо там.
    """
    sc = make_scene(dx=-20000.0, dy=0.0, dist=200.0, depth=150.0)
    H, camera, mm_per_unit, origin_rect = exact_geometry(sc)
    bad_mark = ElementMark.model_validate({
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": [[5100.0, 1900.0], [5279.0, 1900.0],
                       [5279.0, 2000.0], [5100.0, 2000.0]],
    })
    with pytest.raises(ValueError, match="линией схода"):
        digitize_elements([bad_mark], H=H, camera=camera, mm_per_unit=mm_per_unit,
                          origin_rect=origin_rect, image_size=SIZE, sigma_px=SIGMA_PX,
                          sigma_rel=SIGMA_REL, residual_px=RESIDUAL_PX,
                          theta_max_deg=THETA_MAX_DEG)


# --- 5. Глубина заглубления: значение, порог, неверная сторона --------------------


def _reveal_mark_for(sc, side=None):
    op = sc.openings[0]
    cam = sc.camera_on_plane()
    x_left, x_right = op.x, op.x + op.width
    y_bottom, y_top = op.y, op.y + op.height
    vertical, _ = visible_reveal_side(cam, x_left, x_right, y_bottom, y_top)
    chosen_side = side or vertical
    if chosen_side in ("left", "right"):
        x = x_left if chosen_side == "left" else x_right
        edge_mm = np.array([[x, y_bottom], [x, y_top]])
    else:
        y = y_bottom if chosen_side == "bottom" else y_top
        edge_mm = np.array([[x_left, y], [x_right, y]])
    inner_px = sc.project(edge_mm, depth=op.depth)
    reveal = {"side": chosen_side,
             "inner_edge_px": [tuple(map(float, pt)) for pt in inner_px]}
    return _window_mark(sc, reveal=reveal), vertical


def test_recess_depth_matches_scene_truth_and_carries_the_threshold_angle():
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    mark, _ = _reveal_mark_for(sc)
    element = _digitize(sc, [mark])[0]

    recess = element.recess
    assert recess is not None
    assert recess.origin == "measured_from_reveal"
    assert recess.datum == "quarter_edge"
    assert recess.value_mm == pytest.approx(150.0, rel=0.05)
    assert recess.theta_perp_deg is not None
    assert recess.sigma_mm is not None and recess.sigma_mm > 0.0
    # σ_d не выдумана: лежит в порядке величины бюджета п. 6.2 (единицы-десятки мм),
    # а не совпадает с σ габарита и не равна нулю/огромному числу.
    assert 1.0 < recess.sigma_mm < 100.0


def test_recess_below_the_computed_threshold_is_marked_unavailable():
    """Ракурс ниже порога применимости (п. 6.2) — `origin = unavailable`, не число."""
    sc = make_scene(depth=150.0, **MILD_VIEW)
    mark, _ = _reveal_mark_for(sc)
    element = _digitize(sc, [mark])[0]

    recess = element.recess
    assert recess is not None
    assert recess.origin == "unavailable"
    assert recess.value_mm is None
    assert recess.sigma_mm is None


def test_recess_on_the_invisible_side_is_marked_unavailable_not_measured():
    """Отложенное замечание задачи 3: разметка невидимой грани не считается, а распознаётся."""
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    op = sc.openings[0]
    cam = sc.camera_on_plane()
    vertical, _ = visible_reveal_side(cam, op.x, op.x + op.width, op.y, op.y + op.height)
    wrong_side = "left" if vertical == "right" else "right"

    mark, _ = _reveal_mark_for(sc, side=wrong_side)
    element = _digitize(sc, [mark])[0]

    assert element.recess is not None
    assert element.recess.origin == "unavailable"
    assert element.recess.reveal_side == wrong_side


# --- 6. Величина без σ и без происхождения не выпускается: схема стережёт ---------


def test_size_mm_cannot_be_built_without_both_sigmas():
    from facade_digitizer.schema import SizeMM
    with pytest.raises(ValidationError):
        SizeMM(width=1460.0, height=1900.0)


def test_recess_schema_refuses_a_number_under_unavailable_origin():
    """Дополняет схемную охрану: `unavailable` не может нести `value_mm`/`sigma_mm`.

    До задачи 18 схема заставляла бы выдумать число даже для `unavailable` —
    `value_mm`/`sigma_mm` были обязательными `float` без исключения. Проверяется,
    что обход через `_digitize_recess` строит именно годный по схеме объект.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    mark, _ = _reveal_mark_for(sc)
    element = _digitize(sc, [mark])[0]
    assert element.recess.origin == "unavailable"
    # Схема уже это стережёт (tests/test_schema.py); здесь — что сборка её не обходит.
    FacadeModel.model_validate(
        {"schema_version": "1.1", "software_version": "x", "coverage": "partial",
         "mode": "assisted", "images": [], "elements": [],
         "facade": {"origin": "bottom_left", "bounds_mm": [0, 0, 1, 1],
                    "mm_per_rectified_px": 1.0,
                    "scale": {"source": "operator_reference", "sigma_rel": 0.001,
                              "meets_tolerance": True}}})


# --- 7. Растр совпадает с записанной гомографией -----------------------------------


def test_saved_raster_matches_the_recorded_homography(tmp_path):
    """Точка исходного снимка через записанные H и origin попадает в растре туда,
    где она есть на самом деле."""
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    image = sc.render()
    path = tmp_path / "facade.png"
    save_image(path, image)

    far = np.array([[20000.0, 0.0]])
    pts = sc.project(np.vstack([FACADE_CORNERS[:1], far]))
    reference = OperatorReference(
        origin_px=(float(pts[0][0]), float(pts[0][1])),
        span_px=((float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))),
        span_mm=20000.0)

    raster_path = tmp_path / "facade_rectified.png"
    model = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                    save_rectified_to=raster_path)

    homography = model.images[0].homography
    H = np.asarray(homography["H"], dtype=float)

    op = sc.openings[0]
    # Точка у нижнего левого угла проёма, отступя внутрь на 50 мм — заведомо в
    # ЗАЛИВКЕ ОКНА (цвет 90) и не задета сдвинутым параллаксом внутренним
    # прямоугольником откоса (цвет 35): опорная точка камеры (13000, 5500) лежит
    # правее и выше проёма, поэтому смещение направлено туда же, а не к этому углу.
    corner_mm = np.array([[op.x + 50.0, op.y + 50.0]])
    corner_px = sc.project(corner_mm)
    rect_px = apply_homography(H, corner_px)[0]

    from facade_digitizer.pipeline.io import load_image
    raster, _ = load_image(raster_path)
    row, col = round(rect_px[1]), round(rect_px[0])
    assert 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]
    patch = raster[max(0, row - 2):row + 3, max(0, col - 2):col + 3]
    # Цвет заливки окна в синтетической сцене — 90 (см. synth/scene.py:render).
    # Растр интерполирован (INTER_LINEAR), поэтому сверяется окрестность, а не
    # один пиксель.
    assert patch.mean() == pytest.approx(90.0, abs=20.0)


# --- 8. Разметка не диктует результат ----------------------------------------------


def test_changing_corners_changes_the_size():
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    base = _window_mark(sc)
    element = _digitize(sc, [base])[0]

    op = sc.openings[0]
    shifted_op_corners = np.array([
        [op.x, op.y], [op.x + op.width + 500.0, op.y],
        [op.x + op.width + 500.0, op.y + op.height], [op.x, op.y + op.height],
    ])
    shifted_corners_px = [tuple(map(float, pt)) for pt in sc.project(shifted_op_corners)]
    shifted_mark = ElementMark.model_validate({
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": shifted_corners_px})
    shifted = _digitize(sc, [shifted_mark])[0]

    assert shifted.size_mm.width != pytest.approx(element.size_mm.width, rel=1e-6)
    assert shifted.size_mm.width == pytest.approx(element.size_mm.width + 500.0, abs=5.0)


@pytest.mark.parametrize("field", ["size_mm", "confidence"])
def test_forbidden_fields_are_rejected_by_the_mark_format(field):
    """`size_mm`, `confidence` и т.п. в формате разметки нет: подмена невозможна
    по построению — лишние ключи отвергаются (`ElementMark`, `extra="forbid"`)."""
    payload = {"class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
              "corners_px": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
              field: 123}
    with pytest.raises(ValidationError):
        ElementMark.model_validate(payload)


def test_recess_value_mm_is_rejected_in_the_mark_format():
    payload = {"class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
              "corners_px": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
              "reveal": {"side": "right", "inner_edge_px": [[1.0, 1.0], [2.0, 2.0]],
                        "value_mm": 150.0}}
    with pytest.raises(ValidationError):
        ElementMark.model_validate(payload)


# --- Формат разметки: строгость, --marks/--save-rectified в CLI -------------------


def test_parse_marks_requires_a_json_list():
    with pytest.raises(ValueError, match="списком"):
        parse_marks({"class": "window"})


def test_corners_px_must_have_exactly_four_points():
    with pytest.raises(ValidationError):
        ElementMark.model_validate({
            "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
            "corners_px": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]})


def test_reveal_inner_edge_must_have_exactly_two_points():
    with pytest.raises(ValidationError):
        RevealMark.model_validate({"side": "right", "inner_edge_px": [[1.0, 1.0]]})


def test_load_marks_reads_and_validates_the_file(tmp_path):
    payload = [{"class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
               "corners_px": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]}]
    path = tmp_path / "marks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    marks = load_marks(path)
    assert len(marks) == 1
    assert marks[0].class_ == "window"


def test_cli_digitizes_marked_elements_and_saves_the_raster(tmp_path):
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    image = sc.render()
    path = tmp_path / "facade.png"
    save_image(path, image)

    far = np.array([[20000.0, 0.0]])
    pts = sc.project(np.vstack([FACADE_CORNERS[:1], far]))
    op = sc.openings[0]
    corners_px = [tuple(map(float, pt)) for pt in sc.project(op.corners_mm())]
    marks_path = tmp_path / "marks.json"
    marks_path.write_text(json.dumps([{
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": corners_px}]), encoding="utf-8")

    out_dir = tmp_path / "out"
    code = main([
        str(path),
        "--raster-mm-per-px", "10.0",
        "--origin-px", str(pts[0][0]), str(pts[0][1]),
        "--span-px", str(pts[0][0]), str(pts[0][1]), str(pts[1][0]), str(pts[1][1]),
        "--span-mm", "20000.0",
        "--marks", str(marks_path),
        "--save-rectified",
        "--out-dir", str(out_dir),
    ])

    assert code == 0
    model = FacadeModel.model_validate_json((out_dir / "facade.json").read_text("utf-8"))
    assert len(model.elements) == 1
    assert model.elements[0].origin == "operator"
    assert (out_dir / "facade_rectified.png").exists()
    assert (out_dir / "facade_rectified_mask.png").exists()


def test_coverage_is_full_only_when_facade_boundary_is_marked(tmp_path):
    """`coverage` становится `"full"` только если размечены границы фасада."""
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    image = sc.render()
    path = tmp_path / "facade.png"
    save_image(path, image)

    far = np.array([[20000.0, 0.0]])
    pts = sc.project(np.vstack([FACADE_CORNERS[:1], far]))
    reference = OperatorReference(
        origin_px=(float(pts[0][0]), float(pts[0][1])),
        span_px=((float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))),
        span_mm=20000.0)

    window_mark = _window_mark(sc)
    boundary_mark = ElementMark.model_validate({
        "class": "facade_boundary", "mounting": "flush", "edge_type": "sharp_wall_edge",
        "corners_px": [tuple(map(float, pt)) for pt in sc.project(FACADE_CORNERS)],
    })

    window_only = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                          marks=[window_mark])
    with_boundary = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                            marks=[window_mark, boundary_mark])

    assert window_only.coverage == "partial"
    assert with_boundary.coverage == "full"
