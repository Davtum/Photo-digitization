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

**σ здесь проверяется ТОЧНЫМ ЧИСЛОМ, а не диапазоном.** Ожидание считается
`_expected_size_sigma` — буквальной записью формулы п. 6.1 по входу теста, а не
обращением к реализации. Диапазонная проверка («σ лежит между 1 и 40» плюс
«крутой ракурс больше пологого») пропускала любую монотонную порчу внутрь
диапазона: снятый множитель √2, снятый множитель размера, `tg θ → θ`, габарит по
одной паре сторон — всё это оставалось незамеченным. Поэтому диапазоны здесь
только дополняют точные равенства, а не заменяют их.
"""
import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from facade_digitizer.geometry.angles import FIELD_SHAPE, local_gsd_field
from facade_digitizer.geometry.homography import (
    apply_homography,
    camera_pose,
    homography_from_vanishing_points,
)
from facade_digitizer.geometry.parallax import (
    REFERENCE_SIGMA_CZ_REL,
    REFERENCE_SIGMA_THETA_DEG,
    reveal_depth_sigma,
    visible_reveal_side,
)
from facade_digitizer.pipeline.elements import (
    ASSISTED_SIZE_TOLERANCE_MM,
    FOCAL_RELATIVE_ERROR,
    RESIDUAL_DISTORTION_MM,
    RESIDUAL_REFERENCE_MM,
    WALL_FLATNESS_BUDGET_MM,
    WALL_FLATNESS_DEVIATION_MM,
    ElementMark,
    RevealMark,
    _gsd_near,
    digitize_elements,
    load_marks,
    parse_marks,
)
from facade_digitizer.pipeline.io import save_image
from facade_digitizer.pipeline.quality import THETA_MAX_DEG
from facade_digitizer.pipeline.run import (
    DEFAULT_MARK_SIGMA_PX,
    DEFAULT_OPERATOR_SIGMA_PX,
    OperatorReference,
    main,
    process,
)
from facade_digitizer.schema import FacadeModel
from tests.test_synth import SIZE, K, make_scene

FACADE_CORNERS = np.array([[0.0, 0.0], [20000.0, 0.0], [20000.0, 15000.0], [0.0, 15000.0]])

#: σ указания точки для тестов этого модуля. Не `run.py:DEFAULT_OPERATOR_SIGMA_PX`
#: (4.5 px) — та величина откалибрована под клик по ДВУМ точкам опорной базы,
#: растянутой на всю двадцатиметровую сторону фасада, где экран показан с
#: уменьшением 2.75x (докстринг `run.py`). Клик по углу окна оператор делает с
#: увеличением, и `run.py:DEFAULT_MARK_SIGMA_PX` — это отдельная величина; здесь
#: она берётся явно, а не по умолчанию, чтобы ожидаемая σ считалась из ВХОДА.
SIGMA_PX = 1.0
SIGMA_REL = 0.002       # типичная σ масштаба источника 1, п. 6.3 (0.14-0.28 %)
RESIDUAL_PX = 1.4       # пример из раздела 10 спецификации

#: Ракурсы одного и того же проёма (x=4000, y=3000, 1460x1900, depth=150 —
#: умолчание `make_scene`), подобранные по УГЛУ ВИЗИРОВАНИЯ НА САМ ПРОЁМ, а не по
#: наклону камеры: поле зрения широкое, и проём лежит в стороне от центра кадра,
#: поэтому θ_cam и θ(проём) — разные числа (спецификация, п. 2.1).
#:
#: Угол элемента берётся по ХУДШЕМУ УГЛУ КОНТУРА (не по центроиду), поэтому оба
#: ракурса держат ниже порога пригодности (30°, `THETA_MAX_DEG`) именно худший
#: угол — иначе элемент отвергался бы целиком.
#:
#:     MILD_VIEW   θ(худший) ≈  8.05°  σ_width ≈  9.1 мм  meets_tolerance = True
#:     STEEP_VIEW  θ(худший) ≈ 25.38°  σ_width ≈ 18.3 мм  meets_tolerance = False
MILD_VIEW = {"dx": -5000.0, "dy": -3500.0, "dist": 10000.0}
STEEP_VIEW = {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0}

#: Ракурс, у которого ЦЕНТР проёма лежит внутри углового условия (29.66°), а один
#: из углов контура — снаружи (34.46°). Четверть проёма при этом лежит вне
#: пригодной области, и элемент обязан отвергаться: п. 2.2 нормирует пригодную
#: область условием на ТОЧКИ, а не на центр элемента.
CORNER_OUTSIDE_VIEW = {"dx": -500.0, "dy": -1000.0, "dist": 9500.0}

#: Ракурс, на котором θ_⊥ видимой грани откоса лежит МЕЖДУ вычисляемым порогом
#: п. 6.2 (≈10.3° при здешнем GSD) и отозванной спецификацией константой 15°:
#: глубина обязана измеряться. Подстановка константы 15° вместо вычисления
#: превратила бы измеримую грань в `unavailable`.
SHALLOW_REVEAL_VIEW = {"dx": -3500.0, "dy": -2500.0, "dist": 10000.0}


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


def _expected_size_sigma(size_mm, theta_deg, gsd_local, *, sigma_px, sigma_rel,
                         residual_mm, flatness_mm=WALL_FLATNESS_DEVIATION_MM,
                         distortion_mm=RESIDUAL_DISTORTION_MM, focal_rel=0.0):
    """Шесть слагаемых п. 6.1, записанные здесь буквально, а не взятые у кода.

    Именно эта независимая запись и делает проверку проверкой: ожидание считается
    из ВХОДА теста по формуле спецификации, поэтому любая порча формулы в
    реализации — снятый √2, снятый множитель размера, `tg θ → θ`, линейная сумма
    вместо RSS, выброшенное слагаемое — расходится с ним численно.
    """
    return math.hypot(
        math.sqrt(2.0) * sigma_px * gsd_local,           # локализация двух кромок
        size_mm * sigma_rel,                             # масштаб
        residual_mm,                                     # остаточная невязка
        distortion_mm,                                   # остаточная дисторсия
        focal_rel * size_mm * math.sin(math.radians(theta_deg)) ** 2,   # фокусное
        flatness_mm * math.tan(math.radians(theta_deg)),                # неплоскостность
    )


def _gsd_field_of(H, mm_per_unit):
    return local_gsd_field(H, mm_per_unit, SIZE, shape=FIELD_SHAPE).gsd


def _window_mark(sc, *, edge_type="sharp_wall_edge", mounting="embedded", reveal=None):
    op = sc.openings[0]
    corners = [tuple(map(float, pt)) for pt in sc.project(op.corners_mm())]
    payload = {"class": "window", "mounting": mounting, "edge_type": edge_type,
              "corners_px": corners}
    if reveal is not None:
        payload["reveal"] = reveal
    return ElementMark.model_validate(payload)


def _digitize(sc, marks, **overrides):
    H, camera, mm_per_unit, origin_rect = exact_geometry(sc)
    kwargs = {"H": H, "camera": camera, "mm_per_unit": mm_per_unit,
             "origin_rect": origin_rect, "image_size": SIZE,
             "gsd_field": _gsd_field_of(H, mm_per_unit), "sigma_px": SIGMA_PX,
             "sigma_rel": SIGMA_REL, "residual_px": RESIDUAL_PX,
             "calibration": "target", "theta_max_deg": THETA_MAX_DEG}
    kwargs.update(overrides)
    return digitize_elements(marks, **kwargs)


def _worst_corner_theta(sc):
    """Худший угол визирования по ИСТИННЫМ углам проёма — без обращения к модулю."""
    _, camera, _, _ = exact_geometry(sc)
    corners = sc.openings[0].corners_mm()
    return max(camera.theta_deg(float(x), float(y)) for x, y in corners)


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
    # Ширина и высота не перепутаны местами: проём не квадратный (1460 x 1900).
    assert abs(size.width - op.width) < abs(size.width - op.height)
    assert element.id == "w_000"
    assert element.class_name == "window"
    assert len(element.contour_mm) == 4


def test_size_is_the_mean_of_both_pairs_of_opposite_sides():
    """`_quad_size_mm` берёт СРЕДНЕЕ пары противоположных сторон, а не одну сторону.

    Проверяется на заведомо НЕпрямоугольной разметке: нижняя сторона 1460 мм,
    верхняя 1660 мм. Среднее — 1560; любая из сторон порознь дала бы 1460 либо
    1660, и обе величины отстоят от ожидания на 100 мм, то есть на порядок
    больше собственной σ. На прямоугольном проёме обе пары совпадают, и такая
    подмена оставалась бы незаметной.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    op = sc.openings[0]
    trapezoid_mm = np.array([
        [op.x, op.y],
        [op.x + 1460.0, op.y],
        [op.x + 1660.0, op.y + op.height],
        [op.x, op.y + op.height],
    ])
    mark = ElementMark.model_validate({
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": [tuple(map(float, pt)) for pt in sc.project(trapezoid_mm)]})
    element = _digitize(sc, [mark])[0]
    assert element.size_mm.width == pytest.approx(1560.0, abs=2.0)


@pytest.mark.parametrize("view_name,view", [("mild", MILD_VIEW), ("steep", STEEP_VIEW)])
def test_sigma_equals_the_exact_rss_of_paragraph_6_1(view_name, view):
    """σ габарита — ТОЧНО квадратичная сумма шести слагаемых п. 6.1.

    Поле разрешения подано постоянным, поэтому `GSD_local` известно точно и не
    приходится повторять в тесте выбор ближайшего узла. Всё остальное — размер и
    угол — берётся из самого элемента, поэтому равенство проверяет формулу, а не
    согласие двух реализаций геометрии.

    Проверка идёт на ДВУХ ракурсах не для полноты: при θ ≈ 1° `tg θ` и θ в
    радианах совпадают в пятом знаке, и подмена одного другим на пологом ракурсе
    неразличима. На 25° расхождение — единицы миллиметров.
    """
    sc = make_scene(depth=150.0, **view)
    gsd = 4.0
    element = _digitize(sc, [_window_mark(sc)],
                        gsd_field=np.full(FIELD_SHAPE, gsd))[0]

    expected_width = _expected_size_sigma(
        element.size_mm.width, element.theta.full_deg, gsd,
        sigma_px=SIGMA_PX, sigma_rel=SIGMA_REL, residual_mm=RESIDUAL_PX * gsd)
    expected_height = _expected_size_sigma(
        element.size_mm.height, element.theta.full_deg, gsd,
        sigma_px=SIGMA_PX, sigma_rel=SIGMA_REL, residual_mm=RESIDUAL_PX * gsd)

    assert element.size_mm.sigma_width == pytest.approx(expected_width, rel=1e-12)
    assert element.size_mm.sigma_height == pytest.approx(expected_height, rel=1e-12)


def test_sigma_carries_the_residual_distortion_term_of_paragraph_6_1():
    """Строка «остаточная дисторсия» п. 6.1 входит в σ, а не снимается `undistort`.

    Строка названа ОСТАТОЧНОЙ: она описывает то, что остаётся ПОСЛЕ коррекции, и
    стоит в колонке `assisted`, где калибровка предполагается. Снятие её со
    ссылкой на `calib.undistort` обнуляло бы слагаемое, которого этот шаг не
    трогает: при нулевых коэффициентах `undistort` возвращает исходный массив.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    gsd = 4.0
    element = _digitize(sc, [_window_mark(sc)],
                        gsd_field=np.full(FIELD_SHAPE, gsd))[0]
    without_distortion = _expected_size_sigma(
        element.size_mm.width, element.theta.full_deg, gsd, sigma_px=SIGMA_PX,
        sigma_rel=SIGMA_REL, residual_mm=RESIDUAL_PX * gsd, distortion_mm=0.0)
    assert element.size_mm.sigma_width > without_distortion + 0.5


def test_sigma_carries_the_focal_error_only_when_the_camera_is_not_calibrated():
    """Слагаемое ошибки фокусного `ε·L·sin²θ` зависит от происхождения K.

    Источник K — свойство СНИМКА, он записан в выход (`camera.calibration`), и
    докстринг `calib.intrinsics_from_meta` прямо называет причину, по которой он
    обязан туда попасть: без него нельзя оценить этот вклад. Задача 18 — первый
    его потребитель.
    """
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    gsd = 4.0
    field = np.full(FIELD_SHAPE, gsd)
    calibrated = _digitize(sc, [_window_mark(sc)], gsd_field=field,
                           calibration="target")[0]
    from_database = _digitize(sc, [_window_mark(sc)], gsd_field=field,
                              calibration="database")[0]

    assert from_database.size_mm.sigma_width > calibrated.size_mm.sigma_width
    expected = _expected_size_sigma(
        from_database.size_mm.width, from_database.theta.full_deg, gsd,
        sigma_px=SIGMA_PX, sigma_rel=SIGMA_REL, residual_mm=RESIDUAL_PX * gsd,
        focal_rel=FOCAL_RELATIVE_ERROR)
    assert from_database.size_mm.sigma_width == pytest.approx(expected, rel=1e-12)
    # sin²θ, а не sinθ: на 25° различие — вдвое с лишним.
    wrong_power = _expected_size_sigma(
        from_database.size_mm.width, from_database.theta.full_deg, gsd,
        sigma_px=SIGMA_PX, sigma_rel=SIGMA_REL, residual_mm=RESIDUAL_PX * gsd,
        focal_rel=FOCAL_RELATIVE_ERROR
        / math.sin(math.radians(from_database.theta.full_deg)))
    assert from_database.size_mm.sigma_width != pytest.approx(wrong_power, rel=1e-3)


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
    gsd = 4.0
    field = np.full(FIELD_SHAPE, gsd)
    small = _digitize(sc, [_window_mark(sc)], residual_px=0.1, gsd_field=field)[0]
    large = _digitize(sc, [_window_mark(sc)], residual_px=50.0, gsd_field=field)[0]
    unmeasured = _digitize(sc, [_window_mark(sc)], residual_px=None, gsd_field=field)[0]

    assert large.size_mm.sigma_width > small.size_mm.sigma_width
    assert unmeasured.size_mm.sigma_width == pytest.approx(
        _expected_size_sigma(unmeasured.size_mm.width, unmeasured.theta.full_deg, gsd,
                             sigma_px=SIGMA_PX, sigma_rel=SIGMA_REL,
                             residual_mm=RESIDUAL_REFERENCE_MM), rel=1e-12)


def test_wall_flatness_is_a_named_and_overridable_assumption():
    """Неплоскостность стены — ДОПУЩЕНИЕ, названное и перекрываемое вызывающим.

    Все прочие слагаемые п. 6.1 берутся по факту (измеренное разрешение,
    измеренная невязка, фактический угол), а это — принятая величина из
    диапазона ±20–50 мм. Она обязана называться и перекрываться: иначе
    `meets_tolerance` на крутом ракурсе определяется не геометрией снимка, а
    константой модуля, умноженной на тангенс.
    """
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    gsd = 4.0
    field = np.full(FIELD_SHAPE, gsd)
    default = _digitize(sc, [_window_mark(sc)], gsd_field=field)[0]
    flat_wall = _digitize(sc, [_window_mark(sc)], gsd_field=field,
                          wall_flatness_mm=20.0)[0]

    assert flat_wall.size_mm.sigma_width < default.size_mm.sigma_width
    assert flat_wall.size_mm.sigma_width == pytest.approx(
        _expected_size_sigma(flat_wall.size_mm.width, flat_wall.theta.full_deg, gsd,
                             sigma_px=SIGMA_PX, sigma_rel=SIGMA_REL,
                             residual_mm=RESIDUAL_PX * gsd, flatness_mm=20.0),
        rel=1e-12)
    # Умолчание лежит внутри диапазона п. 6.1 и не совпадает с его краями.
    assert 20.0 < WALL_FLATNESS_DEVIATION_MM < 50.0
    # И воспроизводит собственный вывод п. 6.1 о рабочем диапазоне: на
    # благоприятной ветви остальных диапазонов признак переключается при
    # θ ≲ 15–20°, а не вчетверо раньше. Остаток бюджета на неплоскостность там —
    # `WALL_FLATNESS_BUDGET_MM` (вывод — в докстринге константы).
    switch_deg = math.degrees(math.atan(WALL_FLATNESS_BUDGET_MM
                                        / WALL_FLATNESS_DEVIATION_MM))
    assert 15.0 <= switch_deg <= 20.0


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


def test_meets_tolerance_takes_the_worse_of_the_two_sigmas():
    """Сравнивается ХУДШАЯ из σ ширины и высоты, а не лучшая.

    Обе рабочие фикстуры этого модуля держат σ_width и σ_height по ОДНУ сторону
    допуска, и на них консервативный выбор (`max`) неотличим от
    неконсервативного (`min`). Здесь вход подобран так, что они лежат по РАЗНЫЕ
    стороны: σ_width ≈ 9.4 мм, σ_height ≈ 10.6 мм при допуске 10. Признак обязан
    быть ложным — габарит проёма не выдан в допуске ни при каком выборе стороны.

    σ масштаба взята 0.40 %, а не 0.2 %: оговорка задачи 14 о короткой опорной
    базе — реальный оператор берёт базу вдесятеро короче двадцатиметровой
    стороны фасада, и σ поднимается вместе с ней. Именно эта величина и разводит
    σ ширины и высоты: она единственная входит пропорционально габариту.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    gsd = 2.0
    element = _digitize(sc, [_window_mark(sc)], gsd_field=np.full(FIELD_SHAPE, gsd),
                        sigma_rel=0.0040, residual_px=1.0)[0]

    assert element.size_mm.sigma_width < ASSISTED_SIZE_TOLERANCE_MM
    assert element.size_mm.sigma_height > ASSISTED_SIZE_TOLERANCE_MM
    assert element.meets_tolerance is False


# --- 4. Элемент вне пригодной области отвергается, а не тихо посчитан -------------


def test_element_beyond_the_angle_threshold_is_refused():
    """Угол визирования на проём выше порога — отказ с названной причиной."""
    sc = make_scene(dx=17000.0, dy=-500.0, dist=9000.0, depth=150.0)
    mark = _window_mark(sc)
    assert _worst_corner_theta(sc) > THETA_MAX_DEG   # проверка предположения о сцене

    with pytest.raises(ValueError, match="угол визирования .* превышает порог"):
        _digitize(sc, [mark])


def test_element_is_refused_when_one_corner_leaves_the_usable_area():
    """Пригодная область нормируется по ТОЧКАМ контура, а не по центру элемента.

    Ракурс подобран так, что центр проёма лежит внутри углового условия
    (≈29.7° при пороге 30°), а один из углов — снаружи (≈34.5°): четверть проёма
    измеряется там, где спецификация, п. 2.2 измерять запрещает. Угол по
    центроиду принял бы такой элемент как годный, и `_gsd_near`, которая худший
    случай берёт явно, оказалась бы непоследовательна сама с собой.
    """
    sc = make_scene(depth=150.0, **CORNER_OUTSIDE_VIEW)
    _, camera, _, _ = exact_geometry(sc)
    corners = sc.openings[0].corners_mm()
    centroid = corners.mean(axis=0)
    assert camera.theta_deg(float(centroid[0]), float(centroid[1])) < THETA_MAX_DEG
    assert _worst_corner_theta(sc) > THETA_MAX_DEG

    with pytest.raises(ValueError, match="угол визирования .* превышает порог"):
        _digitize(sc, [_window_mark(sc)])


def test_element_theta_is_the_worst_corner_not_the_centroid():
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    element = _digitize(sc, [_window_mark(sc)])[0]
    _, camera, _, _ = exact_geometry(sc)
    corners = sc.openings[0].corners_mm()
    centroid = corners.mean(axis=0)
    theta_centroid = camera.theta_deg(float(centroid[0]), float(centroid[1]))

    assert element.theta.full_deg == pytest.approx(_worst_corner_theta(sc), abs=0.05)
    assert element.theta.full_deg > theta_centroid + 1.0
    # Компоненты взяты в ТОЙ ЖЕ точке, что и полный угол, а не порознь.
    assert math.degrees(math.atan(math.hypot(
        math.tan(math.radians(element.theta.x_deg)),
        math.tan(math.radians(element.theta.y_deg))))) == pytest.approx(
            element.theta.full_deg, rel=1e-9)


def test_element_behind_the_vanishing_line_is_refused():
    """Проём за линией схода — отказ, а не подстановка мусорных координат.

    Ракурс — `tests/test_homography.py::GRAZING_VIEW` (камера в 200 мм от
    плоскости, 20 м вбок): линия схода плоскости фасада проходит в 36 px от
    главной точки, и правая половина кадра лежит по другую сторону от неё
    (измерено: точка (5279, 1978) даёт знаменатель гомографии знака,
    противоположного центру кадра). Метка ставится заведомо там.
    """
    sc = make_scene(dx=-20000.0, dy=0.0, dist=200.0, depth=150.0)
    bad_mark = ElementMark.model_validate({
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": [[5100.0, 1900.0], [5279.0, 1900.0],
                       [5279.0, 2000.0], [5100.0, 2000.0]],
    })
    with pytest.raises(ValueError, match="линией схода"):
        _digitize(sc, [bad_mark])


def test_marks_outside_the_frame_are_refused_by_name():
    """Углы разметки обязаны лежать В КАДРЕ — тот же контракт, что у опорных точек.

    Без этой охраны `_nearest_node` зажимает индекс к краю сетки, и локальное
    разрешение «у проёма» берётся из совсем другого места кадра — ровно тот
    дефект, который проект уже нашёл и закрыл для опорной точки оператора
    (`run.py:_points_in_frame`). Пиксельные координаты привязаны к снимку, и
    разметка, перенесённая на чужой кадр, дала бы уверенные миллиметры для окна,
    которого там нет.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    outside = ElementMark.model_validate({
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": [[100.0, 100.0], [SIZE[0] + 400.0, 100.0],
                       [SIZE[0] + 400.0, 600.0], [100.0, 600.0]]})
    with pytest.raises(ValueError, match="вне кадра"):
        _digitize(sc, [outside])


def test_reveal_marks_outside_the_frame_are_refused_by_name():
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    mark, _ = _reveal_mark_for(sc)
    payload = mark.model_dump(by_alias=True)
    payload["reveal"]["inner_edge_px"] = [(-5.0, 100.0), (-5.0, 600.0)]
    with pytest.raises(ValueError, match="вне кадра"):
        _digitize(sc, [ElementMark.model_validate(payload)])


# --- 5. Глубина заглубления: значение, порог, неверная сторона --------------------


def _reveal_mark_for(sc, side=None, **mark_kwargs):
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
    return _window_mark(sc, reveal=reveal, **mark_kwargs), vertical


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


def test_recess_sigma_carries_the_sec_squared_phi_conversion():
    """σ глубины считается с переводом `σ_u = C_z · sec²φ · σ_θ` (п. 6.2).

    Множитель `sec²φ` спецификация выписала явно именно потому, что его пропажа
    незаметна: σ остаётся правдоподобной, только заниженной. Ожидание собирается
    здесь из ИСТИНЫ СЦЕНЫ (кромка, |u|, C_z) и публичной `reveal_depth_sigma`,
    а не из внутренностей `elements.py`.
    """
    sc = make_scene(depth=150.0, **STEEP_VIEW)
    mark, side = _reveal_mark_for(sc)
    gsd = 4.0
    element = _digitize(sc, [mark], gsd_field=np.full(FIELD_SHAPE, gsd))[0]

    op = sc.openings[0]
    cam = sc.camera_on_plane()
    edge_x = op.x if side == "left" else op.x + op.width
    u_mm = abs(edge_x - cam.cx)
    depth = element.recess.value_mm
    width_mm = depth * u_mm / (cam.cz + depth)
    phi = math.atan(u_mm / cam.cz)
    sigma_u = cam.cz / math.cos(phi) ** 2 * math.radians(REFERENCE_SIGMA_THETA_DEG)
    expected = reveal_depth_sigma(width_mm, u_mm, cam.cz,
                                  math.sqrt(2.0) * SIGMA_PX * gsd, sigma_u,
                                  REFERENCE_SIGMA_CZ_REL * cam.cz)
    without_sec2 = reveal_depth_sigma(
        width_mm, u_mm, cam.cz, math.sqrt(2.0) * SIGMA_PX * gsd,
        cam.cz * math.radians(REFERENCE_SIGMA_THETA_DEG),
        REFERENCE_SIGMA_CZ_REL * cam.cz)

    assert element.recess.sigma_mm == pytest.approx(expected, rel=2e-3)
    assert element.recess.sigma_mm != pytest.approx(without_sec2, rel=2e-3)


def test_recess_threshold_is_computed_not_the_withdrawn_fifteen_degrees():
    """Порог применимости п. 6.2 ВЫЧИСЛЯЕТСЯ, а не равен отозванным 15°.

    Ракурс подобран так, что θ_⊥ ≈ 13.8° — ниже отозванной константы 15°, но
    выше вычисленного по фактическим σ_px и GSD порога (≈10.3°). Глубина обязана
    измеряться; константа 15° на этом же входе объявила бы грань непригодной.
    """
    sc = make_scene(depth=150.0, **SHALLOW_REVEAL_VIEW)
    mark, _ = _reveal_mark_for(sc)
    element = _digitize(sc, [mark])[0]

    assert element.recess.origin == "measured_from_reveal"
    assert 10.5 < element.recess.theta_perp_deg < 15.0
    assert element.recess.value_mm == pytest.approx(150.0, rel=0.05)


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


# --- 5a. Кромка, про которую не известно, что она в плоскости стены ---------------


@pytest.mark.parametrize("edge_type,mounting,edge_reference", [
    ("sharp_wall_edge", "embedded", "wall_plane"),
    ("sharp_wall_edge", "flush", "wall_plane"),
    ("surround", "embedded", "offset_plane"),
    ("cladding_edge", "embedded", "offset_plane"),
    ("sharp_wall_edge", "protruding", "offset_plane"),
    ("unknown", "embedded", "unknown"),
])
def test_edge_reference_is_not_asserted_for_edges_that_are_not_in_the_wall_plane(
        edge_type, mounting, edge_reference):
    """`edge_reference` — утверждение о том, ЧЕМУ равен измеренный контур.

    Спецификация, п. 5.3 определяет измеряемую кромку как линию пересечения
    плоскости стены и плоскости откоса и фиксирует это полем
    `edge_reference: "wall_plane"`. Она же говорит, что при `surround` и
    `cladding_edge` элемент обрабатывается как выступающий (п. 5.6), то есть
    кромка в Π НЕ лежит, а при `unknown` про кромку не известно ничего.
    Безусловное `wall_plane` утверждало бы ровно то, что п. 5.3 в трёх случаях
    из четырёх сам объявляет неверным.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    mark = _window_mark(sc, edge_type=edge_type, mounting=mounting)
    element = _digitize(sc, [mark])[0]
    assert element.edge_reference == edge_reference


def test_an_unreferenced_edge_is_not_metrically_interpretable():
    """П. 5.6: без оценки выноса габарит выдаётся БЕЗ метрической интерпретации.

    Габарит и его σ выпускаются по-прежнему — их считает та же геометрия. Чего
    выпускать нельзя, так это признака соответствия допуску: вынос обрамления
    даёт смещение 10–45 мм (п. 6.1, строка параллакса), то есть больше всего
    допуска п. 2.2, и оно не разброс, а СДВИГ — в квадратичную сумму его класть
    нечего. Поэтому `meets_tolerance` отсутствует, а не равен `False`: «не
    проверено» и «проверено и не прошло» — разные утверждения.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    known = _digitize(sc, [_window_mark(sc, edge_type="sharp_wall_edge")])[0]
    unknown = _digitize(sc, [_window_mark(sc, edge_type="unknown")])[0]
    surround = _digitize(sc, [_window_mark(sc, edge_type="surround")])[0]

    assert known.meets_tolerance is True
    assert unknown.meets_tolerance is None
    assert surround.meets_tolerance is None
    # Габарит и σ при этом те же: геометрия не зависит от того, что мы про кромку
    # знаем, — зависит лишь право утверждать, к чему этот габарит отнесён.
    assert unknown.size_mm.width == pytest.approx(known.size_mm.width, rel=1e-12)
    assert unknown.size_mm.sigma_width == pytest.approx(known.size_mm.sigma_width,
                                                        rel=1e-12)


def test_facade_boundary_is_refused_by_the_mark_format():
    """`facade_boundary` — ПОЛИЛИНИЯ с атрибутом выноса (п. 2.3), а не проём.

    Формат разметки задачи 18 несёт четырёхугольник и не несёт `offset_mm`,
    которого у схемы нет вовсе. Выпустить границу фасада четырёхугольником с
    габаритом, σ и признаком допуска значило бы приписать полилинии ширину и
    высоту и объявить их в допуске — величины, которых п. 2.3 для этого класса
    не определяет.
    """
    sc = make_scene(depth=150.0, **MILD_VIEW)
    payload = {"class": "facade_boundary", "mounting": "flush",
               "edge_type": "sharp_wall_edge",
               "corners_px": [tuple(map(float, pt)) for pt in sc.project(FACADE_CORNERS)]}
    with pytest.raises(ValidationError, match="полилини"):
        ElementMark.model_validate(payload)


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


def _scene_reference(sc):
    far = np.array([[20000.0, 0.0]])
    pts = sc.project(np.vstack([FACADE_CORNERS[:1], far]))
    return OperatorReference(
        origin_px=(float(pts[0][0]), float(pts[0][1])),
        span_px=((float(pts[0][0]), float(pts[0][1])),
                 (float(pts[1][0]), float(pts[1][1]))),
        span_mm=20000.0), pts


def test_saved_raster_matches_the_recorded_homography(tmp_path):
    """Точка исходного снимка через записанные H и origin попадает в растре туда,
    где она есть на самом деле."""
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    image = sc.render()
    path = tmp_path / "facade.png"
    save_image(path, image)

    reference, _ = _scene_reference(sc)
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


def test_parse_marks_refuses_an_empty_list():
    """Пустая разметка — не разметка: `--marks` с `[]` обязан называть причину.

    Молча приравняв её к отсутствию ключа, конвейер уравнял бы два разных
    указания оператора, в том числе на пути `needs_operator`, где НЕПУСТАЯ
    разметка обязана поднять отказ, а пустая проходила бы насквозь.
    """
    with pytest.raises(ValueError, match="пуст"):
        parse_marks([])


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


def _cli_scene(tmp_path):
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=20000.0, depth=150.0)
    path = tmp_path / "facade.png"
    save_image(path, sc.render())
    _, pts = _scene_reference(sc)
    op = sc.openings[0]
    corners_px = [tuple(map(float, pt)) for pt in sc.project(op.corners_mm())]
    return sc, path, pts, corners_px


def _cli_args(path, pts, marks_path=None, out_dir=None, extra=()):
    args = [str(path), "--raster-mm-per-px", "10.0",
            "--origin-px", str(pts[0][0]), str(pts[0][1]),
            "--span-px", str(pts[0][0]), str(pts[0][1]), str(pts[1][0]), str(pts[1][1]),
            "--span-mm", "20000.0"]
    if marks_path is not None:
        args += ["--marks", str(marks_path)]
    if out_dir is not None:
        args += ["--out-dir", str(out_dir)]
    return args + list(extra)


def test_cli_digitizes_marked_elements_and_saves_the_raster(tmp_path):
    _, path, pts, corners_px = _cli_scene(tmp_path)
    marks_path = tmp_path / "marks.json"
    marks_path.write_text(json.dumps([{
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": corners_px}]), encoding="utf-8")

    out_dir = tmp_path / "out"
    code = main(_cli_args(path, pts, marks_path, out_dir, extra=["--save-rectified"]))

    assert code == 0
    model = FacadeModel.model_validate_json((out_dir / "facade.json").read_text("utf-8"))
    assert len(model.elements) == 1
    assert model.elements[0].origin == "operator"
    # `contour_px` раздела 10: пиксельные клики оператора обязаны дойти до выхода —
    # без них демонстрация не воспроизводится, а задача 18 их и вводит.
    assert model.elements[0].contour_px == [
        {"image_id": model.images[0].id,
         "points": [list(pt) for pt in corners_px]}]

    from facade_digitizer.pipeline.io import load_image
    raster, _ = load_image(out_dir / "facade_rectified.png")
    mask, _ = load_image(out_dir / "facade_rectified_mask.png")
    # Маска проверяется СОДЕРЖИМЫМ, а не только существованием файла: маска из
    # одних нулей — это «охвата нет», и она прошла бы проверку на существование.
    assert mask.shape == raster.shape
    assert int(mask.max()) == 255
    assert float((mask > 0).mean()) > 0.5


def test_cli_refuses_one_marking_for_a_batch_of_images(tmp_path):
    """Одна разметка на пакет снимков — отказ: пиксели привязаны к СНИМКУ.

    `--marks` читается один раз, и без этой охраны та же разметка подавалась бы
    на каждый снимок пакета без сверки чего бы то ни было: координаты окна с
    первого кадра дали бы уверенные миллиметры на втором, где этого окна нет.
    """
    sc, path, pts, corners_px = _cli_scene(tmp_path)
    second = tmp_path / "facade2.png"
    save_image(second, sc.render())
    marks_path = tmp_path / "marks.json"
    marks_path.write_text(json.dumps([{
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": corners_px}]), encoding="utf-8")

    out_dir = tmp_path / "out"
    args = _cli_args(path, pts, marks_path, out_dir)
    args.insert(1, str(second))
    assert main(args) == 1
    assert not (out_dir / "facade.json").exists()


def test_cli_refuses_an_empty_marks_file(tmp_path):
    _, path, pts, _ = _cli_scene(tmp_path)
    marks_path = tmp_path / "marks.json"
    marks_path.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(_cli_args(path, pts, marks_path, out_dir)) == 1
    assert not (out_dir / "facade.json").exists()


def test_coverage_and_origin_stay_a_consistent_pair(tmp_path):
    """П. 7 связывает `coverage` с тем, ГДЕ лежит начало координат.

    Начало здесь всегда опорная точка оператора (`facade.origin =
    "operator_reference"`), а её п. 7 сопоставляет частичному охвату. Пара была
    согласована до задачи 18 и обязана остаться согласованной: `coverage =
    "full"` рядом с `origin = "operator_reference"` утверждает два взаимно
    исключающих по п. 7 факта.
    """
    sc, path, _, _ = _cli_scene(tmp_path)
    reference, _ = _scene_reference(sc)
    mark = _window_mark(sc)

    without = process(path, operator_reference=reference, raster_mm_per_px=10.0)
    with_marks = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                         marks=[mark])

    for model in (without, with_marks):
        assert model.facade.origin == "operator_reference"
        assert model.coverage == "partial"


# --- 9. Стык `run` -> `elements`: в элементы уходит то, что записано в выход -------


def test_process_feeds_elements_the_geometry_and_the_numbers_it_records(tmp_path):
    """Центральное утверждение модуля: «та же геометрия, что записана в выход».

    До сих пор оно не держалось ничем: тесты звали `digitize_elements` напрямую
    со своими значениями, а три теста через `process()` не проверяли ни одного
    числа. Здесь σ габарита ПЕРЕСЧИТЫВАЕТСЯ из самого выходного файла и
    сверяется с тем, что в нём записано.

    Поле разрешения восстанавливается из записанных `homography.H` (изображение
    -> пиксели растра) и `facade.mm_per_rectified_px` (миллиметры на пиксель
    растра). Это независимый от `run.py` путь к той же величине: `rectify`
    домножает гомографию на подобие с коэффициентом `mm_per_unit / mm_per_px`,
    и произведение масштабов совпадает с `mm_per_unit` тождественно. Разойдись
    геометрия, ушедшая в элементы, с записанной — σ разошлась бы здесь.

    `_gsd_near` берётся у модуля намеренно: проверяется не выбор ближайшего узла
    (у него свой тест), а то, ПО КАКОМУ ПОЛЮ он сделан.
    """
    sc, path, _, corners_px = _cli_scene(tmp_path)
    reference, _ = _scene_reference(sc)
    mark_sigma_px = 2.5
    model = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                    mark_sigma_px=mark_sigma_px, marks=[_window_mark(sc)])

    element = model.elements[0]
    record = model.images[0]
    assert record.rectification.residual_px is not None
    field = local_gsd_field(np.asarray(record.homography["H"], dtype=float),
                            model.facade.mm_per_rectified_px, SIZE,
                            shape=FIELD_SHAPE).gsd
    gsd_local = _gsd_near(field, SIZE, corners_px, label="check")
    focal_rel = 0.0 if record.camera.calibration == "target" else FOCAL_RELATIVE_ERROR

    for size, sigma in ((element.size_mm.width, element.size_mm.sigma_width),
                        (element.size_mm.height, element.size_mm.sigma_height)):
        expected = _expected_size_sigma(
            size, element.theta.full_deg, gsd_local, sigma_px=mark_sigma_px,
            sigma_rel=model.facade.scale.sigma_rel,
            residual_mm=record.rectification.residual_px * gsd_local,
            focal_rel=focal_rel)
        assert sigma == pytest.approx(expected, rel=1e-9)


def test_process_maps_the_recorded_pixels_to_the_recorded_millimetres(tmp_path):
    """`contour_mm` восстанавливается из `contour_px` ОДНОЙ ЛИШЬ записанной геометрией.

    Проверка на самих КООРДИНАТАХ, а не на σ, и она нужна отдельно. Масштаб,
    разошедшийся на стыке `run` → `elements`, меняет и габарит элемента, и его
    угол; σ же пересчитывается по ним обоим — и сходится сама с собой, оставаясь
    верной σ неверного габарита. Измерено: подмена `mm_per_unit` на
    `mm_per_unit · 1.01` в вызове `digitize_elements` проходит ВЕСЬ набор,
    включая точную сверку σ, и ловится только здесь.

    Ни `plane.H`, ни `mm_per_unit` сборки тест не знает: берутся только
    `homography.H` (изображение → пиксели растра), `homography.origin_rect_px` и
    `facade.mm_per_rectified_px` выходного файла. Тождество следует из устройства
    `rectify`, которая домножает гомографию на подобие с коэффициентом
    `mm_per_unit / mm_per_px`:

        (p − origin_rect_px) · mm_per_rectified_px ≡ (rect − origin_rect) · mm_per_unit

    а правая часть и есть то, что `rectified_to_facade_mm` кладёт в `contour_mm`
    (с переворотом оси Y: растр смотрит вниз, фасад вверх).
    """
    sc, path, _, _ = _cli_scene(tmp_path)
    reference, _ = _scene_reference(sc)
    model = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                    marks=[_window_mark(sc)])

    element = model.elements[0]
    homography = model.images[0].homography
    rect_px = apply_homography(np.asarray(homography["H"], dtype=float),
                               element.contour_px[0]["points"])
    ox, oy = homography["origin_rect_px"]
    mm_per_px = model.facade.mm_per_rectified_px
    expected = np.column_stack([(rect_px[:, 0] - ox) * mm_per_px,
                                -(rect_px[:, 1] - oy) * mm_per_px])

    assert np.asarray(element.contour_mm) == pytest.approx(expected, rel=1e-9, abs=1e-6)
    # И тот же габарит, что у проёма сцены: тождество выше проверяет согласие
    # выхода с самим собой, а это — согласие с истиной, от которой тождество
    # свободно (обе его части сдвинулись бы одинаково при общем сдвиге масштаба).
    op = sc.openings[0]
    assert element.size_mm.width == pytest.approx(op.width, abs=3.0 * element.size_mm.sigma_width)
    assert element.size_mm.height == pytest.approx(op.height, abs=3.0 * element.size_mm.sigma_height)


def test_marking_precision_is_a_separate_quantity_from_the_reference_base(tmp_path):
    """Точность клика по УГЛУ ПРОЁМА и по концам опорной БАЗЫ — разные величины.

    `DEFAULT_OPERATOR_SIGMA_PX` выведена под клик по двум концам двадцатиметровой
    базы на экране с уменьшением 2.75x (докстринг `run.py`). Клик по углу окна
    оператор делает с увеличением, и подстановка первой величины во вторую роль
    делает `meets_tolerance` и порог глубины константами: при GSD 2 мм/px одно
    слагаемое локализации даёт 12.7 мм при допуске 10.
    """
    sc, path, _, _ = _cli_scene(tmp_path)
    reference, _ = _scene_reference(sc)
    marks = [_window_mark(sc)]

    assert DEFAULT_MARK_SIGMA_PX != DEFAULT_OPERATOR_SIGMA_PX

    by_default = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                         marks=marks)
    explicit = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                       mark_sigma_px=DEFAULT_MARK_SIGMA_PX, marks=marks)
    as_base = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                      mark_sigma_px=DEFAULT_OPERATOR_SIGMA_PX, marks=marks)

    assert (by_default.elements[0].size_mm.sigma_width
            == pytest.approx(explicit.elements[0].size_mm.sigma_width, rel=1e-12))
    assert (as_base.elements[0].size_mm.sigma_width
            > by_default.elements[0].size_mm.sigma_width + 1.0)
    # σ масштаба при этом НЕ меняется: она считается из точности указания базы.
    assert (as_base.facade.scale.sigma_rel
            == pytest.approx(by_default.facade.scale.sigma_rel, rel=1e-12))


def test_cli_passes_the_marking_precision_through(tmp_path):
    _, path, pts, corners_px = _cli_scene(tmp_path)
    marks_path = tmp_path / "marks.json"
    marks_path.write_text(json.dumps([{
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": corners_px}]), encoding="utf-8")

    sigmas = []
    for value in ("1.0", "3.0"):
        out_dir = tmp_path / f"out{value}"
        assert main(_cli_args(path, pts, marks_path, out_dir,
                              extra=["--mark-sigma-px", value])) == 0
        model = FacadeModel.model_validate_json(
            (out_dir / "facade.json").read_text("utf-8"))
        sigmas.append(model.elements[0].size_mm.sigma_width)
    assert sigmas[1] > sigmas[0] + 1.0


def test_process_does_not_let_an_unmeasured_residual_reach_the_budget(monkeypatch,
                                                                     tmp_path):
    """Невязка, которой нет, обязана стать `None`, а не бесконечностью в σ.

    `rectification.residual_px` в записи уже обёрнута `_finite_or_none`; стык с
    элементами был единственным местом, где та же величина уходила как есть.
    Бесконечная невязка давала бы бесконечную σ у элемента при `null` в записи —
    то есть два разных ответа об одной величине в одном файле.
    """
    import dataclasses

    from facade_digitizer.pipeline import run as run_module

    real = run_module.estimate_plane

    def infinite_residual(image, K, *args, **kwargs):
        plane = real(image, K, *args, **kwargs)
        return dataclasses.replace(
            plane, confidence=dataclasses.replace(plane.confidence,
                                                  residual_px=float("inf")))

    monkeypatch.setattr(run_module, "estimate_plane", infinite_residual)

    sc, path, _, corners_px = _cli_scene(tmp_path)
    reference, _ = _scene_reference(sc)
    model = process(path, operator_reference=reference, raster_mm_per_px=10.0,
                    marks=[_window_mark(sc)])

    assert model.images[0].rectification.residual_px is None
    sigma = model.elements[0].size_mm.sigma_width
    assert math.isfinite(sigma)
    field = local_gsd_field(np.asarray(model.images[0].homography["H"], dtype=float),
                            model.facade.mm_per_rectified_px, SIZE,
                            shape=FIELD_SHAPE).gsd
    gsd_local = _gsd_near(field, SIZE, corners_px, label="check")
    calibration = model.images[0].camera.calibration
    assert sigma == pytest.approx(_expected_size_sigma(
        model.elements[0].size_mm.width, model.elements[0].theta.full_deg, gsd_local,
        sigma_px=DEFAULT_MARK_SIGMA_PX, sigma_rel=model.facade.scale.sigma_rel,
        residual_mm=RESIDUAL_REFERENCE_MM,
        focal_rel=0.0 if calibration == "target" else FOCAL_RELATIVE_ERROR),
        rel=1e-9)


def test_marks_are_refused_when_the_plane_was_not_recovered(tmp_path):
    """Непустая разметка на пути `needs_operator` — отказ, а не пустой `elements`."""
    image = np.full((600, 800), 120, dtype=np.uint8)
    path = tmp_path / "flat.png"
    save_image(path, image)
    reference = OperatorReference(origin_px=(10.0, 10.0),
                                  span_px=((10.0, 10.0), (700.0, 10.0)),
                                  span_mm=20000.0)
    mark = ElementMark.model_validate({
        "class": "window", "mounting": "embedded", "edge_type": "sharp_wall_edge",
        "corners_px": [[10.0, 10.0], [50.0, 10.0], [50.0, 60.0], [10.0, 60.0]]})
    with pytest.raises(ValueError, match="разметка элементов недоступна"):
        process(path, operator_reference=reference, raster_mm_per_px=50.0,
                marks=[mark])
