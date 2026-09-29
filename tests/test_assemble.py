"""Сведение погрешностей. План 3, задача 21; спецификация, п. 4.4."""
import math

import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline.assemble import (
    Inputs,
    monte_carlo,
    nearest_neighbours,
    propagate,
)
from tests.test_synth import make_scene

#: Ряд окон: 4 колонны × 2 этажа, шаг 4000 × 5000 мм — «номинально одинаковые окна».
WINDOWS = [(2000.0 + 4000.0 * i, 2000.0 + 5000.0 * j) for j in range(2) for i in range(4)]
W, H_OP = 1460.0, 1900.0


def _window_mm(x, y):
    return np.array([[x, y], [x + W, y], [x + W, y + H_OP], [x, y + H_OP]])


@pytest.fixture(scope="module")
def geometry():
    """Точная гомография кадр → миллиметры фасада (ось y вниз), одна единица = 1 мм."""
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=12000.0)
    grid = np.array([[x, y] for x in (0.0, 10000.0, 20000.0) for y in (0.0, 7500.0, 15000.0)])
    img = sc.project(grid)
    rect = np.column_stack([grid[:, 0], -grid[:, 1]])
    H, _ = cv2.findHomography(img, rect)
    corners = tuple(sc.project(_window_mm(x, y)) for x, y in WINDOWS)
    origin_px = tuple(sc.project(np.array([[0.0, 0.0]]))[0])
    return H, corners, origin_px


def _inputs(geometry, *, origin_sigma=1.9, scale_sigma_rel=0.002, click=1.2):
    H, corners, origin_px = geometry
    return Inputs(H=H, mm_per_unit=1.0, origin_px=origin_px, origin_sigma_px=origin_sigma,
                  scale_sigma_rel=scale_sigma_rel, corners_px=corners,
                  corner_sigma_px=(click,) * len(corners))


def test_nominal_outputs_reproduce_the_scene(geometry):
    prop = propagate(_inputs(geometry))
    # Гомография оценена по сетке точек в плавающей точке: сотые доли микрона.
    assert prop.nominal[:, 0] == pytest.approx(W, abs=1e-3)
    assert prop.nominal[:, 1] == pytest.approx(H_OP, abs=1e-3)
    assert prop.nominal[:, 2:4] == pytest.approx(np.array(WINDOWS), abs=1e-3)


def test_analytic_jacobian_matches_finite_differences(geometry):
    from facade_digitizer.pipeline.assemble import _jacobians, _jacobians_numeric

    inp = _inputs(geometry)
    _nominal, jg, jo = _jacobians(inp)
    for i in range(len(WINDOWS)):
        ng, no = _jacobians_numeric(inp, i)
        assert jg[i] == pytest.approx(ng, rel=1e-5, abs=1e-6)
        assert jo[i] == pytest.approx(no, rel=1e-5, abs=1e-6)


def test_linearised_sigma_matches_monte_carlo(geometry):
    inp = _inputs(geometry)
    prop = propagate(inp)
    sample = monte_carlo(inp, 3000, np.random.default_rng(0))
    for i in range(len(WINDOWS)):
        linear = np.sqrt(np.diag(prop.cov(i, i)))
        mc = sample[:, i, :].std(axis=0)
        assert mc == pytest.approx(linear, rel=0.10), f"элемент {i}"
    # И взаимное положение соседей — с корреляцией через общий масштаб.
    for i, j in enumerate(nearest_neighbours(prop.nominal[:, 2:4])):
        linear = np.sqrt(np.diag(prop.relative_cov(i, j)))
        diff = sample[:, i, 2:4] - sample[:, j, 2:4]
        assert diff.std(axis=0) == pytest.approx(linear, rel=0.10), f"пара {i}-{j}"


def test_scale_error_is_common_to_all_elements(geometry):
    """Общая ошибка масштаба и начала отсчёта почти целиком уходит из разности соседей."""
    prop = propagate(_inputs(geometry))
    for i, j in enumerate(nearest_neighbours(prop.nominal[:, 2:4])):
        absolute = math.sqrt(np.linalg.eigvalsh(prop.cov(i, i)[2:4, 2:4]).max())
        relative = math.sqrt(np.linalg.eigvalsh(prop.relative_cov(i, j)).max())
        assert relative < absolute, f"пара {i}-{j}"
    # Независимые ошибки (без общей части) такого выигрыша не дали бы: разность
    # двух точек тогда шумнее каждой из них в √2 раз.
    far = int(np.argmax(np.linalg.norm(prop.nominal[:, 2:4], axis=1)))
    near = nearest_neighbours(prop.nominal[:, 2:4])[far]
    independent = math.sqrt(prop.cov(far, far)[2, 2] + prop.cov(near, near)[2, 2])
    assert math.sqrt(prop.relative_cov(far, near)[0, 0]) < independent / 2


def test_size_sigma_agrees_with_the_closed_form_budget(geometry):
    """Расхождения с замкнутой формулой `elements._size_sigma_mm` — объяснённые.

    1. Масштаб: слагаемое `L · σ_rel` совпадает точно (ширина пропорциональна `k`).
    2. Локализация: замкнутая формула — `√2 · σ · GSD` (две кромки по клику), а
       ширина здесь — среднее двух сторон из четырёх независимых углов: `σ · GSD`.
       Отношение — ровно √2 при постоянном локальном разрешении.
    """
    H, corners, _origin_px = geometry
    scale_only = propagate(_inputs(geometry, origin_sigma=0.0, click=0.0,
                                   scale_sigma_rel=0.003))
    for i in range(len(WINDOWS)):
        assert math.sqrt(scale_only.cov(i, i)[0, 0]) == pytest.approx(W * 0.003, rel=1e-6)
        assert math.sqrt(scale_only.cov(i, i)[1, 1]) == pytest.approx(H_OP * 0.003, rel=1e-6)

    click = 1.3
    clicks_only = propagate(_inputs(geometry, origin_sigma=0.0, click=click,
                                    scale_sigma_rel=0.0))
    # Аффинная (почти постоянная по проёму) часть H: там разрешение по оси x кадра.
    Ha = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    Ha[:2, :2] = np.eye(2)
    affine = propagate(Inputs(H=Ha, mm_per_unit=4.0, origin_px=(0.0, 0.0), origin_sigma_px=0.0,
                              scale_sigma_rel=0.0,
                              corners_px=(np.array([[0, 0], [100, 0], [100, -50], [0, -50]],
                                                   float),),
                              corner_sigma_px=(click,)))
    gsd = 4.0
    closed_form_localisation = math.sqrt(2.0) * click * gsd
    assert math.sqrt(affine.cov(0, 0)[0, 0]) == pytest.approx(closed_form_localisation
                                                              / math.sqrt(2.0), rel=1e-6)
    # На перспективном кадре та же величина — порядка σ·GSD проёма, не √2·σ·GSD.
    for i, c in enumerate(corners):
        rect = cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 2)
        gsd_x = np.mean([np.linalg.norm(rect[1] - rect[0]) / np.linalg.norm(c[1] - c[0]),
                         np.linalg.norm(rect[2] - rect[3]) / np.linalg.norm(c[2] - c[3])])
        sigma_w = math.sqrt(clicks_only.cov(i, i)[0, 0])
        assert click * gsd_x * 0.7 < sigma_w < click * gsd_x * 1.3, f"элемент {i}"


def test_pipeline_writes_position_sigmas(tmp_path):
    """Через `process`: положение с σ, взаимное — при двух элементах и более."""
    from facade_digitizer.pipeline import run
    from facade_digitizer.pipeline.elements import ElementMark
    from facade_digitizer.pipeline.io import save_image

    sc = make_scene(dx=-1500.0, dy=-3000.0, dist=10000.0)
    path = tmp_path / "f.png"
    save_image(path, sc.render())
    op = sc.openings[0]
    base = sc.project(np.array([[op.x, op.y], [op.x + op.width, op.y]]))
    ref = run.OperatorReference(origin_px=tuple(base[0]),
                                span_px=(tuple(base[0]), tuple(base[1])),
                                span_mm=op.width, sigma_px=1.0)
    first = sc.project(op.corners_mm())
    second = sc.project(op.corners_mm() + np.array([2500.0, 0.0]))
    marks = [ElementMark.model_validate({"class": "window", "mounting": "embedded",
                                         "edge_type": "sharp_wall_edge",
                                         "corners_px": c.tolist(), "sigma_px": 1.0})
             for c in (first, second)]
    model = run.process(path, operator_reference=ref, raster_mm_per_px=10.0, marks=marks)
    a, b = model.elements
    assert a.position_sigma_mm > 0 and b.position_sigma_mm > 0
    # Пара соседей: σ разности одна на двоих.
    assert a.relative_position_sigma_mm == pytest.approx(b.relative_position_sigma_mm)
    single = run.process(path, operator_reference=ref, raster_mm_per_px=10.0,
                         marks=marks[:1])
    assert single.elements[0].relative_position_sigma_mm is None
    assert single.elements[0].position_sigma_mm == pytest.approx(a.position_sigma_mm)
