import math

import pytest

from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.parallax import (
    apparent_position,
    correct_for_depth,
    parallax_offset,
    reveal_depth,
    reveal_depth_sigma,
    visible_reveal_side,
)

CAM = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)


def test_point_in_plane_is_not_displaced():
    """Наружная кромка лежит в Π и параллаксом не искажается. Спецификация, п. 5.3."""
    ax, ay = apparent_position(CAM, 5000.0, 3000.0, depth=0.0)
    assert ax == pytest.approx(5000.0, abs=1e-9)
    assert ay == pytest.approx(3000.0, abs=1e-9)


def test_offset_magnitude_equals_d_tan_theta():
    d = 150.0
    dx, dy = parallax_offset(CAM, 5000.0, 3000.0, depth=d)
    expected = d * math.tan(math.radians(CAM.theta_deg(5000.0, 3000.0, depth=d)))
    assert math.hypot(dx, dy) == pytest.approx(expected, rel=1e-9)


def test_offset_points_toward_foot_for_recessed():
    """Заглублённая точка смещается К опорной точке. Ошибка знака в редакции 2 спеки."""
    x, y, d = 5000.0, 3000.0, 150.0
    dx, dy = parallax_offset(CAM, x, y, depth=d)
    assert dx < 0  # опорная точка левее (cx=3236 < 5000)
    assert dy < 0  # опорная точка ниже (cy=-2823 < 3000)


def test_offset_points_away_from_foot_for_protruding():
    dx, dy = parallax_offset(CAM, 5000.0, 3000.0, depth=-150.0)
    assert dx > 0
    assert dy > 0


def test_correction_is_inverse_of_displacement():
    x, y, d = 4200.0, 5100.0, 220.0
    ax, ay = apparent_position(CAM, x, y, d)
    bx, by = correct_for_depth(CAM, ax, ay, d)
    assert bx == pytest.approx(x, abs=1e-9)
    assert by == pytest.approx(y, abs=1e-9)


def test_near_side_inner_edge_is_occluded():
    """Внутреннее ребро на ближней стороне проецируется за стену. Спецификация, п. 5.5."""
    cam = CameraOnPlane(cx=1000.0, cy=3000.0, cz=10000.0)
    x_left = 3500.0
    ax, _ = apparent_position(cam, x_left, 3000.0, depth=150.0)
    assert ax < x_left


@pytest.mark.parametrize("depth,theta_deg,expected_mm", [
    (150.0, 5.0, 13.1), (150.0, 15.0, 40.2), (150.0, 30.0, 86.6), (150.0, 45.0, 150.0),
    (300.0, 15.0, 80.4), (300.0, 30.0, 173.2),
])
def test_table_5_2_of_spec(depth, theta_deg, expected_mm):
    """Таблица смещений п. 5.2 проверяется ПРОТИВ РЕАЛИЗАЦИИ, а не против самой себя."""
    cz = 10000.0
    x_edge, y_edge = 5000.0, 3000.0
    cam = CameraOnPlane(
        cx=x_edge - (cz + depth) * math.tan(math.radians(theta_deg)),
        cy=y_edge,
        cz=cz,
    )
    dx, dy = parallax_offset(cam, x_edge, y_edge, depth=depth)
    assert math.hypot(dx, dy) == pytest.approx(expected_mm, abs=0.1)


def test_apparent_position_rejects_point_behind_camera():
    """Охрана согласована с camera.tan_theta: без неё деление на ноль либо неверный знак."""
    with pytest.raises(ValueError):
        apparent_position(CAM, 5000.0, 3000.0, depth=-CAM.cz)
    with pytest.raises(ValueError):
        apparent_position(CAM, 5000.0, 3000.0, depth=-CAM.cz - 5000.0)


def test_correction_matches_independently_computed_value():
    """Ожидаемые значения посчитаны вручную по п. 5.2 и заморожены литералами.

    Формула в теле теста повторяла бы формулу реализации и ловила бы только
    рассогласование копий, а не ошибку в самой формуле.
    """
    d = 220.0
    observed = (4150.0, 4900.0)
    got = correct_for_depth(CAM, observed[0], observed[1], d)
    assert got[0] == pytest.approx(4170.108, abs=1e-3)
    assert got[1] == pytest.approx(5069.906, abs=1e-3)


def test_correct_for_depth_rejects_point_behind_camera():
    """Область определения обратной функции обязана совпадать с прямой."""
    with pytest.raises(ValueError):
        correct_for_depth(CAM, 5000.0, 3000.0, depth=-CAM.cz)
    with pytest.raises(ValueError):
        correct_for_depth(CAM, 5000.0, 3000.0, depth=-CAM.cz - 5000.0)


def test_reveal_depth_recovers_known_depth():
    """Прямая проверка: породить ширину откоса из известной глубины и вернуть её."""
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    x_edge, y_edge, d_true = 5000.0, 3000.0, 150.0
    dx, _ = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    w = abs(dx)  # ширина ВЕРТИКАЛЬНОГО откоса — только X-компонента
    got = reveal_depth(cam, x_edge, y_edge, w, edge_normal=(1.0, 0.0))
    assert got == pytest.approx(d_true, rel=1e-9)


def test_horizontal_reveal_uses_y_component():
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    x_edge, y_edge, d_true = 5000.0, 3000.0, 220.0
    _, dy = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    got = reveal_depth(cam, x_edge, y_edge, abs(dy), edge_normal=(0.0, 1.0))
    assert got == pytest.approx(d_true, rel=1e-9)


def test_reveal_depth_is_invariant_to_edge_normal_length():
    """Нормаль задаёт направление, её длина на результат влиять не должна.

    Без нормировки edge_normal проекция u масштабируется вместе с нормалью,
    и глубина искажается во столько же раз.
    """
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    x_edge, y_edge, d_true = 5000.0, 3000.0, 150.0
    dx, _ = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    got = reveal_depth(cam, x_edge, y_edge, abs(dx), edge_normal=(3.0, 0.0))
    assert got == pytest.approx(d_true, rel=1e-9)


def test_reveal_depth_rejects_zero_normal():
    """Нулевая нормаль не задаёт направления ребра."""
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    with pytest.raises(ValueError):
        reveal_depth(cam, 5000.0, 3000.0, 26.0, edge_normal=(0.0, 0.0))


@pytest.mark.parametrize("theta_x_deg,theta_y_deg,ratio", [
    (10.0, 30.0, 3.45), (5.0, 25.0, 5.43), (15.0, 15.0, 1.41), (10.0, 0.0, 1.00),
])
def test_scalar_formula_underestimates(theta_x_deg, theta_y_deg, ratio):
    """Скалярная запись по полному углу занижает глубину. Спецификация, п. 5.4.

    Тест ВЫЗЫВАЕТ реализацию: считает ширину откоса из известной глубины, затем
    восстанавливает глубину правильной компонентной формулой и ошибочной скалярной,
    и сравнивает. В редакции 1 плана этот тест считал арифметику над собственными
    входами и проходил при полностью удалённом модуле.
    """
    cz, d_true = 10000.0, 150.0
    x_edge, y_edge = 5000.0, 3000.0
    cam = CameraOnPlane(
        cx=x_edge - (cz + d_true) * math.tan(math.radians(theta_x_deg)),
        cy=y_edge - (cz + d_true) * math.tan(math.radians(theta_y_deg)),
        cz=cz,
    )
    dx, dy = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    w = abs(dx)

    correct = reveal_depth(cam, x_edge, y_edge, w, edge_normal=(1.0, 0.0))
    assert correct == pytest.approx(d_true, rel=1e-6)

    # Скалярная запись: полный угол вместо компоненты.
    tx, ty = cam.tan_theta(x_edge, y_edge, depth=d_true)
    scalar = w / math.hypot(tx, ty)
    assert correct / scalar == pytest.approx(ratio, rel=0.02)


def test_reveal_depth_rejects_degenerate_geometry():
    """Ширина откоса не может превышать расстояние до опорной точки."""
    cam = CameraOnPlane(cx=4990.0, cy=3000.0, cz=10000.0)
    with pytest.raises(ValueError):
        reveal_depth(cam, 5000.0, 3000.0, reveal_width_mm=50.0, edge_normal=(1.0, 0.0))


def test_sigma_grows_as_angle_shrinks():
    """σ_d складывается из ошибки ширины и ошибки позы. Спецификация, п. 6.2."""
    sigma_theta = math.radians(1.0)
    wide = reveal_depth_sigma(150.0, 86.6, 5.0, math.tan(math.radians(30.0)), sigma_theta)
    narrow = reveal_depth_sigma(150.0, 13.1, 5.0, math.tan(math.radians(5.0)), sigma_theta)
    assert narrow > 3 * wide


def test_sigma_includes_pose_contribution():
    """Вклад погрешности позы обязан входить в σ_d наряду с вкладом ширины.

    При σ_θ = 0 остаётся только вклад ширины; при σ_θ > 0 результат строго больше.
    Без сложения вкладов оба значения совпали бы.
    """
    tan_perp = math.tan(math.radians(30.0))
    without_pose = reveal_depth_sigma(150.0, 86.6, 5.0, tan_perp, 0.0)
    with_pose = reveal_depth_sigma(150.0, 86.6, 5.0, tan_perp, math.radians(1.0))
    assert without_pose == pytest.approx(5.0 / tan_perp, rel=1e-12)
    assert with_pose > without_pose


def test_visible_reveal_side_is_far_side():
    """Виден откос на ДАЛЬНЕЙ от опорной точки стороне. Спецификация, п. 5.5."""
    cam = CameraOnPlane(cx=1000.0, cy=1000.0, cz=10000.0)
    vertical, horizontal = visible_reveal_side(cam, 3500.0, 5000.0, 3000.0, 5000.0)
    assert vertical == "right"
    assert horizontal == "top"

    cam2 = CameraOnPlane(cx=9000.0, cy=9000.0, cz=10000.0)
    vertical2, horizontal2 = visible_reveal_side(cam2, 3500.0, 5000.0, 3000.0, 5000.0)
    assert vertical2 == "left"
    assert horizontal2 == "bottom"
