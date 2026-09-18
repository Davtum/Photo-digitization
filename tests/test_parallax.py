import math

import pytest

from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.parallax import (
    apparent_position,
    correct_for_depth,
    parallax_offset,
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
    """Ожидаемое выведено из постановки п. 5.2, а не обратимостью к apparent_position."""
    d = 220.0
    observed = (4150.0, 4900.0)
    k = (CAM.cz + d) / CAM.cz
    expected = (CAM.cx + k * (observed[0] - CAM.cx), CAM.cy + k * (observed[1] - CAM.cy))
    got = correct_for_depth(CAM, observed[0], observed[1], d)
    assert got[0] == pytest.approx(expected[0])
    assert got[1] == pytest.approx(expected[1])
