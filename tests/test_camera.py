import pytest

from facade_digitizer.geometry.camera import CameraOnPlane


def test_foot_is_normal_projection_of_camera():
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    assert cam.foot() == (3236.0, -2823.0)


def test_tan_theta_at_foot_is_zero():
    cam = CameraOnPlane(cx=1000.0, cy=2000.0, cz=10000.0)
    assert cam.tan_theta(1000.0, 2000.0) == (0.0, 0.0)


def test_tan_theta_components_match_geometry():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    tx, ty = cam.tan_theta(10000.0, 0.0)
    assert tx == pytest.approx(1.0)
    assert ty == pytest.approx(0.0)


def test_tan_theta_uses_depth_in_denominator():
    """Луч на заглублённую точку идёт дальше, угол меньше."""
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    shallow, _ = cam.tan_theta(1000.0, 0.0, depth=0.0)
    deep, _ = cam.tan_theta(1000.0, 0.0, depth=500.0)
    assert deep < shallow


def test_theta_deg_is_full_angle():
    """Полный угол при равных смещениях по обеим осям равен arctg √2 = 54.7356°.

    Ожидание — ЛИТЕРАЛ, а не та же формула, какой считает реализация. Прежде
    здесь стояло `math.degrees(math.atan(math.hypot(1.0, 1.0)))`, то есть тест
    подтверждал лишь то, что формулу удалось переписать: подмена `hypot` на
    что-либо другое сместила бы обе стороны равенства одинаково.
    """
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    got = cam.theta_deg(10000.0, 10000.0)
    assert got == pytest.approx(54.735610317245346, abs=1e-9)
    # И не 45°: полный угол не равен ни одной из своих компонент.
    assert got > 45.0 + 1.0


def test_camera_must_be_in_front_of_plane():
    with pytest.raises(ValueError):
        CameraOnPlane(cx=0.0, cy=0.0, cz=0.0)


def test_tan_theta_rejects_point_behind_camera():
    """cz + depth <= 0 — точка за камерой, направление луча не определено."""
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    with pytest.raises(ValueError):
        cam.tan_theta(1000.0, 0.0, depth=-10000.0)
    with pytest.raises(ValueError):
        cam.tan_theta(1000.0, 0.0, depth=-15000.0)


def test_offset_camera_and_depth_combine_correctly():
    """Рабочий случай всех последующих модулей: камера смещена И точка заглублена."""
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    tx, ty = cam.tan_theta(5000.0, 3000.0, depth=150.0)
    assert tx == pytest.approx((5000.0 - 3236.0) / 10150.0)
    assert ty == pytest.approx((3000.0 + 2823.0) / 10150.0)
