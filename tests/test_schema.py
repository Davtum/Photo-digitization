import pytest
from pydantic import ValidationError

from facade_digitizer.schema import FacadeModel, Recess, SizeMM, ThetaDeg


def test_element_requires_uncertainty_with_size():
    """Габарит без σ в схему не принимается — глобальный инвариант спецификации."""
    with pytest.raises(ValidationError):
        SizeMM(width=1460, height=1900)  # нет sigma_width / sigma_height


def test_recess_measured_requires_theta():
    """Измеренная глубина обязана нести угол, по которому она получена."""
    with pytest.raises(ValidationError):
        Recess(value_mm=168, sigma_mm=17, origin="measured_from_reveal", datum="quarter_edge")


def test_recess_assumed_does_not_require_theta():
    r = Recess(value_mm=150, sigma_mm=50, origin="assumed_class_default", datum="quarter_edge")
    assert r.theta_perp_deg is None


def test_recess_unavailable_carries_no_value():
    """`origin = "unavailable"` — данных нет, и это допустимо без value_mm/sigma_mm.

    Задача 18: грань откоса разметили не на той стороне либо ракурс ниже порога
    применимости. До этой задачи `value_mm`/`sigma_mm` были обязательными `float`
    без исключения — сборка была бы вынуждена выдумать число ровно там, где
    спецификация, п. 6.3, это запрещает.
    """
    r = Recess(origin="unavailable", reveal_side="right")
    assert r.value_mm is None
    assert r.sigma_mm is None
    assert r.datum is None


def test_recess_unavailable_rejects_an_invented_value():
    """Обратная охрана: `unavailable` не может НЕСТИ число — иначе оно не отличимо
    от настоящего измерения."""
    with pytest.raises(ValidationError):
        Recess(value_mm=150.0, sigma_mm=17.0, origin="unavailable")
    with pytest.raises(ValidationError):
        Recess(datum="quarter_edge", origin="unavailable")


def test_recess_measured_origin_requires_both_numbers_and_datum():
    """Любое измеренное происхождение обязано нести value_mm, sigma_mm И datum —
    `Recess` без одного из них есть половина измерения, а не измерение."""
    with pytest.raises(ValidationError):
        Recess(sigma_mm=17.0, origin="operator", datum="frame_plane")
    with pytest.raises(ValidationError):
        Recess(value_mm=150.0, origin="operator", datum="frame_plane")
    with pytest.raises(ValidationError):
        Recess(value_mm=150.0, sigma_mm=17.0, origin="operator")


def test_element_theta_is_optional_but_validated_when_present():
    """`theta` — новое поле задачи 18 (п. 2.1, п. 10): не θ_cam, угол ЭЛЕМЕНТА."""
    payload = _minimal_payload()
    payload["elements"][0]["theta"] = {"x_deg": 21.4, "y_deg": 12.8, "full_deg": 24.7}
    m = FacadeModel.model_validate(payload)
    assert m.elements[0].theta == ThetaDeg(x_deg=21.4, y_deg=12.8, full_deg=24.7)

    # Без поля — элемент по-прежнему валиден (обратная совместимость, задача 18).
    m2 = FacadeModel.model_validate(_minimal_payload())
    assert m2.elements[0].theta is None


def test_schema_version_is_pinned():
    m = FacadeModel.model_validate(_minimal_payload())
    assert m.schema_version == "1.1"


def test_roundtrip_preserves_values():
    m = FacadeModel.model_validate(_minimal_payload())
    again = FacadeModel.model_validate_json(m.model_dump_json())
    assert again.elements[0].size_mm.width == 1460


def test_serialised_json_uses_spec_field_name():
    """Раздел 10 спецификации требует ключ "class"; class — ключевое слово Python."""
    payload = FacadeModel.model_validate(_minimal_payload()).model_dump_json()
    assert '"class"' in payload
    assert "class_name" not in payload


def _minimal_payload() -> dict:
    return {
        "schema_version": "1.1",
        "software_version": "facade-digitizer 0.1.0",
        "coverage": "partial",
        "mode": "auto",
        "images": [],
        "facade": {
            "origin": "bottom_left",
            "bounds_mm": [0, 0, 20800, 16200],
            "mm_per_rectified_px": 3.12,
            "scale": {"source": "operator_reference", "sigma_rel": 0.0022,
                      "meets_tolerance": True},
        },
        "elements": [{
            "id": "w_001",
            "class": "window",
            "mounting": "embedded",
            "edge_reference": "wall_plane",
            "edge_type": "sharp_wall_edge",
            "contour_mm": [[0, 0], [1460, 0], [1460, 1900], [0, 1900]],
            "size_mm": {"width": 1460, "height": 1900,
                        "sigma_width": 9, "sigma_height": 11},
            "origin": "auto",
        }],
        "groups": [],
    }
