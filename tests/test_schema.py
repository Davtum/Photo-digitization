import pytest
from pydantic import ValidationError

from facade_digitizer.schema import Element, FacadeModel, Recess, SizeMM


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


def test_schema_version_is_pinned():
    m = FacadeModel.model_validate(_minimal_payload())
    assert m.schema_version == "1.1"


def test_roundtrip_preserves_values():
    m = FacadeModel.model_validate(_minimal_payload())
    again = FacadeModel.model_validate_json(m.model_dump_json())
    assert again.elements[0].size_mm.width == 1460


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
