"""Выходной формат. Версия схемы 1.1 (спецификация, раздел 10)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.1"

Point = tuple[float, float]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SizeMM(Strict):
    """Габарит всегда сопровождается погрешностью — инвариант спецификации, п. 6.3."""
    width: float
    height: float
    sigma_width: float
    sigma_height: float


class Recess(Strict):
    value_mm: float
    sigma_mm: float
    origin: Literal["measured_from_reveal", "measured_from_shadow", "fused",
                    "assumed_class_default", "operator", "unavailable"]
    datum: Literal["quarter_edge", "frame_plane", "glazing_plane"]
    reveal_side: Literal["left", "right", "top", "bottom"] | None = None
    theta_perp_deg: float | None = None
    consistency: Literal["ok", "suspect", "unchecked"] = "unchecked"

    @model_validator(mode="after")
    def measured_requires_theta(self):
        measured = {"measured_from_reveal", "fused"}
        if self.origin in measured and self.theta_perp_deg is None:
            raise ValueError("измеренная глубина обязана нести theta_perp_deg")
        return self


class ScaleEstimate(Strict):
    source: Literal["operator_reference", "photogrammetry", "exif_range",
                    "assumed_floor_height"]
    sigma_rel: float
    meets_tolerance: bool


class QualityReport(Strict):
    verdict: Literal["ok", "degraded", "reject"]
    sharpness: float
    gsd_mm_px_min: float
    gsd_mm_px_max: float
    theta_field_deg_p95: float
    reasons: list[str] = Field(default_factory=list)


class Rectification(Strict):
    """Приведение к плоскости фасада и его происхождение.

    `needs_operator` — тот же признак, что и у `PlaneResult`: ниже порога доверия
    система не гадает, а передаёт управление оператору (спецификация, п. 4.2).
    Без него отказ невозможно выразить данными: `Strict` запрещает лишние поля, и
    потребитель не отличил бы неразрешённую плоскость от разрешённой.

    `residual_px` допускает отсутствие. Невязка отрезков относительно точек схода
    определена не всегда — ручной путь по четырём точкам её не измеряет вовсе, а
    отказ по нехватке отрезков возвращает бесконечность. Аннотация `float`
    заставила бы подставить вместо отсутствия число, и оно стало бы неотличимо от
    измеренного.
    """

    method: Literal["vanishing_points", "manual_four_point"]
    confidence: float
    residual_px: float | None
    needs_operator: bool = False


class CameraIntrinsics(Strict):
    model: str
    K: list[list[float]]
    dist: list[float]
    calibration: Literal["target", "exif", "database"]


class ImageRecord(Strict):
    id: str
    path: str
    camera: CameraIntrinsics
    captured_at: str | None = None
    gnss: dict | None = None
    pose_to_facade: dict | None = None
    theta_cam_deg: float | None = None
    theta_field_deg: dict | None = None
    quality: QualityReport | None = None
    rectification: Rectification | None = None
    homography: dict | None = None


class FacadeRecord(Strict):
    origin: Literal["bottom_left", "operator_reference"]
    bounds_mm: list[float]
    mm_per_rectified_px: float
    scale: ScaleEstimate
    plane_residual_mm: dict | None = None
    valid_mask: str | None = None


class Element(Strict):
    # serialize_by_alias — чтобы model_dump_json() без аргументов давал ключ "class",
    # как требует раздел 10 спецификации; populate_by_name — чтобы вход принимался
    # и по алиасу "class", и по имени поля class_name.
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    id: str
    # В JSON поле называется "class" (так в спецификации, раздел 10),
    # но class — ключевое слово Python, поэтому в коде class_name.
    class_name: Literal["window", "door", "reveal", "facade_boundary"] = Field(alias="class")
    mounting: Literal["embedded", "flush", "protruding"]
    edge_reference: Literal["wall_plane"]
    edge_type: Literal["sharp_wall_edge", "surround", "cladding_edge", "unknown"]
    contour_mm: list[Point]
    size_mm: SizeMM
    origin: Literal["auto", "auto_confirmed", "auto_edited", "operator", "regularized"]
    recess: Recess | None = None
    confidence: float | None = None
    meets_tolerance: bool | None = None


class FacadeModel(Strict):
    schema_version: Literal["1.1"] = SCHEMA_VERSION
    software_version: str
    coverage: Literal["full", "partial"]
    mode: Literal["auto", "assisted"]
    images: list[ImageRecord]
    facade: FacadeRecord
    elements: list[Element]
    groups: list[dict] = Field(default_factory=list)
