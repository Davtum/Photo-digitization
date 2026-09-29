"""Выходной формат. Версия схемы 1.2 (спецификация, раздел 10; план 3, задача 6)."""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.2"

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
    value_mm: float | None = None
    sigma_mm: float | None = None
    origin: Literal["measured_from_reveal", "measured_from_shadow", "fused",
                    "assumed_class_default", "operator", "unavailable"]
    datum: Literal["quarter_edge", "frame_plane", "glazing_plane"] | None = None
    reveal_side: Literal["left", "right", "top", "bottom"] | None = None
    theta_perp_deg: float | None = None
    consistency: Literal["ok", "suspect", "unchecked"] = "unchecked"

    @model_validator(mode="after")
    def measured_requires_theta(self):
        measured = {"measured_from_reveal", "fused"}
        if self.origin in measured and self.theta_perp_deg is None:
            raise ValueError("измеренная глубина обязана нести theta_perp_deg")
        return self

    @model_validator(mode="after")
    def unavailable_carries_no_invented_number(self):
        """`origin = "unavailable"` значит «не измерено» — и не может нести число.

        Задача 18 ввела первого потребителя, которому есть что записывать в этот
        `origin`: грань откоса ниже вычисляемого порога применимости (п. 6.2) либо
        размеченная не на той стороне проёма (п. 5.5). До неё `value_mm`/`sigma_mm`
        были обязательными полями `float` БЕЗ исключения для `unavailable`, то есть
        сама схема заставляла бы выдумать число ровно там, где спецификация,
        п. 6.3, это запрещает: «значение без происхождения в выход не попадает»,
        а `unavailable` и есть запись «происхождения нет». `datum` подчиняется
        тому же правилу: он описывает, ДО ЧЕГО измерена глубина, и для
        неизмеренной глубины не имеет смысла.

        Обратное направление проверяется тем же местом: любое ИЗМЕРЕННОЕ
        происхождение обязано нести оба числа и датум — `Recess` без них не
        измерение, а его историческая половина, неотличимая от забытого поля.
        """
        if self.origin == "unavailable":
            if self.value_mm is not None or self.sigma_mm is not None or self.datum is not None:
                raise ValueError(
                    "recess.origin = 'unavailable' не может нести value_mm, sigma_mm "
                    "или datum: неизмеренная глубина не выпускается числом")
        elif self.value_mm is None or self.sigma_mm is None or self.datum is None:
            raise ValueError(
                "recess: value_mm, sigma_mm и datum обязательны при любом происхождении, "
                "кроме 'unavailable'")
        return self


class ThetaDeg(Strict):
    """Угол визирования В ТОЧКЕ ЭЛЕМЕНТА. Спецификация, п. 2.1 и раздел 10.

    Не θ_cam (наклон камеры) и не сводка поля углов кадра — это θ(x, y) этого
    конкретного элемента, тем же путём, каким конвейер считает поле углов
    (`facade_digitizer.pipeline.elements`). Подмена одной величины другой —
    ровно та ошибка, от которой предостерегает п. 2.1.
    """
    x_deg: float
    y_deg: float
    full_deg: float


class ScaleEstimate(Strict):
    """Масштаб фасада и его происхождение (спецификация, п. 6.3).

    `span_sigma_mm` (с версии 1.2) — погрешность САМОЙ измеренной длины опорной
    базы (рулетка, дальномер), в мм. `null` — погрешность длины не передана, и в
    `sigma_rel` входит только промах указания концов базы; до задачи 11 плана 3
    так в каждом файле.
    """

    source: Literal["operator_reference", "photogrammetry", "exif_range",
                    "assumed_floor_height"]
    sigma_rel: float
    meets_tolerance: bool
    span_sigma_mm: float | None = None


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


class OperatorLog(Strict):
    """Трудозатраты оператора на элемент (спецификация, раздел 10). С версии 1.2.

    `id` — непрозрачная метка оператора, назначаемая исследованием, а не имя
    пользователя: она нужна для разбора эффекта обучения (п. 14), имя — нет.
    """

    id: str
    edits: int = Field(ge=0)
    seconds: float = Field(ge=0.0)


class Element(Strict):
    """Элемент фасада. Спецификация, раздел 10.

    **`edge_reference` несёт три значения, а не одно.** Спецификация, п. 5.3
    определяет измеряемую кромку как линию пересечения плоскости стены и
    плоскости откоса и фиксирует это значением `"wall_plane"`. Она же, тремя
    абзацами ниже, называет случаи, когда кромка в плоскости стены НЕ лежит:
    при `edge_type` = `surround` либо `cladding_edge` элемент обрабатывается как
    выступающий (п. 5.6) и требует `offset_mm`; то же верно для
    `mounting = "protruding"`. При `edge_type = "unknown"` про кромку не
    известно ничего. Поле с единственным допустимым значением заставляло бы
    утверждать `"wall_plane"` и там, где п. 5.3 сам объявляет это утверждение
    неверным, — а вынос обрамления даёт смещение 10–45 мм (п. 6.1, строка
    параллакса), то есть больше всего допуска п. 2.2, и «кромка выглядит чёткой,
    оператор её подтверждает».

    Поэтому: `"wall_plane"` — п. 5.3 применима; `"offset_plane"` — кромка
    заведомо лежит вне Π на НЕ ИЗМЕРЕННЫЙ вынос (оценка выноса — следующий
    этап); `"unknown"` — про кромку не известно. Различать `surround` и
    `cladding_edge` этому полю незачем: их несёт `edge_type`, а `edge_reference`
    отвечает на другой вопрос — к чему отнесён `contour_mm` и вправе ли
    потребитель читать его метрически.

    **Поля версии 1.2** (план 3, задача 6). `contour_px[].sigma_px` — точность
    указания точек, с которой элемент посчитан, своя у каждого элемента: разные
    проёмы размечаются при разном увеличении. `position_sigma_mm` и
    `relative_position_sigma_mm` — σ абсолютного положения нижнего левого угла
    контура и его положения относительно ближайшего соседа (п. 2.2 нормирует обе);
    сводятся в одном месте (`pipeline.assemble`, п. 4.4, задача 21 плана 3) с общей
    ошибкой масштаба как коррелированной составляющей. Взаимное — `null` у
    единственного элемента: соседа нет. `operator` —
    трудозатраты (задача 22); `null`, пока хронометража нет.
    """

    # serialize_by_alias — чтобы model_dump_json() без аргументов давал ключ "class",
    # как требует раздел 10 спецификации; populate_by_name — чтобы вход принимался
    # и по алиасу "class", и по имени поля class_name.
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    id: str
    # В JSON поле называется "class" (так в спецификации, раздел 10),
    # но class — ключевое слово Python, поэтому в коде class_name.
    class_name: Literal["window", "door", "reveal", "facade_boundary"] = Field(alias="class")
    mounting: Literal["embedded", "flush", "protruding"]
    edge_reference: Literal["wall_plane", "offset_plane", "unknown"]
    edge_type: Literal["sharp_wall_edge", "surround", "cladding_edge", "unknown"]
    contour_mm: list[Point]
    #: Пиксельные клики оператора, раздел 10: список записей `{image_id, points}`.
    #: Без них измерение невоспроизводимо: `contour_mm` есть результат применения
    #: `H` к этим точкам, и проверить перевод, не имея исходных кликов, нечем.
    #: Задача 18 и есть та, что вводит пиксельные клики, поэтому она же их и пишет.
    contour_px: list[dict] = Field(default_factory=list)
    theta: ThetaDeg | None = None
    size_mm: SizeMM
    origin: Literal["auto", "auto_confirmed", "auto_edited", "operator", "regularized"]
    recess: Recess | None = None
    confidence: float | None = None
    meets_tolerance: bool | None = None
    position_sigma_mm: float | None = None
    relative_position_sigma_mm: float | None = None
    operator: OperatorLog | None = None


class FacadeModel(Strict):
    schema_version: Literal["1.2"] = SCHEMA_VERSION
    software_version: str
    coverage: Literal["full", "partial"]
    mode: Literal["auto", "assisted"]
    images: list[ImageRecord]
    facade: FacadeRecord
    elements: list[Element]
    groups: list[dict] = Field(default_factory=list)


def migrate_1_1_to_1_2(payload: dict) -> dict:
    """Файл версии 1.1 → 1.2. Явная миграция, а не молчаливое чтение.

    Все поля версии 1.2 необязательны и получают `null`: σ на точку разметки,
    σ положения, трудозатраты и погрешность длины базы в файлах 1.1 не записаны, и
    выдумать их нельзя. Исходный словарь не изменяется.

    **Чего миграция не исправляет.** В файлах 1.1 поле `camera.dist` нулевое при
    любом профиле калибровки (дефект, исправленный задачей 2 плана 3): по нему
    нельзя узнать, снималась ли дисторсия. Миграция это поле не трогает — правды о
    нём в файле 1.1 нет.
    """
    version = payload.get("schema_version")
    if version != "1.1":
        raise ValueError(f"мигрируется только версия 1.1, получена {version!r}")
    migrated = json.loads(json.dumps(payload))
    migrated["schema_version"] = "1.2"
    return migrated
