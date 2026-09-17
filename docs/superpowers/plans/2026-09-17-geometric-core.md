# План 1: Геометрическое ядро — реализация

> **Для агентов-исполнителей:** ОБЯЗАТЕЛЬНЫЙ ПОДНАВЫК — используйте
> superpowers:subagent-driven-development (рекомендуется) либо
> superpowers:executing-plans для исполнения по одной задаче за раз. Шаги размечены
> чекбоксами (`- [ ]`) для отслеживания.

**Цель:** построить конвейер, который принимает одиночный фотоснимок фасада и выдаёт
ректифицированное изображение, позу камеры относительно плоскости фасада, поле углов визирования
и JSON по схеме версии 1.1 — с явной оценкой доверия на каждом шаге.

**Архитектура:** чистое геометрическое ядро (математика параллакса и глубины) отделено от
конвейера обработки изображения; всё, что касается плоскости фасада, выражено в миллиметрах
в системе координат фасада. Каждый модуль конвейера возвращает не только результат, но и меру
доверия к нему; при недостаточном доверии управление передаётся ручному варианту. Тесты
опираются на синтетический генератор сцен с известной наперёд геометрией, поэтому полевые данные
для этого плана не нужны.

**Стек:** Python 3.11+, OpenCV ≥ 4.8, NumPy, SciPy, Pydantic v2, pytest, ruff.

**Спецификация:** `docs/superpowers/specs/2026-09-17-facade-photo-digitization-design.md`
(редакция 3). План аргументирует от спецификации; исполнителю нужно читать оба документа.

## Глобальные ограничения

Действуют для каждой задачи плана, повторно в задачах не упоминаются.

- **Единицы — миллиметры.** Система координат фасада: начало в левом нижнем углу, ось X вправо
  вдоль фасада, ось Y вверх. Плоскость фасада Π: Z = 0, камера при Z > 0.
- **Измеряемая кромка проёма** — линия пересечения плоскости стены и плоскости откоса
  (`edge_reference: "wall_plane"`). Спецификация, п. 5.3.
- **Углы разведены:** `theta_cam` — наклон камеры к нормали фасада; `theta_x`, `theta_y` —
  компоненты угла луча на конкретную точку. Никогда не смешивать. Спецификация, п. 2.1.
- **Версия схемы данных — `"1.1"`.** Любое изменение схемы требует повышения версии.
- **Источник и погрешность всегда сопровождают величину.** Значение без σ и без указания
  происхождения в выход не попадает. Спецификация, п. 6.3.
- **OpenCV ≥ 4.8.0** — в более ранних версиях отсутствует `cv2.createLineSegmentDetector`
  (детектор был изъят из-за патента и возвращён в 4.8.0).
- **Pydantic v2** — синтаксис v1 несовместим.
- **Производительность:** конвейер `io` … `rectify` на снимке 20 Мп — не более 15 с на рабочей
  станции без GPU. Спецификация, раздел 16.
- **Коммит после каждой задачи**, сообщение на русском, в повелительном наклонении не писать —
  описывать сделанное.

---

## Структура файлов

```
facade_digitizer/
  __init__.py
  schema.py                  # Pydantic-модели выходного формата (версия 1.1)
  metrics.py                 # RMSE, P95, смещение, покрытие, острота, interval score
  geometry/
    __init__.py
    camera.py                # CameraOnPlane: поза камеры относительно Π
    parallax.py              # смещение, коррекция, глубина по откосу
    vanishing.py             # отрезки, группировка, точки схода, доверие
    homography.py            # H и поза из точек схода и K; ручной вариант
    angles.py                # поле θ_x, θ_y; локальный GSD
  pipeline/
    __init__.py
    io.py                    # загрузка, EXIF
    calib.py                 # дисторсия, внутренние параметры
    quality.py               # вердикт о пригодности снимка
    plane.py                 # оркестрация: отрезки → VP → H → поза
    rectify.py               # применение H, valid_mask, карты GSD и углов
    run.py                   # сборка конвейера, CLI
  synth/
    __init__.py
    scene.py                 # генератор синтетических сцен с известной геометрией
tests/
  test_camera.py
  test_parallax.py
  test_metrics.py
  test_schema.py
  test_synth.py
  test_vanishing.py
  test_homography.py
  test_angles.py
  test_calib.py
  test_quality.py
  test_rectify.py
  test_e2e.py
pyproject.toml
```

Разделение по ответственности, а не по слою: `geometry/` — чистая математика без ввода-вывода и
без OpenCV-зависимостей от изображений, полностью покрывается тестами на аналитически известных
случаях; `pipeline/` — работа с изображением; `synth/` — генератор данных для тестов.

---

## Задача 1: Скелет проекта и схема данных

**Файлы:**
- Создать: `pyproject.toml`
- Создать: `facade_digitizer/__init__.py`
- Создать: `facade_digitizer/schema.py`
- Тест: `tests/test_schema.py`

**Интерфейсы:**
- Потребляет: ничего
- Предоставляет: `FacadeModel`, `ImageRecord`, `FacadeRecord`, `Element`, `Recess`, `ScaleEstimate`,
  `QualityReport`, `SizeMM` — Pydantic-модели; `FacadeModel.model_validate_json(str)` и
  `FacadeModel.model_dump_json(indent=2)`

- [ ] **Шаг 1: Создать `pyproject.toml`**

```toml
[project]
name = "facade-digitizer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "opencv-python>=4.8.0",
    "pydantic>=2.6",
    "piexif>=1.1.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[project.scripts]
facade-digitize = "facade_digitizer.pipeline.run:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Шаг 2: Написать падающий тест схемы**

Файл `tests/test_schema.py`:

```python
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
```

- [ ] **Шаг 3: Убедиться, что тест падает**

Выполнить: `pytest tests/test_schema.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.schema'`

- [ ] **Шаг 4: Реализовать схему**

Файл `facade_digitizer/schema.py`:

```python
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
    method: Literal["vanishing_points", "manual_four_point"]
    confidence: float
    residual_px: float


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
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

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
```

- [ ] **Шаг 5: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_schema.py -v`
Ожидается: 5 passed

- [ ] **Шаг 6: Коммит**

```bash
git add pyproject.toml facade_digitizer/__init__.py facade_digitizer/schema.py tests/test_schema.py
git commit -m "Схема выходных данных версии 1.1 с обязательной оценкой погрешности"
```

---

## Задача 2: Поза камеры относительно плоскости фасада

**Файлы:**
- Создать: `facade_digitizer/geometry/__init__.py`
- Создать: `facade_digitizer/geometry/camera.py`
- Тест: `tests/test_camera.py`

**Интерфейсы:**
- Потребляет: ничего
- Предоставляет: `CameraOnPlane(cx: float, cy: float, cz: float)` с методами
  `foot() -> tuple[float, float]`, `tan_theta(x, y, depth=0.0) -> tuple[float, float]`,
  `theta_deg(x, y, depth=0.0) -> float`

Отдельный модуль потому, что поза камеры — общий аргумент для параллакса, поля углов и локального
GSD; втягивать её в каждый из них означало бы дублировать инвариант «камера перед плоскостью».

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_camera.py`:

```python
import math

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
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    got = cam.theta_deg(10000.0, 10000.0)
    expected = math.degrees(math.atan(math.hypot(1.0, 1.0)))
    assert got == pytest.approx(expected)


def test_camera_must_be_in_front_of_plane():
    with pytest.raises(ValueError):
        CameraOnPlane(cx=0.0, cy=0.0, cz=0.0)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_camera.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.geometry'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/geometry/__init__.py` — пустой.

Файл `facade_digitizer/geometry/camera.py`:

```python
"""Поза камеры относительно плоскости фасада.

Плоскость фасада Π: Z = 0. Камера в точке (cx, cy, cz), cz > 0.
Опорная точка F = (cx, cy) — проекция камеры на Π по нормали.
Спецификация, п. 5.1.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraOnPlane:
    cx: float
    cy: float
    cz: float

    def __post_init__(self) -> None:
        if self.cz <= 0:
            raise ValueError("камера должна находиться перед плоскостью фасада: cz > 0")

    def foot(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def tan_theta(self, x: float, y: float, depth: float = 0.0) -> tuple[float, float]:
        """Компоненты (tan θ_x, tan θ_y) луча на точку (x, y) при глубине depth.

        depth > 0 — точка заглублена за плоскость, depth < 0 — вынесена вперёд.
        """
        denom = self.cz + depth
        if denom <= 0:
            raise ValueError("точка оказалась за камерой: cz + depth <= 0")
        return ((x - self.cx) / denom, (y - self.cy) / denom)

    def theta_deg(self, x: float, y: float, depth: float = 0.0) -> float:
        tx, ty = self.tan_theta(x, y, depth)
        return math.degrees(math.atan(math.hypot(tx, ty)))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_camera.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/ tests/test_camera.py
git commit -m "Поза камеры относительно плоскости фасада с компонентами угла визирования"
```

---

## Задача 3: Ядро параллакса — смещение и коррекция

**Файлы:**
- Создать: `facade_digitizer/geometry/parallax.py`
- Тест: `tests/test_parallax.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane` из задачи 2
- Предоставляет:
  `apparent_position(cam, x, y, depth) -> tuple[float, float]`,
  `parallax_offset(cam, x, y, depth) -> tuple[float, float]`,
  `correct_for_depth(cam, xa, ya, depth) -> tuple[float, float]`

Это те самые регрессионные тесты, которые спецификация (п. 5.9) требует внести в этап 0:
знак смещения, двухосевой случай, окклюзия. В редакции 2 спецификации знак был указан неверно —
тест `test_offset_points_toward_foot_for_recessed` существует именно затем, чтобы эта ошибка не
могла вернуться.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_parallax.py`:

```python
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
    """Таблица смещений из п. 5.2 спецификации."""
    assert depth * math.tan(math.radians(theta_deg)) == pytest.approx(expected_mm, abs=0.1)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_parallax.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.geometry.parallax'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/geometry/parallax.py`:

```python
"""Параллакс точек, не лежащих в плоскости фасада. Спецификация, раздел 5.

Точка на глубине d за плоскостью после ректификации оказывается смещённой:

    δ = (C_xy − P_xy) · d / (C_z + d)

Для заглублённых точек (d > 0) смещение направлено К опорной точке F,
для выступающих (d < 0) — ОТ неё.
"""
from .camera import CameraOnPlane


def apparent_position(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Где точка (x, y, −depth) окажется после ректификации."""
    k = cam.cz / (cam.cz + depth)
    return (cam.cx + k * (x - cam.cx), cam.cy + k * (y - cam.cy))


def parallax_offset(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Вектор смещения δ = P′ − P_xy."""
    ax, ay = apparent_position(cam, x, y, depth)
    return (ax - x, ay - y)


def correct_for_depth(
    cam: CameraOnPlane, xa: float, ya: float, depth: float
) -> tuple[float, float]:
    """Обратная операция: из наблюдаемого положения в истинное."""
    k = (cam.cz + depth) / cam.cz
    return (cam.cx + k * (xa - cam.cx), cam.cy + k * (ya - cam.cy))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_parallax.py -v`
Ожидается: 12 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/parallax.py tests/test_parallax.py
git commit -m "Ядро параллакса: смещение со знаком, коррекция, регрессионные тесты раздела 5"
```

---

## Задача 4: Оценка глубины по видимой грани откоса

**Файлы:**
- Изменить: `facade_digitizer/geometry/parallax.py`
- Изменить: `tests/test_parallax.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane`, функции задачи 3
- Предоставляет:
  `reveal_depth(cam, edge_x, edge_y, reveal_width_mm, edge_normal) -> float`,
  `reveal_depth_sigma(depth, reveal_width_mm, sigma_width_mm, tan_perp, sigma_theta_rad) -> float`,
  `visible_reveal_side(cam, x_left, x_right, y_bottom, y_top) -> tuple[str, str]`

**Замечание исполнителю: здесь план точнее спецификации.** Спецификация (п. 5.4) предписывает
брать угол «на внутреннем ребре», что делает формулу неявной — угол зависит от глубины, которую
и требуется найти. Уравнение решается в замкнутом виде. Пусть `u` — проекция вектора от опорной
точки к кромке на нормаль к ребру откоса. Тогда

```
w = d · |u| / (C_z + d)   ⟹   d = w · C_z / (|u| − w)
```

Это точное решение без итераций и без выбора, где брать угол. После исполнения задачи внести
уточнение в п. 5.4 спецификации.

- [ ] **Шаг 1: Дописать падающие тесты**

Добавить в `tests/test_parallax.py`:

```python
from facade_digitizer.geometry.parallax import (
    reveal_depth,
    reveal_depth_sigma,
    visible_reveal_side,
)


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


@pytest.mark.parametrize("theta_x,theta_y,ratio", [
    (10.0, 30.0, 3.45), (5.0, 25.0, 5.43), (15.0, 15.0, 1.41), (10.0, 0.0, 1.00),
])
def test_scalar_formula_underestimates(theta_x, theta_y, ratio):
    """Скалярная формула редакции 2 занижает глубину. Спецификация, п. 5.4."""
    tx, ty = math.tan(math.radians(theta_x)), math.tan(math.radians(theta_y))
    scalar = math.hypot(tx, ty)
    assert scalar / tx == pytest.approx(ratio, rel=0.02)


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
```

- [ ] **Шаг 2: Убедиться, что новые тесты падают**

Выполнить: `pytest tests/test_parallax.py -v`
Ожидается: `ImportError: cannot import name 'reveal_depth'`

- [ ] **Шаг 3: Реализовать**

Дописать в `facade_digitizer/geometry/parallax.py`:

```python
import math


def reveal_depth(
    cam: CameraOnPlane,
    edge_x: float,
    edge_y: float,
    reveal_width_mm: float,
    edge_normal: tuple[float, float],
) -> float:
    """Глубина заглубления по видимой ширине грани откоса.

    Ширина откоса, измеренная ПОПЕРЁК его ребра, равна не полному смещению, а его
    компоненте вдоль нормали к ребру. Замкнутое решение:

        w = d · |u| / (C_z + d)   ⟹   d = w · C_z / (|u| − w)

    где u — проекция вектора от опорной точки к кромке на edge_normal.
    Спецификация, п. 5.4.
    """
    nx, ny = edge_normal
    norm = math.hypot(nx, ny)
    if norm == 0:
        raise ValueError("нормаль к ребру откоса не может быть нулевой")
    nx, ny = nx / norm, ny / norm

    u = abs((edge_x - cam.cx) * nx + (edge_y - cam.cy) * ny)
    if reveal_width_mm <= 0:
        raise ValueError("ширина откоса должна быть положительной")
    if u <= reveal_width_mm:
        raise ValueError(
            "вырожденная геометрия: ширина откоса не меньше расстояния до опорной точки; "
            "угол визирования слишком мал либо ширина измерена неверно"
        )
    return reveal_width_mm * cam.cz / (u - reveal_width_mm)


def reveal_depth_sigma(
    depth_mm: float,
    reveal_width_mm: float,
    sigma_width_mm: float,
    tan_perp: float,
    sigma_theta_rad: float,
) -> float:
    """σ глубины: вклад ошибки ширины и вклад ошибки позы.

    От ширины:  σ_d = σ_w / |t·n̂|
    От позы:    σ_d / d = 2 σ_θ / sin 2θ
    Спецификация, п. 6.2.
    """
    if tan_perp <= 0:
        raise ValueError("компонента угла должна быть положительной")
    from_width = sigma_width_mm / tan_perp

    theta = math.atan(tan_perp)
    from_pose = depth_mm * 2.0 * sigma_theta_rad / math.sin(2.0 * theta)

    return math.hypot(from_width, from_pose)


def visible_reveal_side(
    cam: CameraOnPlane,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> tuple[str, str]:
    """Какие грани откоса видны: дальние от опорной точки.

    Возвращает (вертикальная, горизонтальная). Если опорная точка попадает внутрь
    диапазона проёма, соответствующая грань называется по знаку, но её угол мал —
    вызывающий код обязан проверить порог θ_min.
    Спецификация, п. 5.5.
    """
    vertical = "right" if cam.cx < (x_left + x_right) / 2 else "left"
    horizontal = "top" if cam.cy < (y_bottom + y_top) / 2 else "bottom"
    return (vertical, horizontal)
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_parallax.py -v`
Ожидается: 21 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/parallax.py tests/test_parallax.py
git commit -m "Оценка глубины по грани откоса в замкнутой форме с бюджетом погрешности"
```

- [ ] **Шаг 6: Внести уточнение в спецификацию**

В п. 5.4 заменить формулировку «θ вычисляется на внутреннем ребре» на замкнутое решение
`d = w · C_z / (|u| − w)`, отметив, что оно снимает неявность. Коммит:

```bash
git add docs/superpowers/specs/2026-09-17-facade-photo-digitization-design.md
git commit -m "Уточнён п. 5.4: замкнутая форма оценки глубины вместо неявной"
```

---

## Задача 5: Синтетический генератор сцен

**Файлы:**
- Создать: `facade_digitizer/synth/__init__.py`
- Создать: `facade_digitizer/synth/scene.py`
- Тест: `tests/test_synth.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane`
- Предоставляет: `Opening(x, y, width, height, depth)`,
  `SyntheticScene(width_mm, height_mm, openings, camera, K, image_size)` с методами
  `render() -> np.ndarray` (uint8, одноканальное),
  `project(points_mm, depth=0.0) -> np.ndarray` (координаты в пикселях),
  `ground_truth_contours() -> dict[str, np.ndarray]`

Генератор нужен раньше конвейера: без сцены с наперёд известной геометрией нечем проверить ни
точки схода, ни гомографию, ни ректификацию. Рисуется намеренно грубо — прямоугольники разной
яркости; задача генератора в геометрической точности, не в фотореализме.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_synth.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.synth.scene import Opening, SyntheticScene


def make_scene(theta_x_deg=15.0, theta_y_deg=10.0, depth=150.0):
    cz = 10000.0
    cam = CameraOnPlane(
        cx=5000.0 - cz * np.tan(np.radians(theta_x_deg)),
        cy=4000.0 - cz * np.tan(np.radians(theta_y_deg)),
        cz=cz,
    )
    K = np.array([[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]])
    openings = [Opening(x=4000.0, y=3000.0, width=1460.0, height=1900.0, depth=depth)]
    return SyntheticScene(20000.0, 15000.0, openings, cam, K, (5280, 3956))


def test_render_produces_image_of_requested_size():
    img = make_scene().render()
    assert img.shape == (3956, 5280)
    assert img.dtype == np.uint8


def test_render_is_not_uniform():
    img = make_scene().render()
    assert img.std() > 5.0


def test_projection_is_reversible_through_ground_truth():
    scene = make_scene()
    gt = scene.ground_truth_contours()
    assert "wall_plane" in gt
    assert gt["wall_plane"].shape == (1, 4, 2)


def test_deeper_opening_shows_wider_reveal():
    shallow = make_scene(depth=50.0)
    deep = make_scene(depth=300.0)
    assert deep.reveal_width_px("w_0") > shallow.reveal_width_px("w_0")


def test_frontal_view_shows_no_reveal():
    scene = make_scene(theta_x_deg=0.0, theta_y_deg=0.0, depth=150.0)
    assert scene.reveal_width_px("w_0") == pytest.approx(0.0, abs=1.0)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_synth.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.synth'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/synth/__init__.py` — пустой.

Файл `facade_digitizer/synth/scene.py`:

```python
"""Синтетические сцены с наперёд известной геометрией — основа всех геометрических тестов."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import CameraOnPlane
from ..geometry.parallax import apparent_position


@dataclass(frozen=True)
class Opening:
    x: float
    y: float
    width: float
    height: float
    depth: float = 0.0

    def corners_mm(self) -> np.ndarray:
        return np.array([
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + self.height],
            [self.x, self.y + self.height],
        ], dtype=float)


@dataclass
class SyntheticScene:
    width_mm: float
    height_mm: float
    openings: list[Opening]
    camera: CameraOnPlane
    K: np.ndarray
    image_size: tuple[int, int]  # (ширина, высота) в пикселях
    _cache: dict = field(default_factory=dict, repr=False)

    def project(self, points_mm: np.ndarray, depth: float = 0.0) -> np.ndarray:
        """Точки плоскости фасада (мм) → пиксели изображения.

        Сначала точка сносится в Π по лучу (эффект параллакса), затем проецируется
        ортографически-масштабно через K. Этого достаточно: нас интересует
        геометрия в плоскости, а не полная модель камеры.
        """
        pts = np.atleast_2d(np.asarray(points_mm, dtype=float))
        if depth != 0.0:
            pts = np.array([apparent_position(self.camera, px, py, depth) for px, py in pts])

        cam = self.camera
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx_px, cy_px = self.K[0, 2], self.K[1, 2]
        u = cx_px + fx * (pts[:, 0] - cam.cx) / cam.cz
        v = cy_px - fy * (pts[:, 1] - cam.cy) / cam.cz  # ось Y вверх, ось v вниз
        return np.column_stack([u, v])

    def render(self) -> np.ndarray:
        w_px, h_px = self.image_size
        img = np.full((h_px, w_px), 180, dtype=np.uint8)

        wall = self.project(np.array([
            [0.0, 0.0], [self.width_mm, 0.0],
            [self.width_mm, self.height_mm], [0.0, self.height_mm],
        ]))
        cv2.fillPoly(img, [wall.astype(np.int32)], 150)

        for op in self.openings:
            outer = self.project(op.corners_mm())
            cv2.fillPoly(img, [outer.astype(np.int32)], 90)      # откос
            inner = self.project(op.corners_mm(), depth=op.depth)
            cv2.fillPoly(img, [inner.astype(np.int32)], 40)      # полотно
        return img

    def ground_truth_contours(self) -> dict[str, np.ndarray]:
        return {
            "wall_plane": np.array([op.corners_mm() for op in self.openings]),
            "inner": np.array([
                np.array([apparent_position(self.camera, x, y, op.depth)
                          for x, y in op.corners_mm()])
                for op in self.openings
            ]),
        }

    def reveal_width_px(self, opening_id: str) -> float:
        """Ширина видимой вертикальной грани откоса в пикселях."""
        idx = int(opening_id.split("_")[1])
        op = self.openings[idx]
        outer = self.project(op.corners_mm())
        inner = self.project(op.corners_mm(), depth=op.depth)
        return float(np.max(np.abs(inner[:, 0] - outer[:, 0])))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_synth.py -v`
Ожидается: 5 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/synth/ tests/test_synth.py
git commit -m "Синтетический генератор сцен с известной глубиной заглубления"
```

---

## Задача 6: Детекция отрезков и группировка по направлениям

**Файлы:**
- Создать: `facade_digitizer/geometry/vanishing.py`
- Тест: `tests/test_vanishing.py`

**Интерфейсы:**
- Потребляет: ничего из предыдущих задач (работает с изображением)
- Предоставляет: `detect_segments(gray: np.ndarray, min_length_px: float = 40.0) -> np.ndarray`
  формы `(N, 4)` со строками `[x1, y1, x2, y2]`

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_vanishing.py`:

```python
import numpy as np

from facade_digitizer.geometry.vanishing import detect_segments
from tests.test_synth import make_scene


def test_detects_segments_on_synthetic_facade():
    img = make_scene().render()
    segs = detect_segments(img)
    assert segs.shape[1] == 4
    assert len(segs) >= 8  # четыре стороны стены плюс контуры проёма


def test_filters_short_segments():
    img = make_scene().render()
    long_only = detect_segments(img, min_length_px=500.0)
    all_segs = detect_segments(img, min_length_px=10.0)
    assert len(long_only) < len(all_segs)


def test_returns_empty_on_blank_image():
    blank = np.full((400, 400), 128, dtype=np.uint8)
    assert len(detect_segments(blank)) == 0
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_vanishing.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/geometry/vanishing.py`:

```python
"""Отрезки и точки схода. Спецификация, п. 4.2."""
import cv2
import numpy as np


def detect_segments(gray: np.ndarray, min_length_px: float = 40.0) -> np.ndarray:
    """Отрезки на изображении, отфильтрованные по длине.

    Требует OpenCV >= 4.8.0: в 4.1–4.7 LSD был изъят из-за патента.
    """
    if gray.ndim != 2:
        raise ValueError("ожидается одноканальное изображение")

    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        return np.empty((0, 4), dtype=float)

    segs = lines.reshape(-1, 4).astype(float)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= min_length_px]
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_vanishing.py -v`
Ожидается: 3 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/vanishing.py tests/test_vanishing.py
git commit -m "Детекция отрезков на снимке фасада с фильтрацией по длине"
```

---

## Задача 7: Точки схода с метрикой доверия

**Файлы:**
- Изменить: `facade_digitizer/geometry/vanishing.py`
- Изменить: `tests/test_vanishing.py`

**Интерфейсы:**
- Потребляет: `detect_segments` из задачи 6
- Предоставляет: `VanishingPoint(point: np.ndarray, support: int, residual_px: float)`,
  `PlaneConfidence(value: float, support_h: int, support_v: int, residual_px: float,
  orthogonality_deg: float, coverage: float, reasons: list[str])`,
  `estimate_vanishing_points(segments, image_size, K) -> tuple[VanishingPoint, VanishingPoint, PlaneConfidence]`

Спецификация настаивает: модуль возвращает не только результат, но и меру доверия, и ниже порога
система не гадает, а передаёт управление оператору (п. 4.2). Поэтому `PlaneConfidence` —
не диагностика, а часть контракта.

- [ ] **Шаг 1: Дописать падающие тесты**

Добавить в `tests/test_vanishing.py`:

```python
import pytest

from facade_digitizer.geometry.vanishing import estimate_vanishing_points

K = np.array([[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]])


def test_vanishing_points_are_found_on_tilted_facade():
    scene = make_scene(theta_x_deg=20.0, theta_y_deg=15.0)
    segs = detect_segments(scene.render())
    vh, vv, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert conf.value > 0.5
    assert conf.support_h >= 2 and conf.support_v >= 2


def test_confidence_is_low_without_structure():
    noise = np.random.default_rng(0).integers(0, 255, (600, 600), dtype=np.uint8)
    segs = detect_segments(noise)
    _, _, conf = estimate_vanishing_points(segs, (600, 600), K)
    assert conf.value < 0.5
    assert conf.reasons


def test_orthogonality_close_to_ninety_on_valid_scene():
    scene = make_scene(theta_x_deg=20.0, theta_y_deg=15.0)
    segs = detect_segments(scene.render())
    _, _, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=6.0)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Выполнить: `pytest tests/test_vanishing.py -v`
Ожидается: `ImportError: cannot import name 'estimate_vanishing_points'`

- [ ] **Шаг 3: Реализовать**

Дописать в `facade_digitizer/geometry/vanishing.py`:

```python
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VanishingPoint:
    point: np.ndarray       # однородные координаты (3,)
    support: int
    residual_px: float


@dataclass(frozen=True)
class PlaneConfidence:
    value: float
    support_h: int
    support_v: int
    residual_px: float
    orthogonality_deg: float
    coverage: float
    reasons: list[str] = field(default_factory=list)


def _segment_lines(segs: np.ndarray) -> np.ndarray:
    """Прямые в однородных координатах через векторное произведение концов."""
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    return np.cross(p1, p2)


def _fit_vanishing_point(lines: np.ndarray, threshold: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC по парам прямых. Возвращает точку и маску поддержки."""
    best_point, best_mask = None, np.zeros(len(lines), dtype=bool)
    if len(lines) < 2:
        return np.array([0.0, 0.0, 1.0]), best_mask

    for _ in range(200):
        i, j = rng.choice(len(lines), size=2, replace=False)
        v = np.cross(lines[i], lines[j])
        norm = np.linalg.norm(v[:2])
        if norm < 1e-9:
            continue
        dist = np.abs(lines @ v) / (np.linalg.norm(lines[:, :2], axis=1) * np.linalg.norm(v) + 1e-12)
        mask = dist < threshold
        if mask.sum() > best_mask.sum():
            best_point, best_mask = v, mask

    if best_point is None:
        return np.array([0.0, 0.0, 1.0]), best_mask
    return best_point, best_mask


def estimate_vanishing_points(
    segments: np.ndarray,
    image_size: tuple[int, int],
    K: np.ndarray,
    threshold: float = 0.02,
    seed: int = 0,
) -> tuple[VanishingPoint, VanishingPoint, PlaneConfidence]:
    """Две ортогональные точки схода плюс мера доверия к плоскости."""
    reasons: list[str] = []
    if len(segments) < 8:
        reasons.append("слишком мало отрезков")
        empty = VanishingPoint(np.array([1.0, 0.0, 0.0]), 0, float("inf"))
        return empty, empty, PlaneConfidence(0.0, 0, 0, float("inf"), 0.0, 0.0, reasons)

    angles = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1],
                                   segments[:, 2] - segments[:, 0])) % 180.0
    horiz = segments[(angles < 45.0) | (angles > 135.0)]
    vert = segments[(angles >= 45.0) & (angles <= 135.0)]

    rng = np.random.default_rng(seed)
    vh_pt, vh_mask = _fit_vanishing_point(_segment_lines(horiz), threshold, rng) \
        if len(horiz) >= 2 else (np.array([1.0, 0.0, 0.0]), np.zeros(0, dtype=bool))
    vv_pt, vv_mask = _fit_vanishing_point(_segment_lines(vert), threshold, rng) \
        if len(vert) >= 2 else (np.array([0.0, 1.0, 0.0]), np.zeros(0, dtype=bool))

    Kinv = np.linalg.inv(K)
    dh, dv = Kinv @ vh_pt, Kinv @ vv_pt
    dh, dv = dh / (np.linalg.norm(dh) + 1e-12), dv / (np.linalg.norm(dv) + 1e-12)
    orthogonality = math.degrees(math.acos(min(1.0, abs(float(dh @ dv)))))

    support_h, support_v = int(vh_mask.sum()), int(vv_mask.sum())
    coverage = (support_h + support_v) / max(len(segments), 1)
    residual = float(threshold * max(image_size))

    if support_h < 2 or support_v < 2:
        reasons.append("недостаточная поддержка одной из точек схода")
    if abs(orthogonality - 90.0) > 15.0:
        reasons.append(f"направления не ортогональны: {orthogonality:.1f}°")
    if coverage < 0.2:
        reasons.append("отрезки покрывают малую долю кадра")

    value = 0.0 if reasons else min(1.0, 0.3 + 0.7 * coverage)

    return (
        VanishingPoint(vh_pt, support_h, residual),
        VanishingPoint(vv_pt, support_v, residual),
        PlaneConfidence(value, support_h, support_v, residual, orthogonality, coverage, reasons),
    )
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_vanishing.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/vanishing.py tests/test_vanishing.py
git commit -m "Оценка точек схода по RANSAC с явной метрикой доверия к плоскости"
```

---

## Задача 8: Гомография и поза камеры

**Файлы:**
- Создать: `facade_digitizer/geometry/homography.py`
- Тест: `tests/test_homography.py`

**Интерфейсы:**
- Потребляет: `VanishingPoint` из задачи 7, `CameraOnPlane` из задачи 2
- Предоставляет:
  `homography_from_vanishing_points(vh, vv, K, image_size) -> np.ndarray` (3×3),
  `camera_pose_from_homography(H, K, mm_per_px) -> CameraOnPlane`,
  `homography_from_four_points(image_pts, aspect_ratio=None, size_mm=None) -> np.ndarray`

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_homography.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.homography import (
    homography_from_four_points,
    homography_from_vanishing_points,
)
from facade_digitizer.geometry.vanishing import detect_segments, estimate_vanishing_points
from tests.test_synth import make_scene

K = np.array([[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]])


def _apply(H, pts):
    h = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return h[:, :2] / h[:, 2:3]


def test_rectified_rectangle_has_right_angles():
    scene = make_scene(theta_x_deg=20.0, theta_y_deg=15.0, depth=0.0)
    segs = detect_segments(scene.render())
    vh, vv, conf = estimate_vanishing_points(segs, scene.image_size, K)
    H = homography_from_vanishing_points(vh, vv, K, scene.image_size)

    corners_mm = np.array([[0.0, 0.0], [20000.0, 0.0], [20000.0, 15000.0], [0.0, 15000.0]])
    rect = _apply(H, scene.project(corners_mm))

    v1 = rect[1] - rect[0]
    v2 = rect[3] - rect[0]
    cosang = abs(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert cosang < 0.10  # отклонение от прямого угла менее ~6°


def test_four_point_requires_disambiguation():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError):
        homography_from_four_points(pts)


def test_four_point_with_aspect_ratio_produces_that_ratio():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    H = homography_from_four_points(pts, aspect_ratio=2.0)
    rect = _apply(H, pts)
    width = np.linalg.norm(rect[1] - rect[0])
    height = np.linalg.norm(rect[3] - rect[0])
    assert width / height == pytest.approx(2.0, rel=1e-6)


def test_four_point_with_sizes_produces_those_sizes():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    H = homography_from_four_points(pts, size_mm=(1460.0, 1900.0))
    rect = _apply(H, pts)
    assert np.linalg.norm(rect[1] - rect[0]) == pytest.approx(1460.0, rel=1e-6)
    assert np.linalg.norm(rect[3] - rect[0]) == pytest.approx(1900.0, rel=1e-6)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_homography.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/geometry/homography.py`:

```python
"""Гомография приведения к плоскости фасада и поза камеры. Спецификация, п. 4.2."""
import numpy as np

from .camera import CameraOnPlane
from .vanishing import VanishingPoint


def homography_from_vanishing_points(
    vh: VanishingPoint, vv: VanishingPoint, K: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Гомография изображение → фронто-параллельный вид плоскости фасада.

    Две ортогональные точки схода задают направления осей плоскости в системе камеры.
    Ортогонализация по Граму — Шмидту компенсирует неточность оценки.
    """
    Kinv = np.linalg.inv(K)
    d1 = Kinv @ vh.point
    d1 /= np.linalg.norm(d1) + 1e-12
    d2 = Kinv @ vv.point
    d2 -= (d2 @ d1) * d1
    d2 /= np.linalg.norm(d2) + 1e-12
    d3 = np.cross(d1, d2)

    R = np.column_stack([d1, d2, d3])
    H = R.T @ Kinv

    # Центр кадра остаётся на месте: убираем произвол в сдвиге и масштабе.
    w, h = image_size
    center = np.array([w / 2.0, h / 2.0, 1.0])
    mapped = H @ center
    H = H / (mapped[2] + 1e-12)
    return H


def camera_pose_from_homography(
    H: np.ndarray, K: np.ndarray, mm_per_px: float
) -> CameraOnPlane:
    """Положение камеры относительно плоскости фасада в миллиметрах.

    Из H и K восстанавливается направление нормали; расстояние до плоскости даётся
    масштабом: cz = f · mm_per_px, где f — фокусное в пикселях.
    """
    Hinv = np.linalg.inv(H)
    n = Hinv[:, 2]
    n = n / (np.linalg.norm(n) + 1e-12)

    f = float((K[0, 0] + K[1, 1]) / 2.0)
    cz = f * mm_per_px
    cx = -n[0] / (abs(n[2]) + 1e-12) * cz
    cy = -n[1] / (abs(n[2]) + 1e-12) * cz
    return CameraOnPlane(cx=cx, cy=cy, cz=cz)


def homography_from_four_points(
    image_pts: np.ndarray,
    aspect_ratio: float | None = None,
    size_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Ручной вариант: четыре точки прямоугольника плюс доопределение.

    Четыре точки, объявленные прямоугольником, оставляют одну неизвестную —
    отношение сторон. Без него задача не решается. Спецификация, п. 4.2.
    """
    import cv2

    if aspect_ratio is None and size_mm is None:
        raise ValueError(
            "четырёх точек недостаточно: задайте aspect_ratio либо size_mm, "
            "иначе отношение сторон не определено"
        )
    if size_mm is not None:
        w, h = size_mm
    else:
        w, h = float(aspect_ratio), 1.0

    src = np.asarray(image_pts, dtype=np.float32)
    dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(float)
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_homography.py -v`
Ожидается: 4 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/homography.py tests/test_homography.py
git commit -m "Гомография из точек схода, поза камеры и математически корректный ручной вариант"
```

---

## Задача 9: Поле углов визирования и локальный GSD

**Файлы:**
- Создать: `facade_digitizer/geometry/angles.py`
- Тест: `tests/test_angles.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane`
- Предоставляет: `AngleMap(theta_x: np.ndarray, theta_y: np.ndarray, theta_full: np.ndarray)`,
  `angle_map(cam, bounds_mm, shape) -> AngleMap`,
  `local_gsd(cam, H, mm_per_px, shape) -> np.ndarray`,
  `usable_mask(angle_map, theta_max_deg=30.0) -> np.ndarray`

Это выход, которого не было в редакции 2 спецификации; без него коррекция параллакса невыполнима
(п. 4, примечание к `rectify`).

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_angles.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.angles import angle_map, usable_mask
from facade_digitizer.geometry.camera import CameraOnPlane


def test_angle_is_zero_at_foot_point():
    cam = CameraOnPlane(cx=10000.0, cy=7500.0, cz=10000.0)
    am = angle_map(cam, bounds_mm=(0.0, 0.0, 20000.0, 15000.0), shape=(101, 101))
    assert am.theta_full.min() == pytest.approx(0.0, abs=0.5)


def test_angle_grows_toward_edges():
    cam = CameraOnPlane(cx=10000.0, cy=7500.0, cz=10000.0)
    am = angle_map(cam, bounds_mm=(0.0, 0.0, 20000.0, 15000.0), shape=(101, 101))
    assert am.theta_full[0, 0] > am.theta_full[50, 50]


def test_components_have_expected_signs():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    am = angle_map(cam, bounds_mm=(-5000.0, -5000.0, 5000.0, 5000.0), shape=(3, 3))
    assert am.theta_x[1, 0] < 0        # левее опорной точки
    assert am.theta_x[1, 2] > 0        # правее


def test_usable_mask_excludes_steep_angles():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    am = angle_map(cam, bounds_mm=(0.0, 0.0, 20000.0, 15000.0), shape=(51, 51))
    mask = usable_mask(am, theta_max_deg=30.0)
    assert mask.any()
    assert not mask.all()


def test_local_gsd_is_uniform_for_identity_homography():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    gsd = local_gsd(cam, np.eye(3), mm_per_px=3.0, shape=(20, 30))
    assert gsd.shape == (20, 30)
    assert np.allclose(gsd, 3.0)


def test_local_gsd_varies_under_perspective():
    """При наклоне разрешение на дальней стороне хуже. Спецификация, п. 2.2."""
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0008, 0.0, 1.0]])
    gsd = local_gsd(cam, H, mm_per_px=3.0, shape=(20, 30))
    assert gsd.max() / gsd.min() > 1.5
```

Не забыть дописать импорт в начало файла:

```python
from facade_digitizer.geometry.angles import angle_map, local_gsd, usable_mask
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_angles.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/geometry/angles.py`:

```python
"""Поле углов визирования по плоскости фасада и локальное разрешение."""
from dataclasses import dataclass

import numpy as np

from .camera import CameraOnPlane


@dataclass(frozen=True)
class AngleMap:
    theta_x: np.ndarray    # градусы, со знаком
    theta_y: np.ndarray
    theta_full: np.ndarray  # градусы, всегда >= 0
    bounds_mm: tuple[float, float, float, float]


def angle_map(
    cam: CameraOnPlane,
    bounds_mm: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> AngleMap:
    """Углы θ_x, θ_y и полный угол на сетке в плоскости фасада."""
    x0, y0, x1, y1 = bounds_mm
    rows, cols = shape
    xs = np.linspace(x0, x1, cols)
    ys = np.linspace(y0, y1, rows)
    gx, gy = np.meshgrid(xs, ys)

    tan_x = (gx - cam.cx) / cam.cz
    tan_y = (gy - cam.cy) / cam.cz
    return AngleMap(
        theta_x=np.degrees(np.arctan(tan_x)),
        theta_y=np.degrees(np.arctan(tan_y)),
        theta_full=np.degrees(np.arctan(np.hypot(tan_x, tan_y))),
        bounds_mm=bounds_mm,
    )


def usable_mask(am: AngleMap, theta_max_deg: float = 30.0) -> np.ndarray:
    """Область кадра, пригодная для измерений. Спецификация, п. 2.4."""
    return am.theta_full <= theta_max_deg


def local_gsd(
    cam: CameraOnPlane, H: np.ndarray, mm_per_px: float, shape: tuple[int, int]
) -> np.ndarray:
    """Локальное разрешение (мм на пиксель) по полю ректифицированного изображения.

    Якобиан гомографии в точке даёт локальное растяжение; худшее значение по контуру
    элемента и есть тот GSD, к которому относится требование спецификации.
    """
    rows, cols = shape
    ys, xs = np.mgrid[0:rows, 0:cols].astype(float)
    ones = np.ones_like(xs)
    denom = H[2, 0] * xs + H[2, 1] * ys + H[2, 2] * ones
    scale = np.abs(denom) ** 2 / (abs(np.linalg.det(H)) + 1e-12)
    return scale * mm_per_px
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_angles.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/angles.py tests/test_angles.py
git commit -m "Поле углов визирования, маска пригодной области и локальное разрешение"
```

---

## Задача 10: Загрузка снимка и внутренние параметры камеры

**Файлы:**
- Создать: `facade_digitizer/pipeline/__init__.py`
- Создать: `facade_digitizer/pipeline/io.py`
- Создать: `facade_digitizer/pipeline/calib.py`
- Тест: `tests/test_calib.py`

**Интерфейсы:**
- Потребляет: ничего из предыдущих задач
- Предоставляет:
  `load_image(path) -> tuple[np.ndarray, CameraMeta]`,
  `CameraMeta(model: str, focal_mm: float | None, sensor_width_mm: float | None,
  image_size: tuple[int, int], captured_at: str | None, gnss: dict | None)`,
  `intrinsics_from_meta(meta) -> tuple[np.ndarray, str]` — матрица K и источник
  (`"exif"` / `"database"`),
  `undistort(image, K, dist) -> np.ndarray`

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_calib.py`:

```python
import numpy as np
import pytest

from facade_digitizer.pipeline.calib import intrinsics_from_meta, undistort
from facade_digitizer.pipeline.io import CameraMeta


def test_intrinsics_from_focal_and_sensor_width():
    meta = CameraMeta(model="M3E", focal_mm=24.0, sensor_width_mm=17.3,
                      image_size=(5280, 3956), captured_at=None, gnss=None)
    K, source = intrinsics_from_meta(meta)
    assert source == "exif"
    assert K[0, 0] == pytest.approx(5280 * 24.0 / 17.3, rel=1e-6)
    assert K[0, 2] == pytest.approx(2640.0)
    assert K[1, 2] == pytest.approx(1978.0)


def test_intrinsics_fall_back_to_default_fov():
    """Без EXIF принимается типичное поле зрения 84° по диагонали."""
    meta = CameraMeta(model="unknown", focal_mm=None, sensor_width_mm=None,
                      image_size=(4000, 3000), captured_at=None, gnss=None)
    K, source = intrinsics_from_meta(meta)
    assert source == "database"
    assert K[0, 0] > 0


def test_undistort_is_identity_for_zero_coefficients():
    img = np.random.default_rng(0).integers(0, 255, (200, 300), dtype=np.uint8)
    K = np.array([[300.0, 0, 150.0], [0, 300.0, 100.0], [0, 0, 1.0]])
    out = undistort(img, K, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(out, img)


def test_calibration_profile_roundtrip(tmp_path):
    """Профиль камеры сохраняется и читается — калибровка по мишени делается однажды."""
    from facade_digitizer.pipeline.calib import CalibrationProfile, load_profile, save_profile

    prof = CalibrationProfile(
        model="M3E", K=[[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]],
        dist=[-0.21, 0.09, 0.0, 0.0, -0.02], rms_px=0.34, image_size=(5280, 3956),
    )
    path = tmp_path / "m3e.json"
    save_profile(prof, path)
    assert load_profile(path).rms_px == pytest.approx(0.34)


def test_intrinsics_prefer_profile_over_exif(tmp_path):
    from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile

    prof = CalibrationProfile(
        model="M3E", K=[[3333.0, 0, 2640.0], [0, 3333.0, 1978.0], [0, 0, 1.0]],
        dist=[0.0] * 5, rms_px=0.3, image_size=(5280, 3956),
    )
    path = tmp_path / "m3e.json"
    save_profile(prof, path)

    meta = CameraMeta(model="M3E", focal_mm=24.0, sensor_width_mm=17.3,
                      image_size=(5280, 3956), captured_at=None, gnss=None)
    K, source = intrinsics_from_meta(meta, profile_path=path)
    assert source == "target"
    assert K[0, 0] == pytest.approx(3333.0)


def test_calibrate_from_chessboard_rejects_too_few_views():
    from facade_digitizer.pipeline.calib import calibrate_from_chessboard

    with pytest.raises(ValueError):
        calibrate_from_chessboard([], pattern=(9, 6), square_mm=25.0,
                                  image_size=(640, 480), model="test")
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_calib.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/__init__.py` — пустой.

Файл `facade_digitizer/pipeline/io.py`:

```python
"""Загрузка снимка и разбор EXIF."""
import math
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CameraMeta:
    model: str
    focal_mm: float | None
    sensor_width_mm: float | None
    image_size: tuple[int, int]      # (ширина, высота) в пикселях
    captured_at: str | None
    gnss: dict | None


def load_image(path: str | Path):
    """Снимок в градациях серого плюс метаданные камеры."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"не удалось прочитать изображение: {path}")

    h, w = img.shape[:2]
    meta = _read_exif(path, (w, h))
    return img, meta


def _read_exif(path: Path, image_size: tuple[int, int]) -> CameraMeta:
    try:
        import piexif
        exif = piexif.load(str(path))
    except Exception:
        return CameraMeta("unknown", None, None, image_size, None, None)

    zeroth = exif.get("0th", {})
    exif_ifd = exif.get("Exif", {})

    def rational(tag_dict, tag):
        v = tag_dict.get(tag)
        if isinstance(v, tuple) and len(v) == 2 and v[1]:
            return v[0] / v[1]
        return None

    model = zeroth.get(piexif.ImageIFD.Model, b"unknown")
    model = model.decode(errors="replace") if isinstance(model, bytes) else str(model)
    focal = rational(exif_ifd, piexif.ExifIFD.FocalLength)
    captured = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
    captured = captured.decode(errors="replace") if isinstance(captured, bytes) else None

    return CameraMeta(model, focal, None, image_size, captured, None)
```

Файл `facade_digitizer/pipeline/calib.py`:

```python
"""Внутренние параметры камеры и снятие дисторсии. Спецификация, п. 4.2."""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import CameraMeta

DEFAULT_DIAGONAL_FOV_DEG = 84.0  # типично для DJI Zenmuse L2 и Mavic 3


@dataclass(frozen=True)
class CalibrationProfile:
    """Результат разовой калибровки камеры по мишени. Спецификация, п. 4.2."""
    model: str
    K: list[list[float]]
    dist: list[float]
    rms_px: float
    image_size: tuple[int, int]


def save_profile(profile: CalibrationProfile, path) -> None:
    Path(path).write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_profile(path) -> CalibrationProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["image_size"] = tuple(data["image_size"])
    return CalibrationProfile(**data)


def calibrate_from_chessboard(
    images: list[np.ndarray],
    pattern: tuple[int, int],
    square_mm: float,
    image_size: tuple[int, int],
    model: str,
) -> CalibrationProfile:
    """Калибровка по снимкам шахматной мишени. Выполняется один раз на аппарат.

    Ошибка фокусного расстояния при наклоне искажает отношение сторон как ε·sin²θ:
    при ε = 2 % и θ = 30 % это 7.5 мм на полутораметровом окне. Поэтому камеры
    собственного парка калибруются по мишени, а не по EXIF. Спецификация, п. 4.2.
    """
    if len(images) < 5:
        raise ValueError("для устойчивой калибровки нужно не менее 5 снимков мишени")

    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    obj_points, img_points = [], []
    for img in images:
        found, corners = cv2.findChessboardCorners(img, pattern, None)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            img, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < 5:
        raise ValueError(f"мишень распознана лишь на {len(obj_points)} снимках из {len(images)}")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    return CalibrationProfile(model=model, K=K.tolist(), dist=dist.ravel().tolist(),
                              rms_px=float(rms), image_size=image_size)


def intrinsics_from_meta(meta: CameraMeta, profile_path=None) -> tuple[np.ndarray, str]:
    """Матрица K и источник её происхождения.

    Приоритет: профиль калибровки по мишени; затем фокусное и ширина матрицы из
    EXIF; затем типичное поле зрения. Источник обязан попасть в выходной файл —
    без него нельзя оценить вклад ошибки фокусного в бюджет (спецификация, п. 6.1).
    """
    w, h = meta.image_size

    if profile_path is not None and Path(profile_path).exists():
        prof = load_profile(profile_path)
        if tuple(prof.image_size) == (w, h):
            return np.array(prof.K, dtype=float), "target"

    if meta.focal_mm and meta.sensor_width_mm:
        fx = fy = w * meta.focal_mm / meta.sensor_width_mm
        source = "exif"
    else:
        diag_px = math.hypot(w, h)
        fx = fy = diag_px / (2.0 * math.tan(math.radians(DEFAULT_DIAGONAL_FOV_DEG) / 2.0))
        source = "database"

    K = np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    return K, source


def undistort(image: np.ndarray, K: np.ndarray, dist: list[float]) -> np.ndarray:
    """Снятие дисторсии. При нулевых коэффициентах возвращается исходное изображение."""
    coeffs = np.asarray(dist, dtype=float)
    if not np.any(coeffs):
        return image
    return cv2.undistort(image, K, coeffs)
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_calib.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/ tests/test_calib.py
git commit -m "Загрузка снимка, EXIF, калибровка по мишени и профиль камеры с указанием источника"
```

---

## Задача 11: Шлюз качества снимка

**Файлы:**
- Создать: `facade_digitizer/pipeline/quality.py`
- Тест: `tests/test_quality.py`

**Интерфейсы:**
- Потребляет: `QualityReport` из задачи 1, `AngleMap` из задачи 9
- Предоставляет: `assess(image, gsd_min, gsd_max, theta_p95, thresholds=None) -> QualityReport`,
  `Thresholds(sharpness_min, gsd_max_mm_px, theta_p95_max_deg)`

Модуль отбраковывает, а не улучшает: генеративный деблюр синтезирует границы, по которым потом
выполняется измерение (спецификация, п. 4.1). Пороги — не свободные параметры: задача их
калибровки по данным стоит в этапе 1 и здесь лишь предусмотрена точка подстановки.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_quality.py`:

```python
import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline.quality import Thresholds, assess
from tests.test_synth import make_scene


def test_sharp_synthetic_image_is_ok():
    img = make_scene().render()
    rep = assess(img, gsd_min=3.1, gsd_max=4.6, theta_p95=22.0)
    assert rep.verdict == "ok"
    assert rep.reasons == []


def test_blurred_image_is_rejected():
    img = cv2.GaussianBlur(make_scene().render(), (31, 31), 12.0)
    rep = assess(img, gsd_min=3.1, gsd_max=4.6, theta_p95=22.0)
    assert rep.verdict in {"degraded", "reject"}
    assert any("резкост" in r for r in rep.reasons)


def test_coarse_gsd_is_rejected():
    img = make_scene().render()
    rep = assess(img, gsd_min=14.0, gsd_max=16.0, theta_p95=22.0)
    assert rep.verdict == "reject"
    assert any("разрешение" in r for r in rep.reasons)


def test_steep_angle_is_degraded():
    img = make_scene().render()
    rep = assess(img, gsd_min=3.1, gsd_max=4.6, theta_p95=44.0)
    assert rep.verdict in {"degraded", "reject"}
    assert any("угол" in r for r in rep.reasons)


def test_thresholds_are_injectable():
    img = cv2.GaussianBlur(make_scene().render(), (31, 31), 12.0)
    lenient = Thresholds(sharpness_min=0.0, gsd_max_mm_px=5.0, theta_p95_max_deg=30.0)
    assert assess(img, 3.1, 4.6, 22.0, thresholds=lenient).verdict == "ok"
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_quality.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/quality.py`:

```python
"""Шлюз пригодности снимка. Отбраковывает, но не улучшает. Спецификация, п. 4.1."""
from dataclasses import dataclass

import cv2
import numpy as np

from ..schema import QualityReport


@dataclass(frozen=True)
class Thresholds:
    sharpness_min: float = 0.010
    gsd_max_mm_px: float = 5.0
    theta_p95_max_deg: float = 30.0


DEFAULT = Thresholds()


def sharpness(image: np.ndarray) -> float:
    """Нормированная вариация лапласиана: безразмерная мера резкости."""
    lap = cv2.Laplacian(image, cv2.CV_64F)
    denom = float(image.std()) ** 2 + 1e-9
    return float(lap.var() / denom) / 1000.0


def assess(
    image: np.ndarray,
    gsd_min: float,
    gsd_max: float,
    theta_p95: float,
    thresholds: Thresholds | None = None,
) -> QualityReport:
    t = thresholds or DEFAULT
    reasons: list[str] = []

    sharp = sharpness(image)
    if sharp < t.sharpness_min:
        reasons.append(f"недостаточная резкость: {sharp:.4f} < {t.sharpness_min}")
    if gsd_max > t.gsd_max_mm_px:
        reasons.append(f"недостаточное разрешение: {gsd_max:.1f} мм/px > {t.gsd_max_mm_px}")
    if theta_p95 > t.theta_p95_max_deg:
        reasons.append(f"слишком крутой угол визирования: P95 = {theta_p95:.1f}°")

    if not reasons:
        verdict = "ok"
    elif any("разрешение" in r for r in reasons):
        verdict = "reject"
    else:
        verdict = "degraded"

    return QualityReport(
        verdict=verdict,
        sharpness=sharp,
        gsd_mm_px_min=gsd_min,
        gsd_mm_px_max=gsd_max,
        theta_field_deg_p95=theta_p95,
        reasons=reasons,
    )
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_quality.py -v`
Ожидается: 5 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/quality.py tests/test_quality.py
git commit -m "Шлюз качества снимка с вердиктом и перечнем причин отбраковки"
```

---

## Задача 12: Ректификация

**Файлы:**
- Создать: `facade_digitizer/pipeline/rectify.py`
- Тест: `tests/test_rectify.py`

**Интерфейсы:**
- Потребляет: `homography_from_vanishing_points` (задача 8), `angle_map`, `local_gsd` (задача 9)
- Предоставляет: `Rectified(image, H, valid_mask, bounds_px)`,
  `rectify(image, H, output_size=None) -> Rectified`

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_rectify.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.homography import homography_from_vanishing_points
from facade_digitizer.geometry.vanishing import detect_segments, estimate_vanishing_points
from facade_digitizer.pipeline.rectify import rectify
from tests.test_synth import make_scene

K = np.array([[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]])


def _rectified_scene():
    scene = make_scene(theta_x_deg=20.0, theta_y_deg=15.0, depth=0.0)
    img = scene.render()
    vh, vv, _ = estimate_vanishing_points(detect_segments(img), scene.image_size, K)
    H = homography_from_vanishing_points(vh, vv, K, scene.image_size)
    return scene, rectify(img, H)


def test_output_has_valid_mask_of_same_shape():
    _, rect = _rectified_scene()
    assert rect.valid_mask.shape == rect.image.shape
    assert rect.valid_mask.dtype == bool


def test_valid_mask_marks_real_content():
    _, rect = _rectified_scene()
    coverage = rect.valid_mask.mean()
    assert 0.1 < coverage < 1.0


def test_rectangle_stays_rectangular_after_rectification():
    scene, rect = _rectified_scene()
    corners_mm = np.array([[4000.0, 3000.0], [5460.0, 3000.0],
                           [5460.0, 4900.0], [4000.0, 4900.0]])
    img_pts = scene.project(corners_mm)
    h = np.column_stack([img_pts, np.ones(4)]) @ rect.H.T
    r = h[:, :2] / h[:, 2:3]

    top = np.linalg.norm(r[1] - r[0])
    bottom = np.linalg.norm(r[2] - r[3])
    assert top == pytest.approx(bottom, rel=0.05)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_rectify.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/rectify.py`:

```python
"""Приведение снимка к фронто-параллельному виду плоскости фасада."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectified:
    image: np.ndarray
    H: np.ndarray               # изображение → ректифицированные пиксели, с учётом сдвига
    valid_mask: np.ndarray      # bool: где есть реальное содержимое, а не заполнение
    bounds_px: tuple[float, float, float, float]


def rectify(image: np.ndarray, H: np.ndarray, output_size: tuple[int, int] | None = None) -> Rectified:
    """Применяет гомографию, обрезая результат по фактическому охвату углов кадра."""
    h, w = image.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    mapped = corners @ H.T
    mapped = mapped[:, :2] / mapped[:, 2:3]

    x0, y0 = mapped.min(axis=0)
    x1, y1 = mapped.max(axis=0)

    if output_size is None:
        out_w = int(np.clip(round(x1 - x0), 16, 4 * w))
        out_h = int(np.clip(round(y1 - y0), 16, 4 * h))
    else:
        out_w, out_h = output_size

    scale = min(out_w / max(x1 - x0, 1e-6), out_h / max(y1 - y0, 1e-6))
    shift = np.array([
        [scale, 0.0, -x0 * scale],
        [0.0, scale, -y0 * scale],
        [0.0, 0.0, 1.0],
    ])
    H_total = shift @ H

    warped = cv2.warpPerspective(image, H_total, (out_w, out_h), flags=cv2.INTER_LINEAR)
    ones = np.full_like(image, 255)
    mask = cv2.warpPerspective(ones, H_total, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0

    return Rectified(image=warped, H=H_total, valid_mask=mask,
                     bounds_px=(0.0, 0.0, float(out_w), float(out_h)))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_rectify.py -v`
Ожидается: 3 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/rectify.py tests/test_rectify.py
git commit -m "Ректификация с маской валидной области и автоматическим охватом кадра"
```

---

## Задача 13: Оркестрация плоскости с передачей управления оператору

**Файлы:**
- Создать: `facade_digitizer/pipeline/plane.py`
- Тест: дописать `tests/test_rectify.py`

**Интерфейсы:**
- Потребляет: задачи 6–9, 12
- Предоставляет: `PlaneResult(H, camera, confidence, method, needs_operator)`,
  `estimate_plane(image, K, mm_per_px, min_confidence=0.5) -> PlaneResult`,
  `estimate_plane_manual(image_pts, aspect_ratio=None, size_mm=None, mm_per_px=1.0) -> PlaneResult`

- [ ] **Шаг 1: Дописать падающие тесты**

Добавить в `tests/test_rectify.py`:

```python
from facade_digitizer.pipeline.plane import estimate_plane, estimate_plane_manual


def test_plane_is_estimated_on_good_scene():
    scene = make_scene(theta_x_deg=20.0, theta_y_deg=15.0, depth=0.0)
    res = estimate_plane(scene.render(), K, mm_per_px=2.8)
    assert res.method == "vanishing_points"
    assert not res.needs_operator
    assert res.camera.cz > 0


def test_low_confidence_requests_operator_instead_of_guessing():
    """Спецификация, п. 4.2: ниже порога система не гадает."""
    noise = np.random.default_rng(1).integers(0, 255, (600, 600), dtype=np.uint8)
    res = estimate_plane(noise, K, mm_per_px=2.8)
    assert res.needs_operator
    assert res.confidence.reasons


def test_manual_plane_requires_disambiguation():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError):
        estimate_plane_manual(pts)


def test_manual_plane_succeeds_with_sizes():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, size_mm=(1460.0, 1900.0), mm_per_px=1.0)
    assert res.method == "manual_four_point"
    assert res.confidence.value == 1.0
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Выполнить: `pytest tests/test_rectify.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.pipeline.plane'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/plane.py`:

```python
"""Оркестрация оценки плоскости фасада. Спецификация, п. 4.2."""
from dataclasses import dataclass

import numpy as np

from ..geometry.camera import CameraOnPlane
from ..geometry.homography import (
    camera_pose_from_homography,
    homography_from_four_points,
    homography_from_vanishing_points,
)
from ..geometry.vanishing import PlaneConfidence, detect_segments, estimate_vanishing_points


@dataclass(frozen=True)
class PlaneResult:
    H: np.ndarray
    camera: CameraOnPlane
    confidence: PlaneConfidence
    method: str
    needs_operator: bool


def estimate_plane(
    image: np.ndarray, K: np.ndarray, mm_per_px: float, min_confidence: float = 0.5
) -> PlaneResult:
    """Автоматическая оценка. Ниже порога доверия — запрос к оператору, а не догадка."""
    h, w = image.shape[:2]
    segments = detect_segments(image)
    vh, vv, conf = estimate_vanishing_points(segments, (w, h), K)

    if conf.value < min_confidence:
        return PlaneResult(
            H=np.eye(3), camera=CameraOnPlane(0.0, 0.0, mm_per_px * float(K[0, 0])),
            confidence=conf, method="vanishing_points", needs_operator=True,
        )

    H = homography_from_vanishing_points(vh, vv, K, (w, h))
    camera = camera_pose_from_homography(H, K, mm_per_px)
    return PlaneResult(H=H, camera=camera, confidence=conf,
                       method="vanishing_points", needs_operator=False)


def estimate_plane_manual(
    image_pts: np.ndarray,
    aspect_ratio: float | None = None,
    size_mm: tuple[float, float] | None = None,
    mm_per_px: float = 1.0,
) -> PlaneResult:
    """Ручной вариант: четыре точки плюс доопределение отношения сторон."""
    H = homography_from_four_points(image_pts, aspect_ratio=aspect_ratio, size_mm=size_mm)
    conf = PlaneConfidence(
        value=1.0, support_h=4, support_v=4, residual_px=0.0,
        orthogonality_deg=90.0, coverage=1.0, reasons=[],
    )
    camera = CameraOnPlane(0.0, 0.0, max(mm_per_px * 1000.0, 1.0))
    return PlaneResult(H=H, camera=camera, confidence=conf,
                       method="manual_four_point", needs_operator=False)
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_rectify.py -v`
Ожидается: 7 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/plane.py tests/test_rectify.py
git commit -m "Оркестрация плоскости фасада с передачей управления оператору при низком доверии"
```

---

## Задача 14: Сборка конвейера, CLI и сквозной тест

**Файлы:**
- Создать: `facade_digitizer/pipeline/run.py`
- Тест: `tests/test_e2e.py`

**Интерфейсы:**
- Потребляет: все предыдущие задачи
- Предоставляет: `process(image_path, mm_per_px, scale_source) -> FacadeModel`,
  `main() -> int` — точка входа CLI `facade-digitize`

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_e2e.py`:

```python
import time

import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline.run import process
from facade_digitizer.schema import FacadeModel
from tests.test_synth import make_scene


@pytest.fixture
def scene_file(tmp_path):
    scene = make_scene(theta_x_deg=18.0, theta_y_deg=12.0, depth=150.0)
    path = tmp_path / "facade.png"
    cv2.imwrite(str(path), scene.render())
    return path


def test_produces_valid_schema(scene_file):
    model = process(scene_file, mm_per_px=2.8, scale_source="operator_reference")
    assert isinstance(model, FacadeModel)
    assert model.schema_version == "1.1"
    FacadeModel.model_validate_json(model.model_dump_json())


def test_records_scale_source_and_uncertainty(scene_file):
    """Инвариант спецификации: величина без источника и σ не выпускается."""
    model = process(scene_file, mm_per_px=2.8, scale_source="operator_reference")
    assert model.facade.scale.source == "operator_reference"
    assert model.facade.scale.sigma_rel > 0


def test_records_camera_pose_and_angle_field(scene_file):
    model = process(scene_file, mm_per_px=2.8, scale_source="operator_reference")
    img = model.images[0]
    assert img.pose_to_facade is not None
    assert img.theta_field_deg["p95"] >= 0.0


def test_exif_scale_source_is_marked_as_not_meeting_tolerance(scene_file):
    """Источники 3 и 4 допуск не выполняют. Спецификация, п. 6.3."""
    model = process(scene_file, mm_per_px=2.8, scale_source="assumed_floor_height")
    assert model.facade.scale.meets_tolerance is False


def test_pipeline_meets_performance_budget(scene_file):
    """Спецификация, раздел 16: не более 15 с на снимок."""
    start = time.perf_counter()
    process(scene_file, mm_per_px=2.8, scale_source="operator_reference")
    assert time.perf_counter() - start < 15.0
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_e2e.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.pipeline.run'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/run.py`:

```python
"""Сборка геометрического конвейера и точка входа CLI."""
import argparse
import sys
from pathlib import Path

import numpy as np

from ..geometry.angles import angle_map
from ..schema import (
    SCHEMA_VERSION,
    CameraIntrinsics,
    FacadeModel,
    FacadeRecord,
    ImageRecord,
    Rectification,
    ScaleEstimate,
)
from .calib import intrinsics_from_meta, undistort
from .io import load_image
from .plane import estimate_plane
from .quality import assess
from .rectify import rectify

SOFTWARE_VERSION = "facade-digitizer 0.1.0"

# σ и соответствие допуску по источнику масштаба. Спецификация, п. 6.3.
SCALE_PROFILE = {
    "operator_reference": (0.0022, True),
    "photogrammetry": (0.0010, True),
    "exif_range": (0.030, False),
    "assumed_floor_height": (0.050, False),
}


def process(image_path, mm_per_px: float, scale_source: str) -> FacadeModel:
    """Снимок → модель фасада без элементов: геометрия, поза, качество."""
    if scale_source not in SCALE_PROFILE:
        raise ValueError(f"неизвестный источник масштаба: {scale_source}")

    image, meta = load_image(image_path)
    K, k_source = intrinsics_from_meta(meta)
    image = undistort(image, K, [0.0, 0.0, 0.0, 0.0, 0.0])

    plane = estimate_plane(image, K, mm_per_px)
    rect = rectify(image, plane.H)

    h, w = image.shape[:2]
    bounds_mm = (0.0, 0.0, w * mm_per_px, h * mm_per_px)
    am = angle_map(plane.camera, bounds_mm, shape=(64, 64))
    theta_p95 = float(np.percentile(am.theta_full, 95))

    gsd_min = gsd_max = mm_per_px
    quality = assess(image, gsd_min, gsd_max, theta_p95)

    sigma_rel, meets = SCALE_PROFILE[scale_source]

    record = ImageRecord(
        id="img_0",
        path=str(Path(image_path).name),
        camera=CameraIntrinsics(model=meta.model, K=K.tolist(),
                                dist=[0.0] * 5, calibration=k_source),
        captured_at=meta.captured_at,
        pose_to_facade={"cx": plane.camera.cx, "cy": plane.camera.cy, "cz": plane.camera.cz},
        theta_cam_deg=float(plane.camera.theta_deg(bounds_mm[2] / 2, bounds_mm[3] / 2)),
        theta_field_deg={"min": float(am.theta_full.min()),
                         "max": float(am.theta_full.max()), "p95": theta_p95},
        quality=quality,
        rectification=Rectification(method=plane.method,
                                    confidence=plane.confidence.value,
                                    residual_px=plane.confidence.residual_px),
        homography={"from": "image_px", "to": "rectified_px", "H": rect.H.tolist()},
    )

    facade = FacadeRecord(
        origin="bottom_left",
        bounds_mm=list(bounds_mm),
        mm_per_rectified_px=mm_per_px,
        scale=ScaleEstimate(source=scale_source, sigma_rel=sigma_rel, meets_tolerance=meets),
        plane_residual_mm={"rms": plane.confidence.residual_px * mm_per_px, "max": None},
        valid_mask=None,
    )

    return FacadeModel(
        schema_version=SCHEMA_VERSION,
        software_version=SOFTWARE_VERSION,
        coverage="partial",
        mode="auto",
        images=[record],
        facade=facade,
        elements=[],
        groups=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Оцифровка фасадного снимка: геометрическое ядро")
    parser.add_argument("image", help="путь к снимку фасада")
    parser.add_argument("--mm-per-px", type=float, required=True,
                        help="масштаб: миллиметров на пиксель")
    parser.add_argument("--scale-source", default="operator_reference",
                        choices=sorted(SCALE_PROFILE))
    parser.add_argument("-o", "--output", help="куда записать JSON (по умолчанию stdout)")
    args = parser.parse_args()

    model = process(args.image, args.mm_per_px, args.scale_source)
    payload = model.model_dump_json(indent=2)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)

    if model.images[0].quality.verdict == "reject":
        sys.stderr.write("\nснимок отбракован: " +
                         "; ".join(model.images[0].quality.reasons) + "\n")
        return 2
    return 0
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_e2e.py -v`
Ожидается: 5 passed

- [ ] **Шаг 5: Прогнать весь набор тестов**

Выполнить: `pytest -v`
Ожидается: все тесты проходят, не менее 80 штук

- [ ] **Шаг 6: Коммит**

```bash
git add facade_digitizer/pipeline/run.py tests/test_e2e.py
git commit -m "Сборка геометрического конвейера, CLI и сквозные тесты на синтетике"
```

---

## Задача 15: Модуль метрик точности

**Файлы:**
- Создать: `facade_digitizer/metrics.py`
- Тест: `tests/test_metrics.py`

**Интерфейсы:**
- Потребляет: ничего (чистые массивы)
- Предоставляет: `ErrorStats(n, rmse, p95, bias, sigma)`,
  `error_stats(measured, truth) -> ErrorStats`,
  `coverage(measured, truth, sigmas, k=1.0) -> float`,
  `mean_sharpness(sigmas) -> float`,
  `interval_score(measured, truth, sigmas, alpha=0.05) -> float`

Спецификация ставит модуль метрик в этап 0 с критерием «метрики считаются», и не зря: если
средство измерения точности появляется после эксперимента, оно подстраивается под результат.
Здесь оно пишется до того, как появятся реальные данные, и проверяется на массивах с
аналитически известным ответом.

Ключевое требование п. 12.3: покрытие интервалов бессмысленно без остроты, потому что тривиально
накручивается завышением σ. Поэтому обе величины считает один модуль и отчёт обязан содержать обе.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_metrics.py`:

```python
import numpy as np
import pytest

from facade_digitizer.metrics import (
    coverage,
    error_stats,
    interval_score,
    mean_sharpness,
)


def test_error_stats_on_known_arrays():
    truth = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    measured = truth + np.array([10.0, -10.0, 10.0, -10.0])
    st = error_stats(measured, truth)
    assert st.n == 4
    assert st.rmse == pytest.approx(10.0)
    assert st.bias == pytest.approx(0.0)
    assert st.sigma == pytest.approx(11.547, rel=1e-3)


def test_bias_is_separated_from_spread():
    """Систематическое смещение сообщается отдельно. Спецификация, п. 12.1."""
    truth = np.array([1000.0, 2000.0, 3000.0])
    measured = truth + 25.0
    st = error_stats(measured, truth)
    assert st.bias == pytest.approx(25.0)
    assert st.sigma == pytest.approx(0.0, abs=1e-9)
    assert st.rmse == pytest.approx(25.0)


def test_p95_is_reported():
    truth = np.zeros(100)
    measured = np.linspace(0.0, 100.0, 100)
    st = error_stats(measured, truth)
    assert 90.0 < st.p95 <= 100.0


def test_coverage_matches_nominal_for_correct_sigma():
    rng = np.random.default_rng(0)
    truth = np.zeros(20000)
    sigma = 10.0
    measured = rng.normal(0.0, sigma, 20000)
    got = coverage(measured, truth, np.full(20000, sigma), k=1.0)
    assert got == pytest.approx(0.683, abs=0.02)


def test_coverage_can_be_gamed_by_inflating_sigma():
    """Именно поэтому острота обязательна рядом с покрытием. Спецификация, п. 12.3."""
    rng = np.random.default_rng(1)
    truth = np.zeros(5000)
    measured = rng.normal(0.0, 10.0, 5000)
    inflated = np.full(5000, 200.0)
    assert coverage(measured, truth, inflated, k=1.0) > 0.99
    assert mean_sharpness(inflated) == pytest.approx(200.0)


def test_interval_score_penalises_both_width_and_miss():
    truth = np.array([0.0])
    tight_correct = interval_score(np.array([0.0]), truth, np.array([10.0]))
    wide_correct = interval_score(np.array([0.0]), truth, np.array([100.0]))
    tight_miss = interval_score(np.array([500.0]), truth, np.array([10.0]))
    assert tight_correct < wide_correct
    assert tight_correct < tight_miss


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        error_stats(np.array([1.0, 2.0]), np.array([1.0]))
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_metrics.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.metrics'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/metrics.py`:

```python
"""Метрики точности и калибровки неопределённости. Спецификация, раздел 12."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ErrorStats:
    n: int
    rmse: float
    p95: float
    bias: float
    sigma: float


def _check(measured: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    measured = np.asarray(measured, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    if measured.shape != truth.shape:
        raise ValueError(
            f"длины не совпадают: измерено {measured.shape}, эталон {truth.shape}"
        )
    if measured.size == 0:
        raise ValueError("пустая выборка")
    return measured, truth


def error_stats(measured: np.ndarray, truth: np.ndarray) -> ErrorStats:
    """RMSE, P95, смещение и разброс. Смещение сообщается отдельно от разброса."""
    measured, truth = _check(measured, truth)
    err = measured - truth
    bias = float(err.mean())
    return ErrorStats(
        n=int(err.size),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        p95=float(np.percentile(np.abs(err), 95)),
        bias=bias,
        sigma=float(np.std(err, ddof=1)) if err.size > 1 else 0.0,
    )


def coverage(
    measured: np.ndarray, truth: np.ndarray, sigmas: np.ndarray, k: float = 1.0
) -> float:
    """Доля эталонных значений, попавших в заявленный интервал ±k·σ.

    Бессмысленна без mean_sharpness: завышение σ поднимает покрытие до единицы.
    """
    measured, truth = _check(measured, truth)
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if sigmas.shape != measured.shape:
        raise ValueError("массив σ не совпадает по длине с выборкой")
    if np.any(sigmas <= 0):
        raise ValueError("σ должна быть положительной")
    return float(np.mean(np.abs(measured - truth) <= k * sigmas))


def mean_sharpness(sigmas: np.ndarray) -> float:
    """Средняя заявленная σ — острота. Сообщается всегда рядом с покрытием."""
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if sigmas.size == 0:
        raise ValueError("пустой массив σ")
    return float(sigmas.mean())


def interval_score(
    measured: np.ndarray, truth: np.ndarray, sigmas: np.ndarray, alpha: float = 0.05
) -> float:
    """Interval score — proper scoring rule: штрафует и ширину, и промах.

    IS = (u − l) + (2/α)(l − y)·1{y<l} + (2/α)(y − u)·1{y>u}
    Меньше — лучше. Спецификация, п. 12.3.
    """
    measured, truth = _check(measured, truth)
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if sigmas.shape != measured.shape:
        raise ValueError("массив σ не совпадает по длине с выборкой")

    z = norm.ppf(1.0 - alpha / 2.0)
    lower, upper = measured - z * sigmas, measured + z * sigmas
    width = upper - lower
    below = (2.0 / alpha) * np.clip(lower - truth, 0.0, None)
    above = (2.0 / alpha) * np.clip(truth - upper, 0.0, None)
    return float(np.mean(width + below + above))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_metrics.py -v`
Ожидается: 7 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/metrics.py tests/test_metrics.py
git commit -m "Модуль метрик точности с разделением смещения и разброса, покрытием и остротой"
```

---

## Задача 16: Проверка на реальных снимках публичного набора

**Файлы:**
- Создать: `scripts/smoke_cmp.py`
- Создать: `docs/superpowers/plans/results/geometric-core-smoke.md`

**Интерфейсы:**
- Потребляет: `process` из задачи 14
- Предоставляет: отчёт о поведении конвейера на реальных данных

Синтетика проверяет корректность математики, но не устойчивость к реальным снимкам. CMP — набор
уже ректифицированных фасадов, поэтому доверие к плоскости на нём должно быть высоким, а найденная
гомография — близкой к единичной. Это дешёвая и содержательная проверка: расхождение означает
ошибку в оценке точек схода.

- [ ] **Шаг 1: Скачать базовую часть CMP**

```bash
mkdir -p data/cmp
curl -L -o data/cmp/base.zip https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
unzip -q data/cmp/base.zip -d data/cmp/
ls data/cmp/base/*.jpg | head -5
```

- [ ] **Шаг 2: Написать скрипт прогона**

Файл `scripts/smoke_cmp.py`:

```python
"""Прогон геометрического ядра по набору CMP: сводка доверия и остаточной перспективы."""
import statistics
import sys
from pathlib import Path

import numpy as np

from facade_digitizer.pipeline.run import process


def main(folder: str, limit: int = 50) -> int:
    paths = sorted(Path(folder).glob("*.jpg"))[:limit]
    if not paths:
        print(f"не найдено изображений в {folder}")
        return 1

    confidences, verdicts, failures = [], [], []
    for p in paths:
        try:
            model = process(p, mm_per_px=3.0, scale_source="assumed_floor_height")
        except Exception as exc:
            failures.append((p.name, str(exc)))
            continue
        confidences.append(model.images[0].rectification.confidence)
        verdicts.append(model.images[0].quality.verdict)

    print(f"обработано: {len(confidences)} из {len(paths)}")
    if confidences:
        print(f"доверие к плоскости: медиана {statistics.median(confidences):.2f}, "
              f"доля выше 0.5: {np.mean(np.array(confidences) > 0.5):.0%}")
    for v in ("ok", "degraded", "reject"):
        print(f"  вердикт {v}: {verdicts.count(v)}")
    if failures:
        print(f"отказов: {len(failures)}")
        for name, err in failures[:5]:
            print(f"  {name}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/cmp/base"))
```

- [ ] **Шаг 3: Выполнить прогон**

Выполнить: `python scripts/smoke_cmp.py data/cmp/base`

Ожидается: конвейер отрабатывает без исключений на подавляющем большинстве снимков. Поскольку CMP
ректифицирован, доверие к плоскости должно быть высоким на большинстве кадров. Низкая медиана
доверия означает дефект в оценке точек схода — в этом случае остановиться и разобраться, не
переходя к следующему плану.

- [ ] **Шаг 4: Записать результат**

Файл `docs/superpowers/plans/results/geometric-core-smoke.md`: число обработанных, медиана доверия,
распределение вердиктов, перечень отказов с причинами, вывод о готовности к плану 3.

- [ ] **Шаг 5: Коммит**

```bash
git add scripts/smoke_cmp.py docs/superpowers/plans/results/geometric-core-smoke.md
git commit -m "Проверка геометрического ядра на реальных снимках набора CMP"
```

---

## Что этот план сознательно не делает

- **Не оценивает глубину на реальных снимках.** Математика (задача 4) реализована и проверена, но
  вызывается только на синтетике: ширина грани откоса приходит из детекции, которой ещё нет, а
  эталона глубины не существует до полевой кампании (план 2). Модуль `depth` как часть конвейера —
  предмет плана 4.
- **Не определяет масштаб.** `mm_per_px` передаётся снаружи параметром. Источник 1 — это клики
  оператора, то есть интерфейс из плана 3; спецификация специально переносит `scale` на тот этап.
- **Не детектирует элементы.** `elements` в выходном файле пуст по построению.
- **Не калибрует пороги качества.** `Thresholds` вынесены в параметр именно затем, чтобы их можно
  было подставить после полевой кампании; значения по умолчанию — предварительные.

## Критерий готовности плана

1. `pytest -v` — всё зелёное, не менее 80 тестов.
2. Регрессионные тесты раздела 5 спецификации (знак смещения, двухосевой угол, окклюзия) есть и
   проходят.
3. Модуль метрик написан и проверен **до** появления реальных данных — критерий этапа 0
   спецификации.
4. Калибровка по мишени реализована, профиль камеры сохраняется и имеет приоритет над EXIF.
5. `facade-digitize` отрабатывает на синтетическом снимке и выдаёт валидный JSON версии 1.1.
6. Прогон по CMP выполнен, отчёт записан, медиана доверия к плоскости объяснена.
7. Уточнение п. 5.4 спецификации (замкнутая форма) внесено и закоммичено.
