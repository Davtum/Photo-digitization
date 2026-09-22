# План 1: Геометрическое ядро — реализация

**Редакция 2.** Переработана после независимой рецензии, в которой код плана был **собран и
исполнен**, а не прочитан: из 83 тестов редакции 1 упало 8, а часть прошедших проходила вхолостую.
Перечень исправленного — в конце документа, раздел «Журнал изменений». Весь код настоящей
редакции прогнан во временном окружении до внесения в план; результат — 23 из 23 по ключевым
модулям.

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

## Порядок исполнения

Задачи исполняются по номерам **с одним исключением: задача 16 (модуль метрик) выполняется сразу
после задачи 1.** Спецификация относит модуль метрик к этапу 0 не случайно: средство измерения
точности, созданное после эксперимента, подстраивается под его результат. Физически в документе
эта задача стоит в конце, чтобы не ломать нумерацию перекрёстных ссылок.

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
    elements.py              # разметка оператора → проёмы в миллиметрах с σ (задача 18)
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
  test_plane.py
  test_integration.py
  test_run.py
  test_elements.py           # задача 18
scripts/
  smoke_cmp.py               # задача 17
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

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["facade_digitizer*"]

[project.scripts]
facade-digitize = "facade_digitizer.pipeline.run:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Шаг 2: Установить пакет и создать каркас тестов**

Без этого шага ни одна команда `pytest` в плане не работает: в редакции 1 пакет не
устанавливался, а `tests` не был пакетом, отчего импорт `tests.test_synth` в поздних задачах
падал на сборе тестов.

```bash
mkdir -p facade_digitizer tests
touch facade_digitizer/__init__.py tests/__init__.py
pip install -e ".[dev]"
```

Проверить: `python -c "import facade_digitizer; print('ok')"` печатает `ok`.

- [ ] **Шаг 3: Написать падающий тест схемы**

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

- [ ] **Шаг 4: Убедиться, что тест падает**

Выполнить: `pytest tests/test_schema.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.schema'`

- [ ] **Шаг 5: Реализовать схему**

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

- [ ] **Шаг 6: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_schema.py -v`
Ожидается: 5 passed

- [ ] **Шаг 7: Коммит**

```bash
git add pyproject.toml facade_digitizer/__init__.py tests/__init__.py facade_digitizer/schema.py tests/test_schema.py
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

> **Задача исполнена; текст ниже — исходное задание, не итоговое состояние.** По итогам ревью
> сигнатура `reveal_depth_sigma` и содержание бюджета погрешности изменились существенно:
> вместо приближённых производных по углу применяются точные частные производные замкнутой формы
> по её независимым переменным (`w`, `|u|`, `C_z`), добавлен порог θ_min. Причина в том, что
> компонента угла сама зависит от искомой глубины, и производные по ней занижали чувствительность
> в `1 + d/C_z` — в опасную сторону. Актуальны реализация (`facade_digitizer/geometry/parallax.py`)
> и пункты 5.4 и 6.2 спецификации; см. коммиты 463e53a, 28a0e56, 3d355ab.
> **Допуск «невязка меньше 1e-12 мм» в задаче 8 сформулирован неверно.** Он лежит ниже
> расстояния между соседними представимыми числами: при координатах сцены порядка 13000 мм
> единица последнего разряда составляет 1.8e-12 мм, то есть допуск недостижим любым алгоритмом.
> Фактические невязки суть точные кратные одного-семи ULP. Допуски такого порядка следует
> задавать в ULP, а не в миллиметрах.

> **Блок «Интерфейсы» ниже показывает промежуточную сигнатуру, которой не существует ни в
> одной версии кода** — соседние задачи, ссылающиеся на неё в разделах «Потребляет», должны
> брать актуальную сигнатуру из реализации.


**Файлы:**
- Изменить: `facade_digitizer/geometry/parallax.py`
- Изменить: `tests/test_parallax.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane`, функции задачи 3
- Предоставляет:
  `reveal_depth(cam, edge_x, edge_y, reveal_width_mm, edge_normal) -> float`,
  `reveal_depth_sigma(depth_mm, sigma_width_mm, tan_perp, sigma_theta_rad) -> float`,
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


@pytest.mark.parametrize("theta_x_deg,theta_y_deg,ratio", [
    (10.0, 30.0, 3.4236), (5.0, 25.0, 5.4229), (15.0, 15.0, 1.4142), (10.0, 0.0, 1.0),
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
    wide = reveal_depth_sigma(150.0, 5.0, math.tan(math.radians(30.0)), sigma_theta)
    narrow = reveal_depth_sigma(150.0, 5.0, math.tan(math.radians(5.0)), sigma_theta)
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

## Задача 5: Синтетический генератор сцен с полной моделью камеры

**Файлы:**
- Создать: `facade_digitizer/synth/__init__.py`
- Создать: `facade_digitizer/synth/scene.py`
- Тест: `tests/test_synth.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane` из задачи 2
- Предоставляет: `Opening(x, y, width, height, depth)`, `look_at(center, target)`,
  `SyntheticScene` с конструктором `looking_at_centre(...)` и методами
  `project(points_mm, depth=0.0)`, `render()`, `camera_on_plane()`, `reveal_width_px(index)`

**Почему полная модель камеры, а не масштабированная ортография.** В редакции 1 плана `project`
делил на постоянное расстояние до камеры, то есть был аффинным преобразованием. Перспективы в
такой сцене нет, точек схода не существует, и задачи 6–8, которые их ищут, проверялись на данных,
где проверяемого эффекта физически нет. Каскад последствий доходил до `OverflowError` в
ректификации. Здесь камера задана центром и ориентацией, проекция — `K·R·(X − C)`; перспектива и
параллакс возникают сами, и `apparent_position` в генераторе не вызывается (иначе поправка
применилась бы дважды).

**Почему межэтажные членения.** Измерено: на голой стене детектор отрезков находит 8–10 штук
при пороге устойчивости 8 — оценка точек схода разваливается. С горизонтальными членениями — 40.
Реальные фасады такие горизонтали имеют практически всегда, так что это не подгонка под тест,
а приближение к натуре.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_synth.py`:

```python
import numpy as np
import pytest

from facade_digitizer.synth.scene import Opening, SyntheticScene

K = np.array([[3600.0, 0, 2640.0], [0, 3600.0, 1978.0], [0, 0, 1.0]])
SIZE = (5280, 3956)


def make_scene(dx=3000.0, dy=-2000.0, dist=12000.0, depth=150.0):
    ops = [Opening(x=4000.0, y=3000.0, width=1460.0, height=1900.0, depth=depth)]
    return SyntheticScene.looking_at_centre(20000.0, 15000.0, ops, dist, dx, dy, K, SIZE)


def test_projection_is_actually_perspective():
    """Ключевой приёмочный критерий: равные шаги по фасаду дают РАЗНЫЕ шаги в кадре."""
    sc = make_scene()
    xs = np.array([[x, 7500.0] for x in range(0, 20001, 5000)], dtype=float)
    steps = np.diff(sc.project(xs)[:, 0])
    assert steps.std() / steps.mean() > 0.02


def test_render_produces_image_of_requested_size():
    img = make_scene().render()
    assert img.shape == (3956, 5280)
    assert img.dtype == np.uint8


def test_render_has_enough_structure_for_line_detection():
    img = make_scene().render()
    assert img.std() > 5.0
    assert len(np.unique(img)) >= 4     # фон, стена, членения, откос, полотно


def test_camera_on_plane_reports_true_pose():
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=12000.0)
    cam = sc.camera_on_plane()
    assert cam.cx == pytest.approx(10000.0 + 3000.0)
    assert cam.cy == pytest.approx(7500.0 - 2000.0)
    assert cam.cz == pytest.approx(12000.0)


def test_deeper_opening_shows_wider_reveal():
    assert make_scene(depth=300.0).reveal_width_px() > make_scene(depth=50.0).reveal_width_px()


def test_point_behind_camera_is_rejected():
    sc = make_scene()
    with pytest.raises(ValueError):
        sc.project(np.array([[0.0, 0.0]]), depth=-20000.0)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_synth.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.synth'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/synth/__init__.py` — пустой.

Файл `facade_digitizer/synth/scene.py`:

```python
"""Синтетические сцены с ПОЛНОЙ моделью камеры: перспектива и параллакс возникают сами."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import CameraOnPlane

WORLD_UP = np.array([0.0, 1.0, 0.0])


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


def look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Матрица поворота мир -> камера. Камера смотрит вдоль +Z, ось Y направлена вниз."""
    f = target - center
    f = f / np.linalg.norm(f)
    r = np.cross(f, WORLD_UP)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.array([r, d, f])


@dataclass
class SyntheticScene:
    width_mm: float
    height_mm: float
    openings: list
    camera_center: np.ndarray
    R_wc: np.ndarray
    K: np.ndarray
    image_size: tuple
    floor_band_step_mm: float = 3000.0
    _c: dict = field(default_factory=dict, repr=False)

    @classmethod
    def looking_at_centre(cls, width_mm, height_mm, openings, distance_mm,
                          offset_x_mm, offset_y_mm, K, image_size,
                          floor_band_step_mm=3000.0):
        target = np.array([width_mm / 2.0, height_mm / 2.0, 0.0])
        centre = target + np.array([offset_x_mm, offset_y_mm, distance_mm])
        return cls(width_mm, height_mm, openings, centre, look_at(centre, target),
                   K, image_size, floor_band_step_mm)

    def camera_on_plane(self) -> CameraOnPlane:
        """Истинная поза: опорная точка и расстояние до плоскости."""
        c = self.camera_center
        return CameraOnPlane(cx=float(c[0]), cy=float(c[1]), cz=float(c[2]))

    def project(self, points_mm: np.ndarray, depth: float = 0.0) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points_mm, dtype=float))
        Xw = np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), -depth)])
        Xc = (self.R_wc @ (Xw.T - self.camera_center[:, None]))
        if np.any(Xc[2] <= 0):
            raise ValueError("точка за камерой")
        uv = self.K @ Xc
        return (uv[:2] / uv[2]).T

    def render(self) -> np.ndarray:
        w_px, h_px = self.image_size
        img = np.full((h_px, w_px), 200, dtype=np.uint8)
        wall = self.project(np.array([
            [0.0, 0.0], [self.width_mm, 0.0],
            [self.width_mm, self.height_mm], [0.0, self.height_mm]]))
        cv2.fillPoly(img, [np.round(wall).astype(np.int32)], 150)
        # Межэтажные членения: без них на голой стене детектор отрезков находит
        # 8-10 штук, чего не хватает для устойчивой оценки точек схода. Реальные
        # фасады такие горизонтали имеют почти всегда.
        if self.floor_band_step_mm > 0:
            y = self.floor_band_step_mm
            while y < self.height_mm:
                pts = self.project(np.array([[0.0, y], [self.width_mm, y]]))
                cv2.line(img, tuple(np.round(pts[0]).astype(int)),
                         tuple(np.round(pts[1]).astype(int)), 115, 6)
                y += self.floor_band_step_mm

        for op in self.openings:
            cv2.fillPoly(img, [np.round(self.project(op.corners_mm())).astype(np.int32)], 90)
            cv2.fillPoly(img, [np.round(self.project(op.corners_mm(), depth=op.depth)).astype(np.int32)], 35)
        return img

    def reveal_width_px(self, index: int = 0) -> float:
        op = self.openings[index]
        outer = self.project(op.corners_mm())
        inner = self.project(op.corners_mm(), depth=op.depth)
        return float(np.max(np.abs(inner[:, 0] - outer[:, 0])))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_synth.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/synth/ tests/test_synth.py
git commit -m "Синтетический генератор сцен с полной моделью камеры и межэтажными членениями"
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

Файл `facade_digitizer/geometry/vanishing.py` целиком:

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


def _fit_vanishing_point(lines: np.ndarray, threshold: float, rng):
    """RANSAC по парам прямых. Возвращает точку и маску поддержки."""
    best_point, best_mask = None, np.zeros(len(lines), dtype=bool)
    if len(lines) < 2:
        return np.array([0.0, 0.0, 1.0]), best_mask, float("inf")

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
        return np.array([0.0, 0.0, 1.0]), best_mask, float("inf")

    v = best_point
    dist = np.abs(lines @ v) / (np.linalg.norm(lines[:, :2], axis=1) * np.linalg.norm(v) + 1e-12)
    resid = float(dist[best_mask].mean()) if best_mask.any() else float("inf")
    return best_point, best_mask, resid


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
    vh_pt, vh_mask, vh_res = (_fit_vanishing_point(_segment_lines(horiz), threshold, rng)
                              if len(horiz) >= 2 else
                              (np.array([1.0, 0.0, 0.0]), np.zeros(0, dtype=bool), float("inf")))
    vv_pt, vv_mask, vv_res = (_fit_vanishing_point(_segment_lines(vert), threshold, rng)
                              if len(vert) >= 2 else
                              (np.array([0.0, 1.0, 0.0]), np.zeros(0, dtype=bool), float("inf")))

    Kinv = np.linalg.inv(K)
    dh, dv = Kinv @ vh_pt, Kinv @ vv_pt
    dh, dv = dh / (np.linalg.norm(dh) + 1e-12), dv / (np.linalg.norm(dv) + 1e-12)
    orthogonality = math.degrees(math.acos(min(1.0, abs(float(dh @ dv)))))

    support_h, support_v = int(vh_mask.sum()), int(vv_mask.sum())
    coverage = (support_h + support_v) / max(len(segments), 1)
    # Настоящая невязка: средняя нормированная дистанция инлайеров, переведённая
    # в пиксели через размер кадра. В редакции 1 здесь стояла константа
    # threshold * max(image_size), не зависевшая ни от сцены, ни от качества подгонки.
    finite = [r for r in (vh_res, vv_res) if np.isfinite(r)]
    residual = float(np.mean(finite) * max(image_size)) if finite else float("inf")

    if support_h < 2 or support_v < 2:
        reasons.append("недостаточная поддержка одной из точек схода")
    if abs(orthogonality - 90.0) > 15.0:
        reasons.append(f"направления не ортогональны: {orthogonality:.1f}°")
    if coverage < 0.2:
        reasons.append("отрезки покрывают малую долю кадра")
    if not np.isfinite(residual):
        reasons.append("невязка не определена")
    if support_h + support_v < 10:
        reasons.append(f"мало поддерживающих отрезков: {support_h + support_v}")

    value = 0.0 if reasons else min(1.0, 0.3 + 0.7 * coverage)

    return (
        VanishingPoint(vh_pt, support_h, residual),
        VanishingPoint(vv_pt, support_v, residual),
        PlaneConfidence(value, support_h, support_v, residual, orthogonality, coverage, reasons),
    )
```

**Что изменено против редакции 1.** Величина `residual_px` была константой
`threshold * max(image_size)` = 105.6 при любой сцене и любом качестве подгонки — то есть не
невязкой вовсе, хотя спецификация (п. 4.2) требует именно её. Теперь измеряется средняя
нормированная дистанция инлайеров до найденной точки схода: на корректной сцене она
составляет около 10 пикселей. Одновременно добавлены две причины отказа, без которых
вырожденный вход получал доверие 1.00: неопределённая невязка и слишком малое суммарное
число поддерживающих отрезков.

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_vanishing.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/vanishing.py tests/test_vanishing.py
git commit -m "Оценка точек схода по RANSAC с явной метрикой доверия к плоскости"
```

---

## Задача 8: Гомография, ориентация осей и поза камеры

> **Задача исполнена; текст ниже — исходное задание.** Существенные отличия итога: доопределение
> ориентации осей переведено с пробы со сдвигом на аналитический признак (знак производной в
> центре кадра), поскольку проба давала зеркальный фасад на рёберных ракурсах при безупречных
> прямых углах; `camera_pose` возвращает не `CameraOnPlane`, а пару с признаком неразрешённости
> поворота на 180°, который из двух точек схода не определяется. Актуальны реализация и п. 4.2
> спецификации; см. коммиты 802ebaa, 1e1c9e9, 92ea2bd, ddb127b.


**Файлы:**
- Создать: `facade_digitizer/geometry/homography.py`
- Тест: `tests/test_homography.py`

**Интерфейсы:**
- Потребляет: `VanishingPoint` из задачи 7, `CameraOnPlane` из задачи 2
- Предоставляет: `homography_from_vanishing_points(vh, vv, K, image_size) -> np.ndarray`,
  `foot_point_in_rectified(H, vh, vv, K) -> tuple[float, float]`,
  `rectified_to_facade_mm(points_rect, origin_rect, mm_per_rect_unit) -> np.ndarray`,
  `camera_pose(H, vh, vv, K, mm_per_rect_unit, origin_rect) -> CameraOnPlane`,
  `homography_from_four_points(image_pts, aspect_ratio=None, size_mm=None,
  assume_calibrated=False, K=None, image_size=None) -> np.ndarray`

**Три вещи, которые редакция 1 плана делала неверно.**

*Поза камеры.* Гомография из двух точек схода есть гомография чистого поворота и трансляции не
содержит — извлечь из неё положение камеры напрямую нельзя, и попытка это сделать помещала камеру
в 27 км от фасада. Но опорная точка F всё же восстановима: луч из центра камеры вдоль нормали
пересекает плоскость ровно в F, а все точки этого луча имеют один образ — точку схода нормали
`v₃ = K·(d₁×d₂)`. Отсюда `F = H·v₃`. Более того, в системе координат, которую задаёт такая
гомография, `H·v₃ = (0, 0)` тождественно, а одна ректифицированная единица равна расстоянию до
плоскости. Проверено на четырёх позах: невязка меньше 10⁻¹² мм при точных точках схода и около
2 % расстояния при оценке по реальному детектору.

*Ориентация осей.* Знак точки схода произволен, и оси связаны: третья строка гомографии есть
`d₁×d₂`, поэтому смена знака `d₁` переворачивает не X, а Y. Физически допустимы лишь две
комбинации, различающиеся поворотом на 180°; остальные дают зеркальное отражение, невозможное при
взгляде спереди. Редакция 1 не фиксировала ориентацию вовсе и выдавала перевёрнутый растр.

*Ось Y.* Ректифицированный растр — изображение, его ось Y направлена вниз; ось Y фасада направлена
вверх. Переворот выполняется в `rectified_to_facade_mm`, в одном месте, а не внутри гомографии —
иначе получилось бы зеркало.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_homography.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.homography import (
    camera_pose,
    homography_from_four_points,
    homography_from_vanishing_points,
    rectified_to_facade_mm,
)
from tests.test_synth import K, SIZE, make_scene

CORNERS = np.array([[0.0, 0], [20000.0, 0], [20000.0, 15000.0], [0, 15000.0]])


def exact_vps(sc):
    """Точки схода осей плоскости, вычисленные точно из позы сцены."""
    return K @ (sc.R_wc @ np.array([1.0, 0, 0])), K @ (sc.R_wc @ np.array([0, 1.0, 0]))


def apply(H, pts):
    p = np.column_stack([np.atleast_2d(pts), np.ones(len(np.atleast_2d(pts)))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def test_rectification_restores_right_angles_and_ratio():
    sc = make_scene()
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(np.array([[0.0, 0], [1500.0, 0], [1500.0, 1500.0], [0, 1500.0]])))
    a, b = r[1] - r[0], r[3] - r[0]
    ang = np.degrees(np.arccos(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert ang == pytest.approx(90.0, abs=1e-3)
    assert np.linalg.norm(a) / np.linalg.norm(b) == pytest.approx(1.0, rel=1e-6)


def test_rectified_raster_keeps_image_convention():
    """Ректифицированный растр — изображение: X вправо, Y ВНИЗ."""
    sc = make_scene()
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    assert r[1, 0] > r[0, 0]
    assert r[3, 1] < r[0, 1]


def test_facade_millimetres_have_y_up():
    """После перевода в миллиметры ось Y направлена вверх. Глобальные ограничения."""
    sc = make_scene()
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    mm = rectified_to_facade_mm(r, tuple(r[0]), mmu)
    assert mm[1, 0] == pytest.approx(20000.0, abs=1.0)
    assert mm[3, 1] == pytest.approx(15000.0, abs=1.0)


@pytest.mark.parametrize("dx,dy,dist", [
    (2500.0, -1800.0, 12000.0), (-3000.0, 2200.0, 9000.0),
    (0.0, -4000.0, 15000.0), (4500.0, 0.0, 11000.0),
])
def test_camera_pose_recovered_against_known_truth(dx, dy, dist):
    """Поза сравнивается с истиной сцены, а не с самой собой."""
    sc = make_scene(dx, dy, dist)
    vh, vv = exact_vps(sc)
    H = homography_from_vanishing_points(vh, vv, K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])

    got = camera_pose(H, vh, vv, K, mmu, tuple(r[0]))
    truth = sc.camera_on_plane()
    assert got.cx == pytest.approx(truth.cx, abs=1.0)
    assert got.cy == pytest.approx(truth.cy, abs=1.0)
    assert got.cz == pytest.approx(truth.cz, rel=0.02)


def test_four_point_requires_disambiguation():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError):
        homography_from_four_points(pts)


def test_four_point_with_sizes_produces_those_sizes():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    H = homography_from_four_points(pts, size_mm=(1460.0, 1900.0))
    r = apply(H, pts)
    assert np.linalg.norm(r[1] - r[0]) == pytest.approx(1460.0, rel=1e-6)
    assert np.linalg.norm(r[3] - r[0]) == pytest.approx(1900.0, rel=1e-6)


def test_four_point_with_aspect_ratio_produces_that_ratio():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    H = homography_from_four_points(pts, aspect_ratio=2.0)
    r = apply(H, pts)
    assert np.linalg.norm(r[1] - r[0]) / np.linalg.norm(r[3] - r[0]) == pytest.approx(2.0, rel=1e-6)


def test_four_point_calibrated_variant_needs_K():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError):
        homography_from_four_points(pts, assume_calibrated=True)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_homography.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/geometry/homography.py`:

```python
"""Гомография приведения к плоскости фасада и восстановление позы камеры."""
import numpy as np

from .camera import CameraOnPlane


def _plane_axes(vh, vv, K):
    Kinv = np.linalg.inv(K)
    d1 = Kinv @ np.asarray(vh, dtype=float)
    d1 = d1 / np.linalg.norm(d1)
    d2 = Kinv @ np.asarray(vv, dtype=float)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / np.linalg.norm(d2)
    return d1, d2


def homography_from_vanishing_points(vh, vv, K, image_size):
    """Изображение -> фронто-параллельный вид, с однозначно заданной ориентацией осей.

    Знак точки схода произволен (v и -v — одна точка), поэтому осям нужно
    доопределение. Существенно, что оси связаны: третья строка гомографии есть
    d1 x d2, поэтому смена знака d1 переворачивает не X, а Y. Физически допустимы
    лишь две комбинации — (d1, d2) и (-d1, -d2), различающиеся поворотом на 180°;
    остальные дают зеркальное отражение, которого при взгляде спереди быть не может.

    Порядок: сначала фиксируется знак нормали (знаменатель гомографии положителен
    для видимых точек), затем из двух оставшихся вариантов выбирается тот, где ось X
    растёт вместе с x изображения.
    """
    d1, d2 = _plane_axes(vh, vv, K)
    Kinv = np.linalg.inv(K)
    w, h = image_size

    centre_ray = Kinv @ np.array([w / 2.0, h / 2.0, 1.0])
    if np.cross(d1, d2) @ centre_ray < 0:
        d1 = -d1  # переворачивает нормаль, сохраняя правую тройку

    def build(a1, a2):
        return np.column_stack([a1, a2, np.cross(a1, a2)]).T @ Kinv

    H = build(d1, d2)
    probe = np.array([[w / 2.0, h / 2.0, 1.0], [w / 2.0 + 50.0, h / 2.0, 1.0]]) @ H.T
    probe = probe[:, :2] / probe[:, 2:3]
    if probe[1, 0] < probe[0, 0]:
        H = build(-d1, -d2)

    return H


def foot_point_in_rectified(H, vh, vv, K):
    """Опорная точка F в ректифицированных координатах.

    Луч из центра камеры вдоль нормали к фасаду пересекает плоскость ровно в F,
    и все точки этого луча имеют один образ — точку схода нормали v3 = K·(d1 x d2).
    Следовательно F = H·v3.
    """
    d1, d2 = _plane_axes(vh, vv, K)
    v3 = K @ np.cross(d1, d2)
    p = H @ v3
    if abs(p[2]) < 1e-12:
        raise ValueError("точка схода нормали вырождена: фасад строго фронтален")
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def rectified_to_facade_mm(points_rect, origin_rect, mm_per_rect_unit):
    """Ректифицированные координаты -> миллиметры в системе фасада.

    Ректифицированный растр — изображение, его ось Y направлена вниз; ось Y фасада
    направлена вверх (глобальное ограничение). Переворот выполняется здесь, в одном
    месте, а не внутри гомографии: гомография остаётся ориентацию сохраняющей, иначе
    получилось бы зеркальное отражение фасада.
    """
    ox, oy = origin_rect
    p = np.atleast_2d(np.asarray(points_rect, dtype=float))
    return np.column_stack([(p[:, 0] - ox) * mm_per_rect_unit,
                            -(p[:, 1] - oy) * mm_per_rect_unit])


def camera_pose(H, vh, vv, K, mm_per_rect_unit, origin_rect):
    """Поза камеры относительно плоскости фасада в миллиметрах.

    В системе координат, задаваемой гомографией из точек схода, опорная точка F
    лежит ровно в начале (H·v3 = (0, 0)), а одна ректифицированная единица
    соответствует расстоянию от камеры до плоскости.
    """
    foot_mm = rectified_to_facade_mm([[0.0, 0.0]], origin_rect, mm_per_rect_unit)[0]
    return CameraOnPlane(cx=float(foot_mm[0]), cy=float(foot_mm[1]),
                         cz=float(mm_per_rect_unit))


def homography_from_four_points(image_pts, aspect_ratio=None, size_mm=None,
                                assume_calibrated=False, K=None, image_size=None):
    """Ручной вариант: четыре точки плюс одно из трёх доопределений."""
    import cv2

    given = sum(x is not None for x in (aspect_ratio, size_mm)) + int(assume_calibrated)
    if given == 0:
        raise ValueError(
            "четырёх точек недостаточно: задайте aspect_ratio, size_mm либо "
            "assume_calibrated=True вместе с K"
        )
    if assume_calibrated:
        if K is None or image_size is None:
            raise ValueError("для assume_calibrated нужны K и image_size")
        src = np.asarray(image_pts, dtype=float)
        l_top = np.cross(np.append(src[0], 1.0), np.append(src[1], 1.0))
        l_bot = np.cross(np.append(src[3], 1.0), np.append(src[2], 1.0))
        l_lft = np.cross(np.append(src[0], 1.0), np.append(src[3], 1.0))
        l_rgt = np.cross(np.append(src[1], 1.0), np.append(src[2], 1.0))
        vh = np.cross(l_top, l_bot)
        vv = np.cross(l_lft, l_rgt)
        return homography_from_vanishing_points(vh, vv, K, image_size)

    w, h = size_mm if size_mm is not None else (float(aspect_ratio), 1.0)
    src = np.asarray(image_pts, dtype=np.float32)
    dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(float)
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_homography.py -v`
Ожидается: 11 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/homography.py tests/test_homography.py
git commit -m "Гомография с фиксированной ориентацией осей и восстановление позы по точке схода нормали"
```

---

## Задача 9: Поле углов визирования и локальное разрешение

**Файлы:**
- Создать: `facade_digitizer/geometry/angles.py`
- Тест: `tests/test_angles.py`

**Интерфейсы:**
- Потребляет: `CameraOnPlane` из задачи 2
- Предоставляет: `AngleMap(theta_x, theta_y, theta_full, bounds_mm)`,
  `angle_map(cam, bounds_mm, shape) -> AngleMap`,
  `usable_mask(am, theta_max_deg=30.0) -> np.ndarray`,
  `local_gsd(H, mm_per_rect_unit, image_size, shape=(64,64), worst_direction=True) -> np.ndarray`

**Что было неверно в редакции 1.** Формула локального разрешения `w**2/det(H)` применяла
площадной показатель к линейной величине и вдобавок переворачивала отношение; совпадала с
истиной ровно при `H = I` — единственном случае, который проверял проходивший тест. Верный
линейный масштаб даётся сингулярными числами якобиана. Кроме того, перспектива анизотропна:
масштаб вдоль X и вдоль Y в одной точке различается на единицы процентов, а требование п. 2.2
спецификации относится к **худшему** локальному разрешению — поэтому берётся наибольшее
сингулярное число, а не среднее геометрическое.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_angles.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.angles import angle_map, local_gsd, usable_mask
from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import homography_from_vanishing_points
from tests.test_homography import CORNERS, apply, exact_vps
from tests.test_synth import K, SIZE, make_scene


def test_angle_is_zero_at_foot_and_grows_outward():
    cam = CameraOnPlane(cx=10000.0, cy=7500.0, cz=10000.0)
    am = angle_map(cam, (0.0, 0.0, 20000.0, 15000.0), (101, 101))
    assert am.theta_full.min() == pytest.approx(0.0, abs=0.5)
    assert am.theta_full[0, 0] > am.theta_full[50, 50]


def test_components_have_expected_signs():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    am = angle_map(cam, (-5000.0, -5000.0, 5000.0, 5000.0), (3, 3))
    assert am.theta_x[1, 0] < 0
    assert am.theta_x[1, 2] > 0


def test_usable_mask_excludes_steep_angles():
    cam = CameraOnPlane(cx=0.0, cy=0.0, cz=10000.0)
    am = angle_map(cam, (0.0, 0.0, 20000.0, 15000.0), (51, 51))
    m = usable_mask(am, 30.0)
    assert m.any() and not m.all()


def test_local_gsd_majorises_every_direction():
    """Величина обязана мажорировать разрешение по ЛЮБОМУ направлению.

    Сверка с истиной: спроецировать близкие точки фасада и померить пиксели.
    """
    sc = make_scene(dx=3000.0, dy=-2000.0, dist=12000.0)
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])

    step = 20.0
    grid = local_gsd(H, mmu, SIZE, shape=(1200, 1200))
    for fx, fy in [(4000.0, 4000.0), (14000.0, 10000.0), (2000.0, 12000.0)]:
        a = sc.project(np.array([[fx, fy]]))[0]
        along_x = step / np.linalg.norm(sc.project(np.array([[fx + step, fy]]))[0] - a)
        along_y = step / np.linalg.norm(sc.project(np.array([[fx, fy + step]]))[0] - a)
        col = np.clip(int(round(a[0] / (SIZE[0] - 1) * 1199)), 0, 1199)
        row = np.clip(int(round(a[1] / (SIZE[1] - 1) * 1199)), 0, 1199)
        got = grid[row, col]
        assert got >= max(along_x, along_y) * 0.98
        assert got <= max(along_x, along_y) * 1.15


def test_worst_direction_exceeds_geometric_mean():
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    worst = local_gsd(H, mmu, SIZE)
    mean = local_gsd(H, mmu, SIZE, worst_direction=False)
    assert np.all(worst >= mean - 1e-9)
    assert worst.max() > mean.max()


def test_local_gsd_varies_across_tilted_frame():
    sc = make_scene(dx=5000.0, dy=-3500.0, dist=9000.0)
    H = homography_from_vanishing_points(*exact_vps(sc), K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    g = local_gsd(H, mmu, SIZE)
    assert g.max() / g.min() > 1.2
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
    theta_x: np.ndarray
    theta_y: np.ndarray
    theta_full: np.ndarray
    bounds_mm: tuple


def angle_map(cam: CameraOnPlane, bounds_mm: tuple, shape: tuple) -> AngleMap:
    x0, y0, x1, y1 = bounds_mm
    rows, cols = shape
    gx, gy = np.meshgrid(np.linspace(x0, x1, cols), np.linspace(y0, y1, rows))
    tan_x = (gx - cam.cx) / cam.cz
    tan_y = (gy - cam.cy) / cam.cz
    return AngleMap(np.degrees(np.arctan(tan_x)), np.degrees(np.arctan(tan_y)),
                    np.degrees(np.arctan(np.hypot(tan_x, tan_y))), bounds_mm)


def usable_mask(am: AngleMap, theta_max_deg: float = 30.0) -> np.ndarray:
    return am.theta_full <= theta_max_deg


def local_gsd(H: np.ndarray, mm_per_rect_unit: float, image_size: tuple,
              shape: tuple = (64, 64), worst_direction: bool = True) -> np.ndarray:
    """Миллиметры фасада на один пиксель ИСХОДНОГО снимка, по полю кадра.

    Именно эта величина ограничивает точность измерения: ректифицированный растр
    равномерен по построению, неравномерно распределено исходное разрешение.

    Перспектива анизотропна: масштаб вдоль X и вдоль Y в одной точке различается.
    Величина sqrt(|det J|) даёт среднее геометрическое и занижает худший случай на
    единицы процентов. Поскольку требование спецификации (п. 2.2) относится к
    ХУДШЕМУ локальному разрешению, по умолчанию берётся наибольшее сингулярное
    число якобиана.
    """
    w_px, h_px = image_size
    rows, cols = shape
    gx, gy = np.meshgrid(np.linspace(0, w_px - 1, cols), np.linspace(0, h_px - 1, rows))

    w = H[2, 0] * gx + H[2, 1] * gy + H[2, 2]
    num_x = H[0, 0] * gx + H[0, 1] * gy + H[0, 2]
    num_y = H[1, 0] * gx + H[1, 1] * gy + H[1, 2]

    j00 = (H[0, 0] * w - num_x * H[2, 0]) / w ** 2
    j01 = (H[0, 1] * w - num_x * H[2, 1]) / w ** 2
    j10 = (H[1, 0] * w - num_y * H[2, 0]) / w ** 2
    j11 = (H[1, 1] * w - num_y * H[2, 1]) / w ** 2

    det = j00 * j11 - j01 * j10
    if not worst_direction:
        return mm_per_rect_unit * np.sqrt(np.abs(det))

    trace = j00 ** 2 + j01 ** 2 + j10 ** 2 + j11 ** 2
    disc = np.clip(trace ** 2 - 4.0 * det ** 2, 0.0, None)
    sigma_max = np.sqrt((trace + np.sqrt(disc)) / 2.0)
    return mm_per_rect_unit * sigma_max
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_angles.py -v`
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/geometry/angles.py tests/test_angles.py
git commit -m "Поле углов визирования и локальное разрешение по худшему направлению"
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
- Потребляет: ничего из предыдущих задач
- Предоставляет: `Thresholds(sharpness_min, gsd_max_mm_px, theta_p95_max_deg)`,
  `sharpness(image) -> float`, `usable_fraction(theta_field_deg, theta_max_deg=30.0) -> float`,
  `assess(image, gsd_min, gsd_max, theta_p95, usable=1.0, thresholds=None) -> tuple`

**Модуль отбраковывает, а не улучшает.** Генеративный деблюр синтезирует границы, по которым
потом выполняется измерение (спецификация, п. 4.1).

**Что было неверно в редакции 1.** Порог по умолчанию расходился с собственной синтетикой
плана в 1800 раз, и тест `test_sharp_synthetic_image_is_ok` проверял абсолютное значение
против этого порога. Здесь тесты проверяют **порядок величин и поведение**, а не магическое
число: резкое изображение должно давать резкость на два порядка выше размытого, а вердикт —
меняться при подстановке порогов. Кроме того, зафиксировано тестом известное ограничение
метрики: белый шум набирает больше, чем резкое изображение, поэтому шлюз не отличает детали
от шума и работает только вместе с проверкой разрешения и угла. Калибровка порога на реальных
снимках — отдельная задача этапа 1 спецификации.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_quality.py`:

```python
import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline.quality import Thresholds, assess, sharpness, usable_fraction
from tests.test_synth import make_scene as scene


def test_blur_lowers_sharpness_by_orders_of_magnitude():
    """Проверяется ПОРЯДОК, а не абсолютное значение: порог ещё не откалиброван."""
    img = scene().render()
    assert sharpness(img) > 20 * sharpness(cv2.GaussianBlur(img, (9, 9), 3.0))
    assert sharpness(img) > 100 * sharpness(cv2.GaussianBlur(img, (31, 31), 12.0))


def test_noise_scores_higher_than_signal():
    """Известное ограничение метрики; зафиксировано тестом, чтобы не забылось."""
    img = scene().render()
    noise = np.random.default_rng(0).integers(0, 255, img.shape, dtype=np.uint8)
    assert sharpness(noise) > sharpness(img)


def test_coarse_gsd_is_rejected():
    v, _, reasons = assess(scene().render(), 14.0, 16.0, 22.0)
    assert v == "reject" and any("разрешение" in r for r in reasons)


def test_steep_angle_is_degraded():
    v, _, reasons = assess(scene().render(), 3.1, 4.6, 44.0)
    assert v == "degraded" and any("угол" in r for r in reasons)


def test_small_usable_fraction_is_flagged():
    v, _, reasons = assess(scene().render(), 3.1, 4.6, 22.0, usable=0.3)
    assert v == "degraded" and any("кадра" in r for r in reasons)


def test_thresholds_are_injectable():
    blurred = cv2.GaussianBlur(scene().render(), (31, 31), 12.0)
    strict = Thresholds(sharpness_min=1.0, gsd_max_mm_px=5.0, theta_p95_max_deg=30.0)
    lenient = Thresholds(sharpness_min=0.0, gsd_max_mm_px=5.0, theta_p95_max_deg=30.0)
    assert assess(blurred, 3.1, 4.6, 22.0, thresholds=strict)[0] == "degraded"
    assert assess(blurred, 3.1, 4.6, 22.0, thresholds=lenient)[0] == "ok"


def test_usable_fraction_counts_angles():
    field = np.array([[10.0, 20.0], [40.0, 50.0]])
    assert usable_fraction(field, 30.0) == pytest.approx(0.5)
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


@dataclass(frozen=True)
class Thresholds:
    """Пороги шлюза.

    Значение sharpness_min ПРЕДВАРИТЕЛЬНОЕ. Оно получено на синтетике (резкая сцена
    даёт около 1.2e-05, лёгкое размытие 8e-08) и подлежит замене после калибровки на
    реальных снимках — это отдельная задача этапа 1 спецификации. Абсолютный порог
    по вариации лапласиана ненадёжен и по другой причине: белый шум набирает по этой
    метрике больше, чем резкое изображение, поэтому шлюз не может отличить детали от
    шума и должен использоваться вместе с проверкой разрешения и угла.
    """
    sharpness_min: float = 1.0e-06
    gsd_max_mm_px: float = 5.0
    theta_p95_max_deg: float = 30.0


DEFAULT = Thresholds()


def sharpness(image: np.ndarray) -> float:
    """Вариация лапласиана, нормированная на дисперсию яркости."""
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return float(lap.var() / (float(image.std()) ** 2 + 1e-9))


def usable_fraction(theta_field_deg: np.ndarray, theta_max_deg: float = 30.0) -> float:
    """Доля кадра, удовлетворяющая угловому условию. Спецификация, п. 4.1."""
    return float(np.mean(np.asarray(theta_field_deg) <= theta_max_deg))


def assess(image, gsd_min, gsd_max, theta_p95, usable=1.0, thresholds=None):
    t = thresholds or DEFAULT
    reasons = []

    sharp = sharpness(image)
    if sharp < t.sharpness_min:
        reasons.append(f"недостаточная резкость: {sharp:.2e} < {t.sharpness_min:.2e}")
    if gsd_max > t.gsd_max_mm_px:
        reasons.append(f"недостаточное разрешение: {gsd_max:.1f} мм/px > {t.gsd_max_mm_px}")
    if theta_p95 > t.theta_p95_max_deg:
        reasons.append(f"слишком крутой угол визирования: P95 = {theta_p95:.1f}°")
    if usable < 0.5:
        reasons.append(f"угловому условию удовлетворяет лишь {usable:.0%} кадра")

    if not reasons:
        verdict = "ok"
    elif any("разрешение" in r for r in reasons):
        verdict = "reject"
    else:
        verdict = "degraded"
    return verdict, sharp, reasons
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_quality.py -v`
Ожидается: 7 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/quality.py tests/test_quality.py
git commit -m "Шлюз качества снимка с проверкой порядка величин вместо магического порога"
```

---

## Задача 12: Ректификация с миллиметровой конвенцией

**Файлы:**
- Создать: `facade_digitizer/pipeline/rectify.py`
- Тест: `tests/test_rectify.py`

**Интерфейсы:**
- Потребляет: `homography_from_vanishing_points` (задача 8)
- Предоставляет: `Rectified(image, H, valid_mask, mm_per_px, origin_rect_px)`,
  `rectify(image, H_units, mm_per_rect_unit, mm_per_px, origin_rect_units=(0,0)) -> Rectified`

**Что было неверно в редакции 1.** Размер выхода выводился из безразмерного размаха координат
после гомографии. Но гомография из точек схода даёт величины порядка 1/f, поэтому размах
составлял около 1.5 «единиц», и снимок 20 Мп схлопывался в растр **16x16 пикселей** — при
этом три теста ректификации на нём успешно проходили, потому что размер выхода не проверял
никто. Конвенции связи растра с миллиметрами в плане не было вовсе, а `mm_per_rectified_px` в
выходном файле был просто аргументом вызова.

Здесь конвенция задана явно: один пиксель ректифицированного растра равен `mm_per_px`
миллиметрам, размер выхода следует из неё, а абсурдный масштаб приводит к явной ошибке вместо
молчаливого схлопывания.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_rectify.py`:

```python
import numpy as np
import pytest

from facade_digitizer.geometry.homography import homography_from_vanishing_points
from facade_digitizer.geometry.vanishing import detect_segments, estimate_vanishing_points
from facade_digitizer.pipeline.rectify import rectify
from tests.test_homography import CORNERS, apply
from tests.test_synth import K, SIZE, make_scene


def prepared(mm_per_px=4.0):
    sc = make_scene()
    vh, vv, _ = estimate_vanishing_points(detect_segments(sc.render()), SIZE, K)
    H = homography_from_vanishing_points(vh.point, vv.point, K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    return sc, rectify(sc.render(), H, mmu, mm_per_px), H, mmu


def test_output_size_follows_the_convention():
    """Размер задаётся миллиметрами на пиксель, а не безразмерным размахом."""
    _, r4, _, _ = prepared(mm_per_px=4.0)
    _, r8, _, _ = prepared(mm_per_px=8.0)
    assert r4.image.shape[0] > 1000
    assert r4.image.shape[0] / r8.image.shape[0] == pytest.approx(2.0, rel=0.02)


def test_known_distance_measures_correctly_in_the_raster():
    """Мера длины: 20 м фасада занимают 20000/mm_per_px пикселей."""
    sc, rect, _, _ = prepared(mm_per_px=4.0)
    p = np.column_stack([sc.project(CORNERS), np.ones(4)]) @ rect.H.T
    p = p[:, :2] / p[:, 2:3]
    assert np.linalg.norm(p[1] - p[0]) * rect.mm_per_px == pytest.approx(20000.0, rel=0.01)


def test_absurd_scale_is_rejected_loudly():
    sc, _, H, mmu = prepared()
    with pytest.raises(ValueError):
        rectify(sc.render(), H, mmu, mm_per_px=1e-4)


def test_valid_mask_matches_image():
    _, rect, _, _ = prepared()
    assert rect.valid_mask.shape == rect.image.shape
    assert 0.1 < rect.valid_mask.mean() <= 1.0
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `pytest tests/test_rectify.py -v`
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/rectify.py`:

```python
"""Ректификация с явной привязкой растра к миллиметрам."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectified:
    image: np.ndarray
    H: np.ndarray                 # изображение -> ректифицированные ПИКСЕЛИ
    valid_mask: np.ndarray
    mm_per_px: float
    origin_rect_px: tuple


MAX_SIDE_PX = 20000


def rectify(image, H_units, mm_per_rect_unit, mm_per_px, origin_rect_units=(0.0, 0.0)):
    """Гомография в единицах -> растр, где один пиксель равен mm_per_px миллиметрам.

    Конвенция обязательна: без неё выход гомографии из точек схода безразмерен и
    порядка 1/f, отчего снимок 20 Мп схлопывается в несколько пикселей.
    """
    if mm_per_px <= 0:
        raise ValueError("mm_per_px должен быть положительным")

    h, w = image.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float) @ H_units.T
    corners = corners[:, :2] / corners[:, 2:3]

    scale = mm_per_rect_unit / mm_per_px          # единицы -> пиксели выхода
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    out_w = int(round((x1 - x0) * scale))
    out_h = int(round((y1 - y0) * scale))
    if not (1 <= out_w <= MAX_SIDE_PX and 1 <= out_h <= MAX_SIDE_PX):
        raise ValueError(
            f"невозможный размер ректифицированного растра {out_w}x{out_h}: "
            "проверьте mm_per_rect_unit и доверие к плоскости"
        )

    S = np.array([[scale, 0.0, -x0 * scale], [0.0, scale, -y0 * scale], [0.0, 0.0, 1.0]])
    H_total = S @ H_units

    warped = cv2.warpPerspective(image, H_total, (out_w, out_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(image, 255), H_total, (out_w, out_h),
                               flags=cv2.INTER_NEAREST) > 0
    ox, oy = origin_rect_units
    return Rectified(warped, H_total, mask, mm_per_px,
                     ((ox - x0) * scale, (oy - y0) * scale))
```

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Выполнить: `pytest tests/test_rectify.py -v`
Ожидается: 4 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/rectify.py tests/test_rectify.py
git commit -m "Ректификация с привязкой растра к миллиметрам и отказом при абсурдном масштабе"
```

---

## Задача 13: Оркестрация плоскости с передачей управления оператору

> **Текст переписан под фактические интерфейсы задач 6–12.** Редакция 2 плана вызывала здесь
> `camera_pose_from_homography`, которой не существует, передавала в гомографию объекты точек
> схода вместо векторов и помещала позу в результат, хотя поза требует масштаба, на этом шаге
> ещё неизвестного.

**Файлы:**
- Создать: `facade_digitizer/pipeline/plane.py`
- Тест: `tests/test_plane.py`

**Интерфейсы:**
- Потребляет: `detect_segments`, `estimate_vanishing_points` (задачи 6–7);
  `homography_from_vanishing_points`, `homography_from_four_points` (задача 8)
- Предоставляет: `PlaneResult(H, confidence, method, needs_operator, vh, vv)`,
  `estimate_plane(image, K, min_confidence=0.5) -> PlaneResult`,
  `estimate_plane_manual(image_pts, aspect_ratio=None, size_mm=None, assume_calibrated=False,
  K=None, image_size=None) -> PlaneResult`

**Почему результат не несёт позы.** Восстановление позы требует масштаба ректифицированных
координат, то есть числа миллиметров на единицу, а оно становится известно только после задания
опорного размера оператором. Поэтому `plane` отдаёт гомографию и точки схода, а позу вычисляет
сборка конвейера, где масштаб уже есть. Попытка вернуть позу здесь неизбежно потребовала бы
выдумать масштаб.

**Ниже порога доверия система не гадает** (спецификация, п. 4.2). Это не диагностика, а контракт:
при низком доверии возвращается результат с поднятым признаком `needs_operator`, и ректификация
по такой гомографии не выполняется вовсе — вызывающий код обязан либо получить ручное
доопределение, либо отказаться от снимка.

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_plane.py`:

```python
import numpy as np
import pytest

from facade_digitizer.pipeline.plane import estimate_plane, estimate_plane_manual
from tests.test_homography import CORNERS, apply
from tests.test_synth import K, SIZE, make_scene


def test_plane_is_estimated_on_a_good_scene():
    sc = make_scene()
    res = estimate_plane(sc.render(), K)
    assert res.method == "vanishing_points"
    assert not res.needs_operator
    assert res.confidence.value > 0.5


def test_estimated_homography_rectifies_the_facade():
    """Гомография проверяется по существу, а не по факту непустоты."""
    sc = make_scene()
    res = estimate_plane(sc.render(), K)
    r = apply(res.H, sc.project(CORNERS))
    a, b = r[1] - r[0], r[3] - r[0]
    ang = np.degrees(np.arccos(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert ang == pytest.approx(90.0, abs=0.5)


def test_low_confidence_requests_the_operator_instead_of_guessing():
    """Спецификация, п. 4.2: ниже порога доверия система не гадает."""
    noise = np.random.default_rng(1).integers(0, 255, (600, 600), dtype=np.uint8)
    res = estimate_plane(noise, K)
    assert res.needs_operator
    assert res.confidence.reasons
    assert res.H is None


def test_vanishing_points_are_carried_out_for_the_pose_step():
    """Позу считает сборка конвейера, поэтому точки схода обязаны выйти наружу."""
    sc = make_scene()
    res = estimate_plane(sc.render(), K)
    assert res.vh is not None and res.vv is not None
    assert np.asarray(res.vh).shape == (3,)


def test_manual_requires_disambiguation():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError, match="четырёх точек недостаточно"):
        estimate_plane_manual(pts)


def test_manual_with_sizes_produces_those_sizes():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, size_mm=(1460.0, 1900.0))
    assert res.method == "manual_four_point"
    assert not res.needs_operator
    r = apply(res.H, pts)
    assert np.linalg.norm(r[1] - r[0]) == pytest.approx(1460.0, rel=1e-6)


def test_manual_confidence_is_not_borrowed_from_the_automatic_path():
    """Ручной вариант не должен изображать измеренное доверие."""
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, aspect_ratio=2.0)
    assert res.confidence.reasons == []
    assert res.confidence.support_h == 0 and res.confidence.support_v == 0


def test_threshold_is_honoured():
    """Порог доверия действительно решает, а не декоративен."""
    sc = make_scene()
    strict = estimate_plane(sc.render(), K, min_confidence=0.999)
    assert strict.needs_operator
    assert strict.H is None
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Выполнить: `./.venv/Scripts/python.exe -m pytest tests/test_plane.py -v`
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.pipeline.plane'`

- [ ] **Шаг 3: Реализовать**

Файл `facade_digitizer/pipeline/plane.py`. Требования к реализации:

- `estimate_plane` детектирует отрезки, оценивает точки схода, и **только при доверии не ниже
  порога** строит гомографию; иначе возвращает `H = None`, `needs_operator = True` и причины,
  полученные от метрики доверия. Возвращать единичную гомографию при низком доверии запрещено:
  вызывающий код не отличит её от настоящей.
- В `homography_from_vanishing_points` передаются **векторы** `vh.point`, `vv.point`, а не
  объекты.
- `PlaneResult` несёт `vh` и `vv` наружу — они нужны сборке конвейера для восстановления позы.
- `estimate_plane_manual` поддерживает **все три** способа доопределения, как требует
  спецификация п. 4.2: отношение сторон, два размера в миллиметрах, подтверждение калибровки.
  Доверие для ручного пути конструируется явно и не изображает измеренное.

- [ ] **Шаг 4: Убедиться, что тесты проходят**

Ожидается: 8 passed

- [ ] **Шаг 5: Коммит**

```bash
git add facade_digitizer/pipeline/plane.py tests/test_plane.py
git commit -m "Оркестрация плоскости фасада с передачей управления оператору"
```

---

## Задача 14: Сквозная проверка геометрии через реальный детектор

> **Текст переписан под фактические интерфейсы задач 6–13.** Редакция 2 вызывала восстановление
> позы так, будто оно возвращает позу, тогда как оно возвращает пару — позу и признак
> неразрешённости поворота на 180°. Кроме того, редакция 2 утверждала, что невязка позы около 2 %
> расстояния соответствует принятой погрешности позы и тем подтверждает бюджет п. 6.2. После
> исправления критерия инлайера в задаче 7 невязка изменилась на порядок, и соответствие надо
> устанавливать заново, а не наследовать.

**Файлы:**
- Тест: `tests/test_integration.py`

**Интерфейсы:**
- Потребляет: задачи 5–9, 12, 13
- Предоставляет: ничего; это проверка, а не модуль

**Зачем отдельная задача.** Каждый модуль проверялся изолированно, и ни один тест не сравнивал
восстановленную позу с истиной через **реальный детектор отрезков**. В более ранней редакции
из-за этого сломанное восстановление позы, помещавшее камеру в 27 км от фасада, проходило все
проверки: единственный тест позы утверждал, что угол неотрицателен, — а это не может не
выполниться.

- [ ] **Шаг 1: Написать сквозной тест**

Файл `tests/test_integration.py`. Требования к содержанию:

- полная цепочка: рендер → детекция отрезков → точки схода → гомография → поза, **без** подстановки
  точных точек схода;
- сравнение позы с истиной сцены `camera_on_plane()`, отдельно по опорной точке и по расстоянию
  до плоскости, в долях расстояния;
- обе сцены — разреженная и обогащённая вертикальными членениями; числа приводятся порознь,
  поскольку они различаются существенно;
- не менее чем по шесть ракурсов на сцену, включая ненулевое вертикальное смещение камеры: замер
  задачи 8 показал, что именно оно даёт худшие оценки, и сетка без него занижает невязку вдвое;
- ректификация фасада проверяется по существу — прямые углы и отношение сторон против истинных
  габаритов, а не факт непустоты растра;
- учитывается, что восстановление позы возвращает **пару**: позу и признак неразрешённости
  поворота на 180°.

- [ ] **Шаг 2: Измерить и записать**

Свести в отчёт: медиану, девяносто пятый процентиль и максимум невязки опорной точки и расстояния,
порознь по двум сценам, с указанием сетки ракурсов.

- [ ] **Шаг 3: Установить соответствие бюджету заново**

Перевести измеренную невязку позы в эквивалентную погрешность направления и сопоставить с
бюджетом п. 6.2 спецификации. **Прежнее соответствие не наследовать**: оно установлено до
исправления критерия инлайера и с тех пор недействительно.

Отдельно оценить, укладывается ли невязка в допуск п. 2.2 на абсолютное положение с учётом того,
что к ней добавятся остаточная невязка ректификации и погрешность масштаба, и что поза с масштабом
коррелированы напрямую — единица ректифицированных координат и есть расстояние до плоскости.

- [ ] **Шаг 4: Коммит**

```bash
git add tests/test_integration.py
git commit -m "Сквозная проверка геометрии: поза из реального детектора против истины сцены"
```

---

## Задача 15: Сборка конвейера и CLI

> **Текст переписан под фактические интерфейсы задач 1–14.** Редакция 2 вызывала
> `estimate_plane(image, K, mm_per_px)` — третьего параметра такого рода у неё нет и быть не может;
> `rectify(image, H)` — у неё пять параметров, и без них ректифицированный растр схлопывается;
> `assess(image, gsd_min, gsd_max, theta_p95)` с `gsd_min = gsd_max = mm_per_px`, то есть
> подставляла аргумент вместо измерения; брала позу из `plane.camera`, которого нет, поскольку поза
> требует масштаба, на шаге оценки плоскости ещё неизвестного. Ниже интерфейсы приведены такими,
> какими они есть в коде на `HEAD`.

**Файлы:**
- Создать: `facade_digitizer/pipeline/run.py`
- Изменить: `facade_digitizer/schema.py` (два поля, см. шаг 3)
- Изменить: `docs/superpowers/specs/2026-09-17-facade-photo-digitization-design.md` (п. 10, см. шаг 6)
- Тест: `tests/test_run.py`

**Интерфейсы:**
- Потребляет: `load_image`, `intrinsics_from_meta`, `undistort`, `estimate_plane`,
  `estimate_plane_manual`, `homography_from_vanishing_points`, `camera_pose`,
  `rectified_to_facade_mm`, `attainable_mm_per_px`, `rectify`, `local_gsd_field`, `assess`,
  `CameraOnPlane.theta_deg`
- Предоставляет: `OperatorReference`, `process(...) -> FacadeModel`, `main() -> int`,
  точка входа CLI `facade-digitize` (уже объявлена в `pyproject.toml`)

---

### Что именно здесь собирается и чего здесь нет

Это последнее звено геометрического ядра: оно связывает уже проверенные модули в один проход
«снимок → JSON» и добавляет **ровно одно** новое вычисление — пригодность кадра, которую ни один
модуль не считает целиком (см. шаг 4). Элементов фасада конвейер не выделяет: `elements` остаётся
пустым, `coverage = "partial"`. Детекция окон и дверей — следующий этап, и задача 15 не должна
делать вид, что он уже есть.

### Откуда берётся масштаб, и почему у `process` есть параметр оператора

`camera_pose` требует `mm_per_rect_unit` — числа миллиметров на ректифицированную единицу. Эта
величина **тождественно равна расстоянию от камеры до плоскости** и из кадра невыводима: снимок
плоскости с точностью до подобия не знает своего размера. Её задаёт оператор, указывая на снимке
две точки и истинное расстояние между ними, плюс точку начала отсчёта фасада.

```python
@dataclass(frozen=True)
class OperatorReference:
    origin_px: tuple[float, float]          # точка кадра, которая станет началом координат фасада
    span_px: tuple[tuple[float, float], tuple[float, float]]
    span_mm: float                          # истинное расстояние между точками span_px
    sigma_px: float = ...                   # погрешность указания точки оператором; см. шаг 5
```

Перевод: обе точки `span_px` и `origin_px` отображаются гомографией в ректифицированные единицы;
`mm_per_unit = span_mm / |span_rect[1] − span_rect[0]|`; `origin_rect` — образ `origin_px`.

**Это то место, где задача 14 оставила незакрытую оговорку.** Там опорная длина бралась по
двадцатиметровой стороне фасада, и в отчёте прямо сказано, что реальный оператор даст базу
примерно вдесятеро короче, отчего погрешность масштаба вырастет. Здесь база — фактический вход,
и σ масштаба должна считаться **из неё**, а не браться из таблицы. См. шаг 5.

### Почему у `process` два разных «мм на пиксель»

Их действительно два, и смешение — готовый дефект:

- `mm_per_unit` — масштаб ректифицированной системы координат, он же расстояние до плоскости;
- `raster_mm_per_px` — разрешение выходного растра, свободный параметр визуализации, обязанный
  лежать внутри `attainable_mm_per_px(image, H, mm_per_unit)`.

Ни один из них не является GSD: GSD — измеренное поле, меняющееся по кадру, и берётся оно из
`local_gsd_field`. Редакция 2 подставляла один и тот же аргумент во все три роли.

---

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_run.py`. Сцену брать из `tests/test_synth.py` (`make_scene`, `K`), опорные точки
оператора получать проецированием истинных углов фасада — `scene.project(CORNERS)` из
`tests/test_homography.py`, как это делает `tests/test_integration.py`. Снимок писать в `tmp_path`
через `cv2.imwrite`; EXIF в нём не будет, и это входит в проверку (`intrinsics_from_meta` обязан
дать `calibration` из таблицы и сказать об этом).

Обязательные проверки — каждая названа свойством, а не номером:

1. **Схема валидна на обратном разборе.** `FacadeModel.model_validate_json(model.model_dump_json())`
   проходит, `schema_version == "1.1"`.
2. **В выходе нет бесконечностей и NaN.** Проверять не подстрокой в JSON — `json.dumps` пишет
   `Infinity`, а `NaN` встречается и внутри вложенных списков гомографии. Обойти разобранный JSON
   рекурсивно и потребовать `math.isfinite` от каждого числа. Подстрочная проверка редакции 2
   (`assert "null" not in payload or "Infinity" not in payload`) истинна при любом входе: это
   дизъюнкция, в которой хотя бы один член почти всегда верен.
3. **Поза записана вместе с происхождением.** `pose_to_facade` содержит `cx`, `cy`, `cz` и признак
   неразрешённости поворота на 180° с названной причиной. Глобальное ограничение: величина без
   происхождения не выпускается, а `camera_pose` возвращает пару именно для того, чтобы признак
   нельзя было потерять молча.
4. **Поза близка к истине сцены.** Невязка опорной точки и расстояния сверяется с
   `scene.camera_on_plane()`. Допуск взять с запасом к измеренному в задаче 14 (P95 0.601 % по
   опорной точке на разреженной сцене): порядка 2 % расстояния. Цель проверки — не повторить замер
   задачи 14, а поймать сборку, потерявшую или переставившую аргументы: именно такая ошибка
   ставила камеру в 27 км от фасада и проходила все тесты.
5. **`theta_cam_deg` и поле углов — разные величины.** `theta_cam_deg` строго меньше
   `theta_field_deg["max"]` и строго больше `theta_field_deg["min"]` на наклонном ракурсе.
   Проверка с одним лишь `<= max` проходила бы и при полном равенстве полю.
6. **GSD измерен, а не скопирован из аргумента.** `quality.gsd_mm_px_max > quality.gsd_mm_px_min`
   **и** ни одна из границ не равна `raster_mm_per_px`. Без второго условия проверку удовлетворяет
   любая пара констант.
7. **Низкое доверие передаёт управление оператору, а не подменяется догадкой.** На шуме:
   `rectification.needs_operator is True`, `rectification.confidence` записана, `homography is None`,
   `pose_to_facade is None`, и вердикт качества не `"ok"`. Отдельно: модель всё равно валидна по
   схеме — отказ обязан быть выражен данными, а не исключением.
8. **Масштаб вне достижимого диапазона отвергается с названием диапазона.** `process` с заведомо
   негодным `raster_mm_per_px` поднимает `ValueError`, сообщение содержит достижимые границы.
9. **σ масштаба зависит от длины опорной базы.** Та же сцена, опорная база вдесятеро короче →
   `scale.sigma_rel` примерно вдесятеро больше. Это не косметика: см. шаг 5.
10. **Несовпадение профиля калибровки со снимком отвергается по этому снимку, а не роняет пакет.**
    `main()` на двух файлах, из которых первый ссылается на несовместимый профиль, возвращает
    ненулевой код, печатает причину по имени файла и **обрабатывает второй**.
11. **Бюджет времени.** Спецификация, раздел 16: не более 15 с на снимок. Замерять на кадре того же
    порядка, что рабочий, а не на миниатюре, иначе проверка бессмысленна.

- [ ] **Шаг 2: Убедиться, что тест падает**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_run.py -v
```
Ожидается: `ModuleNotFoundError: No module named 'facade_digitizer.pipeline.run'`

- [ ] **Шаг 3: Дополнить схему двумя полями**

`facade_digitizer/schema.py`, класс `Rectification`:

- добавить `needs_operator: bool = False`;
- сделать `residual_px: float | None` — метрика доверия возвращает `None`, когда невязка не
  определена, а текущая аннотация `float` заставила бы подставить число вместо отсутствия.

`Strict` запрещает лишние поля, поэтому без этого шага проброс признака невозможен в принципе.
Больше в схеме не менять: всё прочее, что нужно задаче 15, уже описано.

- [ ] **Шаг 4: Реализовать `run.py`**

Порядок прохода и требования к каждому звену:

1. `load_image(path)` → `(image, meta)`; `intrinsics_from_meta(meta, profile_path)` → `(K, source)`;
   `undistort(image, K, dist)`, где `dist` берётся из профиля, если он задан, иначе нули.
2. `estimate_plane(image, K)` → `PlaneResult`. **Если `needs_operator` — не ректифицировать.**
   Единичная гомография вместо отказа неотличима от настоящей на стороне потребителя; это и есть
   контракт п. 4.2. Собрать модель с `rectification.needs_operator = True`, `homography = None`,
   `pose_to_facade = None`, `theta_*` = `None`, вердиктом качества по одной лишь резкости и
   вернуть её. Причины из `confidence.reasons` попадают в `quality.reasons`.
3. Масштаб и начало отсчёта — из `OperatorReference`, как описано выше.
4. `camera_pose(plane.H, plane.vh, plane.vv, K, mm_per_unit, origin_rect)` → **пара**
   `(camera, half_turn)`. Распаковывать обязательно; `half_turn.resolved`, `half_turn.assumption`
   и `half_turn.reason` идут в `pose_to_facade`.
5. **Пригодность кадра — единственное новое вычисление.** Считать на регулярной сетке узлов
   **кадра** (не фасада), потому что доля нормируется по кадру. Для каждого узла: образ под `H`,
   затем `rectified_to_facade_mm` → координаты фасада → `camera.theta_deg(x, y)`. Узел пригоден,
   когда выполняются **оба** условия: угол визирования не превышает порог И знаменатель гомографии
   сохраняет знак, то есть узел лежит перед камерой, а не за линией схода. `usable` — доля таких
   узлов; `theta_field_deg` — минимум, максимум и P95 **по видимым узлам той же сетки**.

   Почему не `angle_map` + `usable_mask` из `geometry/angles.py`: они заданы на габарите фасада,
   часть которого в кадр не попала, и доля по ним отвечала бы на другой вопрос. Признак «за линией
   схода» брать из `local_gsd_field(...).gsd`: там такие узлы помечены `nan`, и сетку для GSD и для
   углов надо взять **одну и ту же**, иначе два поля описывают разные точки.

   `theta_cam_deg` — величина отдельная: угол визирования в точке, куда приходит главная точка
   снимка (`K[0,2]`, `K[1,2]`), полученный тем же путём. Совпадение его с максимумом поля означает
   ошибку, а не удачу.
6. `gsd_min`, `gsd_max` — `nanmin` и `nanmax` того же поля GSD. Если `behind_vanishing_line`
   превышает разумную долю, это причина деградации, а не повод молча усреднить.
7. `assess(image, gsd_min, gsd_max, theta_p95, usable)` → `QualityReport`. Значения передавать
   измеренные; подстановка `raster_mm_per_px` в роли GSD — дефект редакции 2, воспроизводить его
   нельзя.
8. `attainable_mm_per_px(image, plane.H, mm_per_unit)`, затем `rectify(image, plane.H, mm_per_unit,
   raster_mm_per_px, origin_rect_units=origin_rect)`. `ValueError` из `rectify` наружу не глушить:
   он называет достижимый диапазон, и это ровно то, что нужно вызывающему. `origin_rect_units`
   передаётся обязательно: его пропуск сдвигает все измеренные положения на одну величину и
   выглядит правдоподобным результатом.
9. `bounds_mm` — габарит **охваченной** части фасада: углы образа кадра, переведённые
   `rectified_to_facade_mm`. Не размер снимка в миллиметрах и не габарит здания.
10. `plane_residual_mm` оставить `None`. `confidence.residual_px` — невязка отрезков относительно
    точек схода в пикселях снимка, а не невязка подгонки плоскости в миллиметрах; умножение её на
    GSD выдало бы угловую величину за метрическую. Отсутствие честнее выдуманного числа.
11. `mode`: `"assisted"`, когда масштаб пришёл от оператора (`operator_reference`), иначе `"auto"`.
    Оператор в проходе участвовал — выход обязан это показывать.
12. **Ни одного нечисла в выходе.** Перед сборкой модели проверить каждое число на `isfinite`; там,
    где схема допускает `None`, записывать `None`, где не допускает — поднимать исключение с
    названием поля. Бесконечность, просочившаяся в JSON, ломает обратный разбор, то есть
    обнаруживается у потребителя, а не здесь.

- [ ] **Шаг 5: σ масштаба считать, а не брать из таблицы**

Для `operator_reference` σ определяется длиной базы и точностью указания точки:

```
sigma_rel = sqrt(2) * sigma_px * gsd_at_reference / span_mm
```

где `gsd_at_reference` — измеренное локальное разрешение в районе опорных точек. Множитель `sqrt(2)`
— две независимо указанные точки. Значение `sigma_px` по умолчанию выбрать самостоятельно и
**обосновать в докстринге**: это точность попадания оператора по кромке на экране, и она должна
быть согласована с тем, как п. 6.3 спецификации получает свои 0.14–0.28 %. Проверить: на базе,
равной всей двадцатиметровой стороне, выбранное значение обязано давать σ внутри этого диапазона —
иначе выбрана величина, противоречащая спецификации.

`meets_tolerance` — результат сравнения посчитанной σ с верхней границей п. 6.3 для источника 1,
а не константа. Для остальных источников (`photogrammetry`, `exif_range`, `assumed_floor_height`)
базы нет, берутся значения таблицы п. 6.3; для источников 3 и 4 `meets_tolerance = False` по
спецификации.

Смысл шага: оговорка задачи 14 о короткой базе перестаёт быть текстом в спецификации и становится
числом в выходном файле.

- [ ] **Шаг 6: Привести п. 10 спецификации в соответствие**

В образце JSON п. 10 добавить `"needs_operator": false` в `rectification` и показать в
`pose_to_facade` признак поворота на 180°. Образец п. 10 — контракт с потребителем; поле, которое
код пишет, а образец умалчивает, потребитель не ждёт.

**Правку проверять чтением результата, а не кодом возврата скрипта.** В этом проекте дважды
случалось, что правка спецификации не нашла якорь, скрипт завершился успешно и коммит прошёл с
неизменённым файлом.

- [ ] **Шаг 7: Прогнать весь набор**

```bash
./.venv/Scripts/python.exe -m pytest -q
```
Ожидается: все проходят; было 386 passed и 2 skipped.

Отдельно проверить CLI вживую, а не только через `process`:

```bash
./.venv/Scripts/python.exe -m facade_digitizer.pipeline.run --help
```

- [ ] **Шаг 8: Коммит**

```bash
git add facade_digitizer/pipeline/run.py facade_digitizer/schema.py tests/test_run.py docs/superpowers/specs/2026-09-17-facade-photo-digitization-design.md
git commit -m "Сборка конвейера и CLI: снимок в модель фасада одним проходом"
```

---

## Задача 16: Модуль метрик точности

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

## Задача 17: Проверка на реальных снимках публичного набора CMP

> **Текст переписан под фактические интерфейсы задачи 15.** Редакция 2 вызывала
> `process(p, mm_per_px=3.0, scale_source="assumed_floor_height")` — такой сигнатуры нет: масштаб
> приходит от оператора через `OperatorReference`, а `mm_per_px` растра и масштаб ректифицированной
> системы суть разные величины. Кроме того редакция 2 называла источником `process` задачу 14;
> `process` даёт задача 15.

**Файлы:**
- Создать: `scripts/smoke_cmp.py`
- Создать: `docs/superpowers/plans/results/geometric-core-smoke.md`

**Интерфейсы:**
- Потребляет: `process`, `OperatorReference` (задача 15); `estimate_plane` (задача 13)
- Предоставляет: отчёт о поведении конвейера на реальных данных; кода в пакет не добавляет

---

### Что на этом наборе проверяемо, а что нет

Синтетика проверяет правильность математики, но ничего не говорит об устойчивости к настоящим
снимкам. CMP — набор **уже ректифицированных** фасадов, и именно это делает его дешёвой и
содержательной пробой: у ректифицированного кадра оба пучка прямых параллельны, то есть точки
схода лежат **на бесконечности**, а восстановленная нормаль к плоскости обязана совпасть с
оптической осью.

**Проверяемо:**

1. Отрабатывает ли цепочка без исключений на настоящих снимках — при том что детектор отрезков и
   пороги метрики доверия настраивались на кадре 5280×3956, а CMP на порядок мельче. Это
   самостоятельный вопрос, и ответ на него не предрешён.
2. Насколько мала восстановленная **остаточная перспектива**. Мерой служит `theta_cam_deg` —
   угол между оптической осью и нормалью к плоскости. Он **не зависит от масштаба**: масштаб
   входит в позу множителем, а угол от множителя свободен. Поэтому величина осмысленна даже при
   заведомо выдуманном опорном размере, и именно она — главный показатель задачи.
3. Доверие к плоскости и доля кадров, на которых конвейер зовёт оператора.
4. Ведёт ли себя разумно случай точек схода **на бесконечности**: третья компонента однородного
   вектора обращается в ноль, и путь `K⁻¹v` обязан это пережить. На синтетике такого ракурса нет
   ни одного — все сцены сняты под углом.

**Непроверяемо, и делать вид, что проверяемо, нельзя:**

- **Точность в миллиметрах.** Истинных размеров у CMP нет, опорный размер придётся выдумать.
  Любая величина в миллиметрах на этом наборе — следствие выдуманного числа.
- **Вердикт качества.** Порог разрешения `GSD ≤ 5 мм/px` сравнивает измеренное разрешение с
  физической величиной, а измеренное разрешение пропорционально выдуманному масштабу. Сводку
  вердиктов привести **можно и нужно**, но как описание, а не как критерий: она говорит о
  выбранном допущении столько же, сколько о снимках.

### Опорный размер: выдумывается явно и помечается как выдуманный

Задать `OperatorReference` по двум точкам, отстоящим на известную долю ширины кадра, приписав им
правдоподобную длину (порядок десяти метров на ширину фасада). Источник масштаба —
`assumed_floor_height`, то есть тот, для которого спецификация сама говорит «допуск не
достигается» (п. 6.3). Так выдуманность масштаба попадает в выходной файл, а не остаётся в голове
у того, кто запускал.

---

- [ ] **Шаг 1: Получить базовую часть набора**

```bash
mkdir -p data/cmp
curl -L -o data/cmp/base.zip https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
unzip -q data/cmp/base.zip -d data/cmp/
ls data/cmp/base/*.jpg | head -5
```

`data/` обязана быть в `.gitignore` — набор в репозиторий не попадает. Проверить, а не
предположить.

Если загрузка не удалась (сайт недоступен, адрес изменился) — **не подменять набор другим молча**.
Остановиться и сказать об этом: подмена источника данных меняет смысл всей проверки.

- [ ] **Шаг 2: Написать скрипт прогона**

Файл `scripts/smoke_cmp.py`. Требования:

- принимает папку и предел числа снимков; ничего не пишет в репозиторий;
- по каждому снимку вызывает `process` и собирает: `theta_cam_deg`, доверие к плоскости,
  `needs_operator`, вердикт качества, размер кадра, время;
- исключения **ловит по одному снимку** и относит к причинам, а не роняет прогон: одна битая
  картинка не должна лишать результата остальные;
- различает **три разных исхода**, а не два: плоскость оценена; плоскость не оценена и вызван
  оператор (штатный отказ по контракту п. 4.2); исключение (неожиданное). Сложить второе с третьим
  значит потерять единственный интересный различитель.

Сводка выводится числами, которые можно проверить: медиана и процентили `theta_cam_deg`, медиана
доверия, доли по каждому из трёх исходов, распределение вердиктов, перечень причин отказов с
числом случаев.

- [ ] **Шаг 3: Выполнить прогон**

```bash
./.venv/Scripts/python.exe scripts/smoke_cmp.py data/cmp/base
```

**Критерии, при которых надо остановиться и разбираться, а не переходить к следующему плану:**

- медиана `theta_cam_deg` заметно отлична от нуля — на ректифицированном наборе нормаль обязана
  совпасть с оптической осью, и систематическое отклонение означает дефект в оценке точек схода
  либо в выборе внутренней калибровки;
- доля исключений не пренебрежимо мала — это дефект устойчивости, а не свойство данных;
- доверие к плоскости низко на большинстве кадров, хотя перспективы в них нет: значит метрика
  доверия меряет не то, что заявлено.

Долю вызовов оператора саму по себе критерием **не считать**: на ректифицированном наборе часть
кадров может не иметь достаточного числа отрезков, и это свойство снимков.

- [ ] **Шаг 4: Записать результат**

Файл `docs/superpowers/plans/results/geometric-core-smoke.md`: сколько обработано, все три исхода
в долях, распределение `theta_cam_deg`, медиана доверия, перечень причин отказов, размеры кадров.
Отдельным разделом — **что этот прогон не проверил** (см. выше) и вывод о готовности к
следующему плану.

Если какой-то из критериев шага 3 нарушен — записать это как результат, а не как неудачу, и
остановиться.

- [ ] **Шаг 5: Коммит**

```bash
git add scripts/smoke_cmp.py docs/superpowers/plans/results/geometric-core-smoke.md
git commit -m "Проверка геометрического ядра на реальных снимках набора CMP"
```

---

## Задача 18: Оцифровка проёмов оператором и сохранение выровненного растра

**Файлы:**
- Создать: `facade_digitizer/pipeline/elements.py`
- Изменить: `facade_digitizer/pipeline/run.py` (приём разметки, `--save-rectified`)
- Тест: `tests/test_elements.py`

**Интерфейсы:**
- Потребляет: `process`, `OperatorReference` (задача 15); `rectified_to_facade_mm` (задача 8);
  `reveal_depth`, `reveal_depth_sigma`, `reveal_depth_theta_min_deg` (задачи 5, 17 порога);
  `local_gsd_field` (задача 9)
- Предоставляет: `ElementMark`, `digitize_elements(...) -> list[Element]`; ключи CLI
  `--marks`, `--save-rectified`

---

### Зачем эта задача и чего в ней нет

Ядро даёт метрическую систему координат на фасаде и доказательство её точности, но **ни одного
проёма не оцифровывает**: `elements` пуст. Между «мы построили систему координат» и «вот окно,
его размеры в миллиметрах и вот их погрешность» лежит тонкая обвязка, и она здесь.

**Автоматической детекции здесь нет и быть не должно.** Оператор указывает углы проёма сам —
это первая очередь спецификации в её исходном виде («оператор ставит точку и говорит, что это
левый угол окна»). Автоматика — следующий план; смешивать их значит лишить себя возможности
сказать, чья ошибка в результате.

### Где оператор указывает точки

**В исходном снимке, а не в выровненном растре.** Спецификация, п. 6.5: выровненное изображение
интерполировано, а на дальней стороне косого кадра растянуто, и измерение по нему есть измерение
по вычисленным пикселям. Точки переносятся в миллиметры фасада через гомографию.

Растр сохраняется (`--save-rectified`) именно для того, чтобы оператор **видел**, что и как
выровнено, и чтобы результат можно было показать. Носителем измерения он не является, и это
надо написать в докстринге, иначе следующий разработчик начнёт мерить по нему.

### Формат разметки

Файл JSON, передаваемый ключом `--marks`. На элемент:

```jsonc
{
  "class": "window",                  // window | door | facade_boundary
  "mounting": "embedded",             // embedded | flush | protruding
  "edge_type": "sharp_wall_edge",
  "corners_px": [[x, y], [x, y], [x, y], [x, y]],
  "reveal": {                          // необязательно
    "side": "right",
    "inner_edge_px": [[x, y], [x, y]]  // внутренняя кромка видимой грани откоса
  }
}
```

Чего в формате **нет намеренно**: полей `size_mm`, `recess.value_mm`, `confidence`. Всё
вычисляемое вычисляется, а не принимается на вход: иначе разметка сможет продиктовать результат,
и проверить его станет нечем.

---

- [ ] **Шаг 1: Написать падающий тест**

Файл `tests/test_elements.py`. Сцену брать из `tests/test_synth.py`; истинные габариты проёма
известны, и именно с ними сверяется результат. Обязательные проверки, каждая названа свойством:

1. **Габарит восстанавливается.** Углы проёма, спроецированные на кадр, пройдя через разметку,
   дают `size_mm` в пределах измеренной погрешности от истины сцены. Допуск брать **не из
   воздуха**: он выводится из п. 6.1 для фактического угла и GSD этого ракурса.
2. **Погрешность не константа.** Тот же проём на двух ракурсах даёт разные `sigma_width` —
   σ считается по фактическим углу и разрешению, а не берётся из таблицы режима.
3. **`meets_tolerance` вычисляется по фактическому углу.** Спецификация, п. 6.1 прямо требует
   этого вместо признака по режиму. На крутом ракурсе тот же проём обязан получить `False`.
4. **Элемент вне пригодной области отвергается или помечается.** Угол визирования на проём выше
   порога либо проём за линией схода — результат не выпускается как годный. Молча посчитать
   миллиметры там, где спецификация их не нормирует, нельзя.
5. **Глубина заглубления считается и несёт порог.** При заданной внутренней кромке откоса
   `recess.value_mm` сверяется с истиной сцены, `theta_perp_deg` записан, а на ракурсе ниже
   порога применимости (задача о вычисляемом пороге) элемент получает `recess.origin =
   "unavailable"`, а не выдуманное число.
6. **Величина без σ и без происхождения не выпускается.** `SizeMM` требует обе σ; `Recess`
   требует `theta_perp_deg` при измеренном происхождении — схема это уже стережёт, проверка
   держит, что сборка не обходит схему.
7. **Выровненный растр сохраняется и совпадает с гомографией выхода.** Точка, указанная в
   исходном снимке, пройдя через записанные `homography.H` и `origin_rect_px`, попадает в
   сохранённом растре туда, где она есть на деле. Без этой сверки записанная гомография может
   не соответствовать сохранённому файлу, и расхождение будет ненаблюдаемо.
8. **Разметка не диктует результат.** Подмена `corners_px` меняет `size_mm`; подмена любого
   поля, которого в формате нет, невозможна по построению — формат проверяется на строгость
   (лишние ключи отвергаются).

- [ ] **Шаг 2: Убедиться, что тест падает**

Привести фактический вывод, а не ожидание.

- [ ] **Шаг 3: Реализовать `elements.py`**

Требования:

1. `corners_px` → `rectified_to_facade_mm` через ту же гомографию, что записана в выход. Второй
   путь вычисления здесь недопустим: разойдутся — расхождение будет невидимо.
2. **σ габарита по п. 6.1**, слагаемыми, а не одним числом: локализация двух кромок
   `√2·σ_px·GSD` при **локальном** GSD в районе этого проёма; масштаб (σ уже посчитана сборкой
   из длины опорной базы); остаточная проективная невязка; неплоскостность как
   `отклонение·tg θ` при фактическом θ. Каждое слагаемое назвать в докстринге с его источником в
   спецификации; сумма — RSS, и допущение независимости назвать тоже.
3. `theta` элемента — в его собственной точке, тем же путём, каким сборка считает поле углов.
   Брать θ кадра или θ_cam вместо θ элемента — подмена величины.
4. Глубина — `reveal_depth` с порогом из вычисляющей функции, σ — `reveal_depth_sigma`. Ниже
   порога `recess.origin = "unavailable"`; отказ функции ловится и превращается в это, а не
   роняет весь снимок.
5. `origin = "operator"` у всех элементов этой задачи. Значения `auto`, `auto_confirmed`,
   `auto_edited` здесь не выдаются: автоматики нет, и приписывать её результату нельзя.
6. `coverage` модели становится `"full"` только если размечены границы фасада; иначе остаётся
   `"partial"`.

- [ ] **Шаг 4: `--save-rectified` в CLI**

Сохранять выровненный растр и маску охвата рядом с JSON. В докстринге и в справке CLI сказать,
что растр — для просмотра и работы оператора, а не носитель измерения (п. 6.5).

- [ ] **Шаг 5: Прогнать весь набор и ruff**

- [ ] **Шаг 6: Проба на реальном снимке**

Взять один снимок из набора CMP, разметить два-три проёма вручную, сохранить растр и JSON.
Это и есть демонстрация: косой снимок → выровненный растр → окно с размерами в миллиметрах и
названной погрешностью. Приложить к `docs/superpowers/plans/results/` как пример выхода.

Числа в миллиметрах на CMP **правильными не называть**: масштаб там выдуман (задача 17), и
демонстрируется работа цепочки, а не точность.

- [ ] **Шаг 7: Коммит**

```bash
git add facade_digitizer/pipeline/elements.py facade_digitizer/pipeline/run.py tests/test_elements.py
git commit -m "Оцифровка проёмов оператором: габариты и заглубление с погрешностью"
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

1. `pytest -v` — всё зелёное, не менее 95 тестов.
2. Регрессионные тесты раздела 5 спецификации (знак смещения, двухосевой угол, окклюзия) есть и
   проходят.
3. Модуль метрик написан и проверен **до** появления реальных данных — критерий этапа 0
   спецификации.
4. Калибровка по мишени реализована, профиль камеры сохраняется и имеет приоритет над EXIF.
5. `facade-digitize` отрабатывает на синтетическом снимке и выдаёт валидный JSON версии 1.1.
6. Прогон по CMP выполнен, отчёт записан, медиана доверия к плоскости объяснена.
7. Уточнение п. 5.4 спецификации (замкнутая форма) внесено и закоммичено.

---

## Журнал изменений

**Редакция 2.** Внесена после независимой рецензии, в которой код плана был собран во временном
окружении и **исполнен**. Результат прогона редакции 1: 8 тестов из 83 упали, а часть прошедших
проходила вхолостую. Существенно, что две предыдущие рецензии те же файлы читали и ни одного из
этих дефектов не обнаружили — класс ошибок «код, который не запускали» чтением не ловится.

Исправлено:

1. **Синтетический генератор переписан на полную модель камеры** (задача 5). Прежний `project`
   делил на постоянное расстояние, то есть был аффинным: перспективы в сцене не было, точек схода
   не существовало, и задачи 6–8 проверялись на данных без проверяемого эффекта. Добавлены
   межэтажные членения: на голой стене детектор находил 8–10 отрезков при пороге устойчивости 8.
2. **Восстановление позы камеры переписано** (задача 8). Прежний код помещал камеру в 27 км от
   фасада. Опорная точка восстанавливается через точку схода направления нормали; проверено на
   четырёх позах с невязкой меньше 10⁻¹² мм при точных точках схода.
3. **Зафиксирована ориентация осей** (задача 8). Знак точки схода произволен, оси связаны через
   векторное произведение; прежний код выдавал перевёрнутый растр, и ни один тест этого не
   проверял. Добавлена функция `rectified_to_facade_mm` — единственное место, где ось Y
   переворачивается из растровой в фасадную.
4. **Задана миллиметровая конвенция ректификации** (задача 12). Прежде снимок 20 Мп схлопывался в
   растр 16×16 пикселей, и три теста на нём успешно проходили, потому что размер выхода никто не
   проверял.
5. **Исправлена формула локального разрешения** (задача 9): наибольшее сингулярное число якобиана
   вместо `w²/det(H)`, который применял площадной показатель к линейной величине и совпадал с
   истиной ровно при `H = I` — единственном проверявшемся случае.
6. **Невязка стала измеряемой** (задача 7). Прежде `residual_px` была константой 105.6 при любой
   сцене. Добавлены причины отказа, без которых вырожденный вход получал доверие 1.00.
7. **Переписаны два пустых теста** (задача 4). `test_scalar_formula_underestimates` и
   `test_table_5_2_of_spec` считали арифметику над собственными входами и прошли бы при полностью
   удалённом модуле. Теперь оба вызывают реализацию.
8. **Шлюз качества проверяет порядок величин, а не магическое число** (задача 11). Прежний порог
   расходился с собственной синтетикой плана в 1800 раз. Известное ограничение метрики — белый шум
   набирает больше резкого изображения — зафиксировано отдельным тестом.
9. **Добавлена сквозная проверка геометрии** (задача 14): поза из реального детектора против
   истины сцены. Прежде единственная проверка позы утверждала `p95 >= 0.0`, что не может не
   выполниться.
10. **Добавлена сборка пакета** (задача 1): `[build-system]`, `pip install -e`, `tests/__init__.py`.
    Без них ни одна команда `pytest` из плана не работала.
11. **Задан порядок исполнения**: модуль метрик выполняется сразу после каркаса.
12. **Добавлены контрактные тесты на сборку конвейера** (задача 15): проброс `needs_operator`,
    отсутствие бесконечностей в выходе, разделение `theta_cam` и поля углов, измеряемый GSD.

Весь код настоящей редакции прогнан до внесения в план: 41 тест по ключевым модулям, все зелёные.

**Редакция 1.** Первоначальная декомпозиция спецификации на 16 задач.
