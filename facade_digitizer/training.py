"""Обучение операторов: блоки с известной истиной и критерий завершения.

План 3, задача 24; спецификация, п. 13 и п. 14 («обучение до измерений»). Без
обучения замеры задач 25–26 измеряют кривую обучения, а не разметку.

Блок — несколько синтетических фасадов с проёмами РАЗНЫХ размеров под разными
ракурсами; оператор размечает их в интерфейсе и выгружает JSON. `score_block`
сверяет экспорт с истиной (габариты и глубина: начало отсчёта у оператора своё,
поэтому положение здесь не сравнивается), `converged` решает, закончено ли
обучение: **время на элемент и промах стабилизировались на трёх последовательных
блоках**.

Снимки рендерятся с размытием и шумом (`BLUR_SIGMA_PX`, `NOISE_SIGMA`): идеально
резкая кромка учила бы кликать по тому, чего на реальных снимках нет. Уровни —
допущение до задачи 25, где они подбираются по резкости реальных снимков.
"""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from facade_digitizer.geometry.parallax import visible_reveal_side
from facade_digitizer.synth.scene import Opening, SyntheticScene

K = np.array([[3600.0, 0.0, 2640.0], [0.0, 3600.0, 1978.0], [0.0, 0.0, 1.0]])
SIZE = (5280, 3956)
FACADE_MM = (20000.0, 15000.0)
SCENES_PER_BLOCK = 3
OPENINGS_PER_SCENE = 3
#: Размытие и шум рендера — допущение до задачи 25 (см. докстринг модуля).
BLUR_SIGMA_PX = 1.2
NOISE_SIGMA = 4.0
#: Критерий завершения: три последовательных блока, в которых медианное время на
#: элемент различается не более чем на 15 %, а медианный промах — не более 2 мм.
WINDOW = 3
TIME_TOLERANCE_REL = 0.15
ERROR_TOLERANCE_MM = 2.0

_WIDTHS = (900.0, 1200.0, 1460.0, 1800.0)
_HEIGHTS = (1500.0, 1900.0, 2100.0, 2400.0)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    dx: float
    dy: float
    dist: float
    openings: tuple

    def scene(self) -> SyntheticScene:
        ops = [Opening(**o) for o in self.openings]
        return SyntheticScene.looking_at_centre(*FACADE_MM, ops, self.dist, self.dx,
                                                self.dy, K, SIZE)


def _fits(sc: SyntheticScene) -> bool:
    w, h = SIZE
    cam = sc.camera_on_plane()
    for op in sc.openings:
        px = sc.project(op.corners_mm())
        if not (np.all(px[:, 0] > 40) and np.all(px[:, 0] < w - 40)
                and np.all(px[:, 1] > 40) and np.all(px[:, 1] < h - 40)):
            return False
        # Худший угол контура — тем же правилом, что у ядра (`_worst_corner_theta`),
        # с запасом до порога 30°: проём не должен отвергаться ядром.
        worst = max(math.degrees(math.atan(math.hypot(x - cam.cx, y - cam.cy) / cam.cz))
                    for x, y in op.corners_mm())
        if worst > 27.0:
            return False
        side, _ = visible_reveal_side(cam, op.x, op.x + op.width, op.y, op.y + op.height)
        if side is None:
            return False
    return True


def make_block(block: int, seed: int = 0) -> list[SceneSpec]:
    """Сцены блока — детерминированно по номеру блока и зерну."""
    rng = np.random.default_rng([seed, block])
    specs = []
    while len(specs) < SCENES_PER_BLOCK:
        widths = rng.choice(_WIDTHS, OPENINGS_PER_SCENE, replace=False)
        heights = rng.choice(_HEIGHTS, OPENINGS_PER_SCENE, replace=False)
        xs = 5500.0 + 3000.0 * np.arange(OPENINGS_PER_SCENE) + rng.uniform(-300, 300, 3)
        ys = rng.uniform(4000.0, 8000.0, OPENINGS_PER_SCENE)
        openings = tuple({"x": float(x), "y": float(y), "width": float(w),
                          "height": float(h), "depth": float(rng.uniform(120, 250))}
                         for x, y, w, h in zip(xs, ys, widths, heights, strict=True))
        spec = SceneSpec(name=f"блок{block:02d}_сцена{len(specs) + 1}",
                         dx=float(rng.uniform(-3000, 3000)),
                         dy=float(rng.uniform(-3000, -1000)),
                         dist=float(rng.uniform(8000, 11000)), openings=openings)
        if _fits(spec.scene()):
            specs.append(spec)
    return specs


def render(spec: SceneSpec, seed: int = 0, *, blur_sigma_px: float = BLUR_SIGMA_PX,
           noise_sigma: float = NOISE_SIGMA) -> np.ndarray:
    img = spec.scene().render().astype(np.float32)
    if blur_sigma_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma_px)
    img += np.random.default_rng(seed).normal(0.0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def truth_record(spec: SceneSpec) -> dict:
    first = spec.openings[0]
    return {**asdict(spec),
            "base_hint": ("нижняя кромка самого левого проёма, "
                          f"{first['width']:.0f} мм")}


@dataclass(frozen=True)
class ElementScore:
    scene: str
    element_id: str
    width_error_mm: float
    height_error_mm: float
    depth_error_mm: float | None
    seconds: float | None


def score_scene(model: dict, truth: dict) -> list[ElementScore]:
    """Сопоставить элементы экспорта с проёмами истины (по габаритам) и оценить.

    Габариты проёмов сцены различны по построению, поэтому сопоставление по
    минимуму суммарного расхождения размеров однозначно.
    """
    elements = model["elements"]
    ops = truth["openings"]
    if not elements:
        return []
    cost = np.array([[abs(e["size_mm"]["width"] - o["width"])
                      + abs(e["size_mm"]["height"] - o["height"]) for o in ops]
                     for e in elements])
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols, strict=True):
        e, o = elements[r], ops[c]
        recess = e.get("recess") or {}
        depth = recess.get("value_mm")
        operator = e.get("operator") or {}
        out.append(ElementScore(
            scene=truth["name"], element_id=e["id"],
            width_error_mm=e["size_mm"]["width"] - o["width"],
            height_error_mm=e["size_mm"]["height"] - o["height"],
            depth_error_mm=None if depth is None else depth - o["depth"],
            seconds=operator.get("seconds")))
    return out


@dataclass(frozen=True)
class BlockStats:
    block: int
    elements: int
    median_seconds: float | None
    median_abs_error_mm: float


def block_stats(block: int, scores: list[ElementScore]) -> BlockStats:
    if not scores:
        raise ValueError(f"блок {block}: ни одного оценённого элемента")
    errors = [abs(v) for s in scores for v in (s.width_error_mm, s.height_error_mm)]
    seconds = [s.seconds for s in scores if s.seconds is not None]
    return BlockStats(block=block, elements=len(scores),
                      median_seconds=float(np.median(seconds)) if seconds else None,
                      median_abs_error_mm=float(np.median(errors)))


def converged(stats: list[BlockStats]) -> tuple[bool, str]:
    """Обучение закончено, если последние `WINDOW` блоков стабильны по времени и промаху."""
    if len(stats) < WINDOW:
        return False, f"пройдено блоков {len(stats)} из минимум {WINDOW}"
    last = stats[-WINDOW:]
    times = [s.median_seconds for s in last]
    if any(t is None or t <= 0 for t in times):
        return False, "нет хронометража (оператор без метки или без журнала)"
    spread_t = max(times) / min(times) - 1.0
    errors = [s.median_abs_error_mm for s in last]
    spread_e = max(errors) - min(errors)
    ok_t, ok_e = spread_t <= TIME_TOLERANCE_REL, spread_e <= ERROR_TOLERANCE_MM
    why = (f"время на элемент: разброс {spread_t:.0%} (порог {TIME_TOLERANCE_REL:.0%}); "
           f"промах: разброс {spread_e:.1f} мм (порог {ERROR_TOLERANCE_MM:.1f} мм)")
    return ok_t and ok_e, why


def write_block(out_dir, block: int, seed: int = 0, *,
                blur_sigma_px: float = BLUR_SIGMA_PX, prefix: str = "блок") -> list[Path]:
    """Снимки и `истина.json` блока — для оператора и для оценки."""
    from facade_digitizer.pipeline.io import save_image

    out = Path(out_dir) / f"{prefix}{block:02d}"
    out.mkdir(parents=True, exist_ok=True)
    specs = make_block(block, seed)
    paths = []
    for i, spec in enumerate(specs):
        path = out / f"{spec.name}.png"
        save_image(path, render(spec, seed=seed * 1000 + block * 10 + i,
                                blur_sigma_px=blur_sigma_px))
        paths.append(path)
    (out / "истина.json").write_text(
        json.dumps([truth_record(s) for s in specs], ensure_ascii=False, indent=2),
        encoding="utf-8")
    return paths
