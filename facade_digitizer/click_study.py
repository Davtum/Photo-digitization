"""Измерение σ клика на операторах: анализ. План 3, задача 25; спецификация, п. 12.1.

Данные — файлы сессии окна (`ui.session_file`): у каждого клика там записаны
координаты кадра И масштаб просмотра, при котором он сделан. Истина — углы
синтетических проёмов (`training.SceneSpec`), спроецированные по соглашению OpenCV
о центре пикселя (центр пикселя — целые координаты), тем же, каким окно переводит
клик в координаты кадра; иначе в смещение попало бы до 0.5 px соглашения.

Что оценивается:

* **σ(s) по масштабам** и параметры модели `ui.zoom`:
  `σ² = (σ_экрана / s)² + σ_кромки²` — линейная регрессия σ² на 1/s²
  (взвешенная по числу кликов). **Форма проверяется, а не предполагается**: нужно
  не менее четырёх масштабов, из них не менее двух увеличений (s > 1) — иначе
  член σ_кромки неотличим от нуля; невязка подгонки сообщается;
* **смещение отдельно от разброса** (п. 2.2, п. 12.1): средний вектор ошибки по
  масштабу; σ — разброс вокруг него, а не RMS;
* **повторяемость внутри оператора** — по повторной разметке ОДНОГО снимка
  (истина не нужна: годится и реальный снимок): σ положения угла по повторам.

Ошибка ВЫБОРА кромки (рама вместо кромки стены) здесь не измеряется — это задача 26.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

#: Минимум масштабов и увеличений для проверки формы σ(s).
MIN_SCALES = 4
MIN_MAGNIFIED = 2
#: Кликом по углу считается точка не дальше этого от истинного угла, px кадра.
MATCH_RADIUS_PX = 25.0


@dataclass(frozen=True)
class ClickError:
    scale: float
    dx: float
    dy: float


def clicks_of_session(raw: dict) -> list[tuple[float, float, float]]:
    """Все клики углов проёмов из файла сессии: (x, y, масштаб)."""
    return [(p["x"], p["y"], p["view_scale"]) for m in raw["marks"] for p in m["corners"]]


def click_errors(raw: dict, true_corners_px: np.ndarray) -> list[ClickError]:
    """Ошибки кликов против ближайшего истинного угла (в пределах `MATCH_RADIUS_PX`)."""
    truth = np.asarray(true_corners_px, dtype=float).reshape(-1, 2)
    out = []
    for x, y, scale in clicks_of_session(raw):
        d = np.hypot(truth[:, 0] - x, truth[:, 1] - y)
        k = int(d.argmin())
        if d[k] <= MATCH_RADIUS_PX:
            out.append(ClickError(float(scale), x - truth[k, 0], y - truth[k, 1]))
    return out


def nominal_scale(scale: float, nominal: tuple[float, ...]) -> float:
    """Ближайший номинальный масштаб протокола — по логарифму."""
    return min(nominal, key=lambda n: abs(math.log(scale / n)))


@dataclass(frozen=True)
class ScaleRow:
    scale: float
    n: int
    bias_x: float
    bias_y: float
    sigma: float          # разброс одной координаты вокруг смещения, px кадра


@dataclass(frozen=True)
class SigmaFit:
    sigma_screen_px: float
    sigma_edge_px: float
    rows: tuple
    residual_rel: float   # относительная невязка модели по σ, взвешенная


def per_scale(errors: list[ClickError], nominal: tuple[float, ...]) -> list[ScaleRow]:
    groups: dict[float, list[ClickError]] = {}
    for e in errors:
        groups.setdefault(nominal_scale(e.scale, nominal), []).append(e)
    rows = []
    for s in sorted(groups):
        g = groups[s]
        if len(g) < 3:
            continue
        dx = np.array([e.dx for e in g])
        dy = np.array([e.dy for e in g])
        # σ одной координаты: объединённая несмещённая дисперсия x и y.
        var = (dx.var(ddof=1) + dy.var(ddof=1)) / 2.0
        rows.append(ScaleRow(scale=s, n=len(g), bias_x=float(dx.mean()),
                             bias_y=float(dy.mean()), sigma=float(math.sqrt(var))))
    return rows


def fit_sigma_model(rows: list[ScaleRow]) -> SigmaFit:
    """σ_экрана и σ_кромки по σ(s) — с проверкой, что форму вообще можно проверить."""
    if len(rows) < MIN_SCALES or sum(r.scale > 1 for r in rows) < MIN_MAGNIFIED:
        raise ValueError(
            f"форма σ(s) не проверяема: масштабов {len(rows)} (нужно ≥ {MIN_SCALES}), "
            f"увеличений {sum(r.scale > 1 for r in rows)} (нужно ≥ {MIN_MAGNIFIED})")
    x = np.array([1.0 / r.scale ** 2 for r in rows])
    y = np.array([r.sigma ** 2 for r in rows])
    w = np.array([r.n for r in rows], dtype=float)
    A = np.column_stack([x, np.ones_like(x)]) * np.sqrt(w)[:, None]
    (a, b), *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    a, b = max(a, 0.0), max(b, 0.0)
    model = np.sqrt(a * x + b)
    sigma = np.sqrt(y)
    residual = float(np.sqrt(np.average(((model - sigma) / sigma) ** 2, weights=w)))
    return SigmaFit(sigma_screen_px=math.sqrt(a), sigma_edge_px=math.sqrt(b),
                    rows=tuple(rows), residual_rel=residual)


def repeatability(sessions: list[dict]) -> tuple[float, int]:
    """σ положения угла по повторной разметке одного снимка одним оператором.

    Углы повторов сопоставляются с углами первой сессии по близости. Возвращает
    σ одной координаты (объединённую по углам) и число углов, вошедших в оценку.
    """
    if len(sessions) < 2:
        raise ValueError("для повторяемости нужно не менее двух повторов")
    reference = np.array([(x, y) for x, y, _ in clicks_of_session(sessions[0])])
    positions = [[] for _ in reference]
    for raw in sessions:
        for x, y, _ in clicks_of_session(raw):
            d = np.hypot(reference[:, 0] - x, reference[:, 1] - y)
            k = int(d.argmin())
            if d[k] <= MATCH_RADIUS_PX:
                positions[k].append((x, y))
    variances = []
    for pts in positions:
        if len(pts) >= 2:
            p = np.asarray(pts)
            variances.append((p[:, 0].var(ddof=1) + p[:, 1].var(ddof=1)) / 2.0)
    if not variances:
        raise ValueError("ни один угол не размечен повторно")
    return float(math.sqrt(np.mean(variances))), len(variances)


#: Ширина перехода 10–90 % у гауссова размытия ступеньки в единицах σ размытия:
#: 2·Φ⁻¹(0.9) = 2.563.
EDGE_WIDTH_PER_SIGMA = 2.5631031310892007


def edge_width_px(image: np.ndarray, p0, p1, *, half_length: float = 8.0,
                  samples: int = 15) -> float:
    """Ширина кромки (переход 10–90 %) поперёк отрезка p0–p1, px кадра.

    Мера резкости ИМЕННО размеченной кромки. `quality.sharpness` (дисперсия
    лапласиана) для этого непригодна: на кадре с шумом она меряет шум, и при шуме
    σ = 4 уровня яркости почти не меняется от размытия 0…3 px (замер задачи 25).
    Профили берутся в `samples` точках средней половины отрезка и усредняются —
    шум гасится усреднением, а не входит в меру.
    """
    img = np.asarray(image, dtype=np.float32)
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    offsets = np.linspace(-half_length, half_length, int(8 * half_length) + 1)
    profile = np.zeros_like(offsets)
    for f in np.linspace(0.25, 0.75, samples):
        c = a + f * (b - a)
        pts = c[None, :] + offsets[:, None] * n[None, :]
        mx = pts[:, 0].astype(np.float32).reshape(1, -1)
        my = pts[:, 1].astype(np.float32).reshape(1, -1)
        profile += cv2.remap(img, mx, my, cv2.INTER_LINEAR).ravel()
    profile /= samples
    lo, hi = float(profile[:4].mean()), float(profile[-4:].mean())
    if abs(hi - lo) < 1e-6:
        raise ValueError("поперёк отрезка нет перепада яркости: это не кромка")
    rising = (profile - lo) / (hi - lo)

    def crossing(level):
        idx = int(np.argmax(rising >= level))
        if idx == 0:
            return offsets[0]
        x0, x1 = offsets[idx - 1], offsets[idx]
        y0, y1 = rising[idx - 1], rising[idx]
        return x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    return float(abs(crossing(0.9) - crossing(0.1)))
