"""Единая точка сведения погрешностей. Спецификация, п. 4.4; план 3, задача 21.

«Перевод пикселей в миллиметры и расчёт итоговых σ выполняются в одном месте»:
погрешности коррелированы, и ошибка масштаба ОБЩАЯ для всех элементов фасада — она
не усредняется. Замкнутый бюджет `elements._size_sigma_mm` складывает слагаемые
как независимые и потому не умеет главного для п. 2.2 — различить σ абсолютного
положения и σ взаимного положения соседей.

**Что распространяется якобианом.** Входы, возмущение которых порождает ошибку:

* клик начала отсчёта (2 координаты кадра, σ клика этого конца базы);
* множитель масштаба `k` (номинал 1, σ = `facade.scale.sigma_rel` ТОГО ЖЕ выходного
  файла): σ масштаба уже сведена сборкой из концов базы и длины
  (`run._scale_sigma_rel`), и заводить здесь её второй расчёт — завести второй
  источник истины. Общий для всех элементов параметр — это и есть корреляция п. 4.4;
* клики по углам каждого проёма (8 координат, σ своей разметки).

Цепочка — та же, что у `elements`: `H` → `rectified_to_facade_mm` → ширина и высота
по средним противоположных сторон, положение — нижний левый угол контура. Якобиан
аналитический и поблочный: у элемента он зависит лишь от трёх общих параметров и
восьми своих; сто элементов — единицы миллисекунд. Численный якобиан
(`_jacobians_numeric`) оставлен эталоном для теста.

**Что добавляется квадратично.** Невязка, остаточная дисторсия, фокусное и
неплоскостность (`elements.size_model_terms_mm`) — допущения модели, у них нет
входа для возмущения. Для положения сверх того — градиент невязки ректификации
по полю (п. 6.4), `RECTIFICATION_GRADIENT_REL` от плеча.

**Взаимное положение — как габарит длиной в зазор.** Разность положений соседей
по построению та же величина, что и габарит: расстояние между двумя точками одной
плоскости. Поэтому модельные слагаемые к ней — те же, что к габариту длины зазора,
без градиента поля: п. 6.4 прямо говорит, что взаимное положение «свободно от
накопленного градиента». Абсолютное положение градиент несёт.

**Допущение, названное явно.** Клик начала отсчёта считается независимым от
ошибки масштаба, хотя он же — конец базы (корреляция слабая: сдвиг начала
смещает все точки одинаково, ошибка масштаба — пропорционально плечу).

**Расхождение с замкнутым бюджетом объяснимо, а не спрятано.** Локализация в
замкнутой формуле — `√2·σ·GSD`: габарит как разность двух кромок, каждая одним
кликом. Но ширина здесь — СРЕДНЕЕ двух противоположных сторон, каждая — разность
двух углов; четыре независимых клика дают `σ·GSD`, то есть в √2 раз меньше.
Второе расхождение — разрешение: замкнутая формула берёт худший узел сетки поля
(`_gsd_near`), якобиан — точную производную `H` в самих углах.
"""
from dataclasses import dataclass

import numpy as np

from facade_digitizer.geometry.homography import apply_homography, rectified_to_facade_mm
from facade_digitizer.pipeline.elements import (
    ASSISTED_SIZE_TOLERANCE_MM,
    FOCAL_ERROR_BY_CALIBRATION,
    RESIDUAL_DISTORTION_MM,
    RESIDUAL_REFERENCE_MM,
    WALL_FLATNESS_DEVIATION_MM,
    _gsd_near,
    _quad_size_mm,
    size_model_terms_mm,
)
from facade_digitizer.schema import Element

#: Градиент остаточной невязки ректификации по полю, доля плеча от начала отсчёта.
#: П. 6.4: «0.2–0.5 %»; взята нижняя граница — верхняя одна исчерпала бы допуск
#: абсолютного положения п. 2.2 (0.4 %). Величина подлежит измерению (задача 25).
RECTIFICATION_GRADIENT_REL = 0.002

#: Шаги центральных разностей: пиксель кадра и относительный масштаб.
_STEP_PX = 1e-3
_STEP_K = 1e-6
#: Выходы элемента: ширина, высота, x и y нижнего левого угла, мм.
OUTPUTS = ("width", "height", "x", "y")


@dataclass(frozen=True)
class Inputs:
    """Номинальные входы и их σ. Общие: начало отсчёта и масштаб; свои — углы."""

    H: np.ndarray
    mm_per_unit: float
    origin_px: tuple[float, float]
    origin_sigma_px: float
    scale_sigma_rel: float
    corners_px: tuple          # по элементу массив 4×2 в порядке ElementMark
    corner_sigma_px: tuple     # по элементу σ клика, px

    @property
    def global_sigmas(self) -> np.ndarray:
        s = self.origin_sigma_px
        return np.array([s, s, self.scale_sigma_rel])


def outputs(H, mm_per_unit: float, origin_px, k: float, corners_px) -> np.ndarray:
    """Ширина, высота, x, y нижнего левого угла — той же цепочкой, что `elements`."""
    origin_rect = apply_homography(H, [origin_px])[0]
    contour = rectified_to_facade_mm(apply_homography(H, corners_px), origin_rect,
                                     k * mm_per_unit)
    width, height = _quad_size_mm(contour)
    return np.array([width, height, contour[0, 0], contour[0, 1]])


def _jacobians_numeric(inp: Inputs, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Якобиан центральными разностями — эталон для аналитического (тесты)."""
    corners = np.asarray(inp.corners_px[index], dtype=float)
    ox, oy = inp.origin_px

    def f(gx, gy, k, c):
        return outputs(inp.H, inp.mm_per_unit, (gx, gy), k, c)

    jg = np.empty((4, 3))
    for col, (dx, dy, dk) in enumerate(((_STEP_PX, 0, 0), (0, _STEP_PX, 0),
                                        (0, 0, _STEP_K))):
        step = _STEP_K if dk else _STEP_PX
        jg[:, col] = (f(ox + dx, oy + dy, 1.0 + dk, corners)
                      - f(ox - dx, oy - dy, 1.0 - dk, corners)) / (2 * step)
    jo = np.empty((4, 8))
    for col in range(8):
        delta = np.zeros(8)
        delta[col] = _STEP_PX
        plus = corners + delta.reshape(4, 2)
        minus = corners - delta.reshape(4, 2)
        jo[:, col] = (f(ox, oy, 1.0, plus) - f(ox, oy, 1.0, minus)) / (2 * _STEP_PX)
    return jg, jo


def _homography_jacobian(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Производная отображения `H` в точках: массив (..., 2, 2)."""
    x, y = pts[..., 0], pts[..., 1]
    q = np.stack([H[r, 0] * x + H[r, 1] * y + H[r, 2] for r in range(3)], axis=-1)
    u, v, w = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2], q[..., 2]
    j = np.empty(pts.shape[:-1] + (2, 2))
    j[..., 0, 0] = (H[0, 0] - u * H[2, 0]) / w
    j[..., 0, 1] = (H[0, 1] - u * H[2, 1]) / w
    j[..., 1, 0] = (H[1, 0] - v * H[2, 0]) / w
    j[..., 1, 1] = (H[1, 1] - v * H[2, 1]) / w
    return j


def _jacobians(inp: Inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Номинал n×4 и якобианы n×4×3 (общие) и n×4×8 (свои) — аналитически.

    Контур `P = s·S·(H(c) − H(o))`, `s = k·mm_per_unit`, `S = diag(1, −1)`.
    Выходы — функции контура: ширина и высота через единичные векторы сторон,
    положение — `P₀`. Все выходы однородны первой степени по `P`, поэтому
    производная по `k` при `k = 1` равна самим выходам.
    """
    H = inp.H
    s = inp.mm_per_unit
    S = np.diag([1.0, -1.0])
    corners = np.asarray(inp.corners_px, dtype=float).reshape(-1, 4, 2)
    n = corners.shape[0]
    origin_rect = apply_homography(H, [inp.origin_px])[0]
    rect = apply_homography(H, corners.reshape(-1, 2)).reshape(n, 4, 2)
    P = s * (rect - origin_rect) * np.array([1.0, -1.0])

    def unit(a, b):
        d = P[:, a] - P[:, b]
        return d / np.linalg.norm(d, axis=1, keepdims=True), np.linalg.norm(d, axis=1)

    e10, l10 = unit(1, 0)
    e23, l23 = unit(2, 3)
    e30, l30 = unit(3, 0)
    e21, l21 = unit(2, 1)
    nominal = np.column_stack([(l10 + l23) / 2, (l30 + l21) / 2, P[:, 0, 0], P[:, 0, 1]])

    # G: производная выходов по координатам контура, n×4×(4·2).
    G = np.zeros((n, 4, 4, 2))
    G[:, 0, 1], G[:, 0, 0] = e10 / 2, -e10 / 2
    G[:, 0, 2], G[:, 0, 3] = e23 / 2, -e23 / 2
    G[:, 1, 3], G[:, 1, 0] = e30 / 2, -e30 / 2
    G[:, 1, 2], G[:, 1, 1] = e21 / 2, -e21 / 2
    G[:, 2, 0, 0] = 1.0
    G[:, 3, 0, 1] = 1.0

    # Свои углы: dP_m/dc_m = s·S·J(c_m) — блочно-диагонально.
    dP_dc = s * np.einsum("ab,nmbc->nmac", S, _homography_jacobian(H, corners))
    jo = np.einsum("nima,nmab->nimb", G, dP_dc).reshape(n, 4, 8)
    # Начало отсчёта: dP_m/do = −s·S·J(o) у всех углов одинаково.
    dP_do = -s * S @ _homography_jacobian(H, np.asarray(inp.origin_px, dtype=float))
    jg = np.empty((n, 4, 3))
    jg[:, :, :2] = np.einsum("nima,ab->nib", G, dP_do)
    jg[:, :, 2] = nominal
    return nominal, jg, jo


@dataclass(frozen=True)
class Propagated:
    """Ковариации распространённых входов (без модельных слагаемых)."""

    nominal: np.ndarray        # n×4
    jg: np.ndarray             # n×4×3
    jo: np.ndarray             # n×4×8
    sg: np.ndarray             # 3 — σ общих параметров
    so: np.ndarray             # n — σ клика своих углов

    def cov(self, i: int, j: int) -> np.ndarray:
        """Ковариация 4×4 выходов элементов i и j."""
        c = self.jg[i] @ np.diag(self.sg ** 2) @ self.jg[j].T
        if i == j:
            c = c + self.so[i] ** 2 * self.jo[i] @ self.jo[i].T
        return c

    def relative_cov(self, i: int, j: int) -> np.ndarray:
        """Ковариация 2×2 разности положений нижних левых углов i и j."""
        p = slice(2, 4)
        return (self.cov(i, i)[p, p] + self.cov(j, j)[p, p]
                - self.cov(i, j)[p, p] - self.cov(j, i)[p, p])


def propagate(inp: Inputs) -> Propagated:
    nominal, jg, jo = _jacobians(inp)
    return Propagated(nominal=nominal, jg=jg, jo=jo, sg=inp.global_sigmas,
                      so=np.asarray(inp.corner_sigma_px, dtype=float))


def monte_carlo(inp: Inputs, samples: int, rng) -> np.ndarray:
    """Выборка выходов samples×n×4 при гауссовых возмущениях тех же входов."""
    n = len(inp.corners_px)
    sg = inp.global_sigmas
    out = np.empty((samples, n, 4))
    corners = [np.asarray(c, dtype=float) for c in inp.corners_px]
    for s in range(samples):
        g = rng.normal(0.0, 1.0, 3) * sg
        origin = (inp.origin_px[0] + g[0], inp.origin_px[1] + g[1])
        for i, c in enumerate(corners):
            noisy = c + rng.normal(0.0, inp.corner_sigma_px[i], c.shape)
            out[s, i] = outputs(inp.H, inp.mm_per_unit, origin, 1.0 + g[2], noisy)
    return out


def _worst_axis(cov2: np.ndarray) -> float:
    """σ по худшему направлению: корень наибольшего собственного числа."""
    return float(np.sqrt(max(np.linalg.eigvalsh(cov2).max(), 0.0)))


def nearest_neighbours(positions: np.ndarray) -> list[int | None]:
    """Ближайший сосед каждого элемента по нижнему левому углу; одному — `None`."""
    n = len(positions)
    if n < 2:
        return [None] * n
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return [int(j) for j in d.argmin(axis=1)]


def assemble(elements: list[Element], *, H, mm_per_unit: float, origin_px,
             origin_sigma_px: float, scale_sigma_rel: float, gsd_field: np.ndarray,
             image_size, residual_px, calibration: str,
             wall_flatness_mm: float = WALL_FLATNESS_DEVIATION_MM,
             distortion_mm: float = RESIDUAL_DISTORTION_MM) -> list[Element]:
    """σ габарита, абсолютного и взаимного положения — в одном месте.

    Углы и σ клика берутся из `contour_px` самих элементов: это те пиксели и та σ,
    с которыми элемент посчитан, и второго входа, способного с ними разойтись, нет.
    Признак допуска пересчитывается по новой σ габарита и лишь там, где он был
    определён (`edge_reference = "wall_plane"`): отсутствие признака по п. 5.6
    сведение погрешностей не отменяет.
    """
    if not elements:
        return elements
    focal_rel = FOCAL_ERROR_BY_CALIBRATION[calibration]
    corners = tuple(np.asarray(e.contour_px[0]["points"], dtype=float) for e in elements)
    inp = Inputs(H=np.asarray(H, dtype=float), mm_per_unit=float(mm_per_unit),
                 origin_px=(float(origin_px[0]), float(origin_px[1])),
                 origin_sigma_px=float(origin_sigma_px),
                 scale_sigma_rel=float(scale_sigma_rel), corners_px=corners,
                 corner_sigma_px=tuple(float(e.contour_px[0]["sigma_px"]) for e in elements))
    prop = propagate(inp)
    positions = prop.nominal[:, 2:4]
    neighbours = nearest_neighbours(positions)

    thetas = [e.theta.full_deg for e in elements]
    residuals = [residual_px * _gsd_near(gsd_field, image_size, c, label=e.id)
                 if residual_px is not None else RESIDUAL_REFERENCE_MM
                 for e, c in zip(elements, corners, strict=True)]

    out = []
    for i, element in enumerate(elements):
        theta, residual_mm = thetas[i], residuals[i]
        cov = prop.cov(i, i)
        width, height = element.size_mm.width, element.size_mm.height

        def terms(length, _theta=theta, _res=residual_mm):
            return size_model_terms_mm(length, _theta, _res, wall_flatness_mm,
                                       distortion_mm, focal_rel)

        sigma_w = float(np.hypot.reduce([np.sqrt(cov[0, 0]), *terms(width)]))
        sigma_h = float(np.hypot.reduce([np.sqrt(cov[1, 1]), *terms(height)]))
        arm = float(np.linalg.norm(positions[i]))
        sigma_pos = float(np.hypot.reduce([_worst_axis(cov[2:4, 2:4]), *terms(arm),
                                           RECTIFICATION_GRADIENT_REL * arm]))
        sigma_rel_pos = None
        j = neighbours[i]
        if j is not None:
            # У пары одна σ разности: модельные слагаемые — по худшему из двух.
            gap = float(np.linalg.norm(positions[i] - positions[j]))
            sigma_rel_pos = float(np.hypot.reduce([
                _worst_axis(prop.relative_cov(i, j)),
                *terms(gap, max(thetas[i], thetas[j]), max(residuals[i], residuals[j]))]))
        meets = element.meets_tolerance
        if meets is not None:
            meets = bool(max(sigma_w, sigma_h) <= ASSISTED_SIZE_TOLERANCE_MM)
        out.append(element.model_copy(update={
            "size_mm": element.size_mm.model_copy(update={"sigma_width": sigma_w,
                                                          "sigma_height": sigma_h}),
            "position_sigma_mm": sigma_pos,
            "relative_position_sigma_mm": sigma_rel_pos,
            "meets_tolerance": meets,
        }))
    return out
