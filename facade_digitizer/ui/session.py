"""Состояние сессии оператора. План 3, задача 7.

Qt здесь НЕ импортируется (проверяется тестом): всё, что оператор сделал, и всё, что
из этого следует, хранится и проверяется в обычном Python, а виджеты только читают
сессию и посылают ей события. Так логика «чего не хватает, чтобы мерить» тестируется
без окна, а сама сессия годится и для пакетного пути без интерфейса.

Вычисления идут через фазы ядра (`pipeline.run`: `frame_stage`, `scale_stage`,
`elements_stage`) — своей геометрии у сессии нет. Ошибки фаз не поднимаются, а
сохраняются текстом в `error`: окно показывает причину, а не падает.

σ клика сессия не вычисляет сама: она хранит масштаб просмотра каждого клика
(`ClickedPoint.view_scale`) и спрашивает σ у переданной функции — ею служит
`ui.zoom.sigma_image_px` (задача 8). Масштаб хранится рядом с координатой, а не в
настройках сессии, потому что оператор меняет увеличение между кликами.
"""
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.elements import ElementMark
from facade_digitizer.pipeline.plane import ManualPlane
from facade_digitizer.schema import FacadeModel

#: Префикс идентификатора разметки по классу — тот же, что у `elements._element_id`.
_ID_PREFIX = {"window": "w", "door": "d"}


@dataclass(frozen=True)
class ClickedPoint:
    """Точка, указанная оператором, ВМЕСТЕ с масштабом, при котором он её указал.

    Координаты — в пикселях единого кадра (`pipeline.frame`), центры пикселей в
    целых числах. `view_scale` — экранных пикселей на пиксель кадра: 1.0 — 1:1.
    """

    x: float
    y: float
    view_scale: float

    @property
    def xy(self) -> tuple[float, float]:
        return (float(self.x), float(self.y))


SigmaOf = Callable[[ClickedPoint], float]


@dataclass
class ReferenceDraft:
    """Опорная база: два конца (первый — начало отсчёта) и длина между ними."""

    ends: list[ClickedPoint] = field(default_factory=list)
    span_mm: float | None = None
    scale_source: str = "operator_reference"
    #: Погрешность самой длины (рулетка, дальномер), мм; `None` — не задана.
    span_sigma_mm: float | None = None
    #: Первый конец базы — левый нижний угол фасада (п. 7: начало `bottom_left`).
    origin_is_facade_corner: bool = False


@dataclass
class MarkDraft:
    """Черновик разметки проёма. Проверяется `ElementMark`, а не своей проверкой."""

    id: str
    class_: str = "window"
    mounting: str = "embedded"
    edge_type: str = "sharp_wall_edge"
    corners: list[ClickedPoint] = field(default_factory=list)
    reveal_side: str | None = None
    reveal_points: list[ClickedPoint] = field(default_factory=list)


def order_corners(points) -> np.ndarray:
    """Четыре угла в обходе `ElementMark`: нижний левый, нижний правый, верхний правый,
    верхний левый — при любом порядке кликов.

    Углы сортируются по направлению от центра четырёхугольника в координатах кадра
    (ось y вниз): нижний левый лежит в (90°, 180°), нижний правый — в (0°, 90°),
    верхний правый — в (−90°, 0°), верхний левый — в (−180°, −90°). Отсчёт ведётся
    от 180° — от направления влево, между верхним левым и нижним левым углами, где
    угла выпуклого четырёхугольника без сильного крена не бывает. Первая версия
    отсчитывала от 135° и на сильной перспективе ошибалась: нижний левый угол
    косого фасада лежал на 145°, «перескакивал» в конец, и обход сдвигался на одну
    позицию (найдено тестом окна задачи 10). Порядок кликов оператору поэтому
    безразличен, и расхождение обходов двух форматов ядра (`ElementMark` — от
    нижнего левого, ручная плоскость — от верхнего левого) до оператора не доходит.
    Предполагается снимок без крена больше 45° — кадр уже повёрнут по EXIF.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"у проёма четыре угла, передано {pts.shape[0]}")
    cx, cy = pts.mean(axis=0)
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return pts[np.argsort((180.0 - angles) % 360.0)]


def _file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OperatorSession:
    """Всё, что оператор сделал со снимком, и результаты фаз ядра."""

    def __init__(self):
        self.image_path: Path | None = None
        self.image_hash: str | None = None
        self.profile_path: Path | None = None
        self.profile_hash: str | None = None
        self.plane_override: ManualPlane | None = None
        self.roi = None
        self.reference = ReferenceDraft()
        self.marks: list[MarkDraft] = []
        self._next_mark = 0
        self.plane_points: list[ClickedPoint] = []      # углы ручной плоскости
        self.roi_points: list[ClickedPoint] = []        # область оценки точек схода
        self.frame = None            # run.FrameStage
        self.scale = None            # run.ScaleStage
        self.model: FacadeModel | None = None
        self.error: str | None = None

    # --- снимок и профиль -------------------------------------------------------

    def has_clicks(self) -> bool:
        return (bool(self.reference.ends) or bool(self.plane_points)
                or bool(self.roi_points)
                or any(m.corners or m.reveal_points for m in self.marks))

    def open_image(self, path) -> None:
        """Новый снимок — новая сессия: клики по прежнему снимку к нему не относятся."""
        path = Path(path)
        self.__init__()
        self.image_path = path
        self.image_hash = _file_hash(path)

    def set_profile(self, path, *, discard_clicks: bool = False) -> None:
        """Подключить (или снять, `None`) профиль калибровки.

        Профиль сдвигает кадр (снятие дисторсии), и все клики, сделанные по прежнему
        кадру, теряют смысл. Молча сбросить их нельзя: без `discard_clicks=True`
        смена профиля при наличии кликов — отказ, и окно спрашивает оператора.
        """
        if self.has_clicks() and not discard_clicks:
            clicks = len(self.reference.ends) + sum(len(m.corners) + len(m.reveal_points)
                                                     for m in self.marks)
            raise ValueError(
                f"смена профиля калибровки сдвигает кадр и сбросит указанные точки "
                f"({clicks}): подтвердите сброс")
        self.reference = ReferenceDraft()
        self.marks = []
        self.plane_points, self.roi_points = [], []
        self.plane_override, self.roi = None, None
        self.frame = self.scale = self.model = None
        self.error = None
        self.profile_path = Path(path) if path is not None else None
        self.profile_hash = _file_hash(path) if path is not None else None

    def set_plane_override(self, override: ManualPlane | None, roi=None) -> None:
        self.plane_override, self.roi = override, roi
        self.frame = self.scale = self.model = None

    # --- ручная плоскость и область оценки (задача 10) ----------------------------

    def add_plane_point(self, point: ClickedPoint) -> None:
        if len(self.plane_points) >= 4:
            raise ValueError("у ручной плоскости четыре угла, все уже указаны")
        self.plane_points.append(point)

    def manual_plane(self, *, aspect_ratio=None, size_mm=None,
                     assume_calibrated: bool = False) -> ManualPlane:
        """Четыре указанных угла в порядке ядра (верхний левый, верхний правый, нижний
        правый, нижний левый) — при любом порядке кликов — и одно доопределение."""
        if len(self.plane_points) != 4:
            raise ValueError(f"ручная плоскость: углов {len(self.plane_points)} из 4")
        bl, br, tr, tl = order_corners([p.xy for p in self.plane_points])
        return ManualPlane(image_pts=np.array([tl, tr, br, bl]), aspect_ratio=aspect_ratio,
                           size_mm=size_mm, assume_calibrated=assume_calibrated)

    def add_roi_point(self, point: ClickedPoint) -> None:
        self.roi_points.append(point)

    def roi_polygon(self) -> np.ndarray:
        if len(self.roi_points) < 3:
            raise ValueError(f"область оценки: точек {len(self.roi_points)}, нужно не менее 3")
        return np.array([p.xy for p in self.roi_points])

    def accept_frame(self, fs, *, plane_override=None, roi=None) -> None:
        """Новая фаза кадра (в том числе с заменённой плоскостью): масштаб и модель,
        посчитанные по прежней плоскости, недействительны."""
        self.frame, self.plane_override, self.roi = fs, plane_override, roi
        self.scale = self.model = None
        self.error = None

    def compute_frame(self):
        """Фаза кадра (секунды): кадр, K, плоскость, резкость. Цветной — для показа."""
        self.frame = self.scale = self.model = None
        self.error = None
        if self.image_path is None:
            self.error = "снимок не открыт"
            return None
        try:
            self.frame = run.frame_stage(self.image_path, profile_path=self.profile_path,
                                         with_color=True, plane_override=self.plane_override,
                                         roi=self.roi)
        except (ValueError, FileNotFoundError) as error:
            self.error = str(error)
        return self.frame

    # --- опорная база ------------------------------------------------------------

    def add_reference_end(self, point: ClickedPoint) -> None:
        if len(self.reference.ends) >= 2:
            raise ValueError("у опорной базы два конца, оба уже указаны")
        self.reference.ends.append(point)
        self.scale = self.model = None

    def clear_reference(self) -> None:
        self.reference = ReferenceDraft(scale_source=self.reference.scale_source,
                                        span_sigma_mm=self.reference.span_sigma_mm,
                                        origin_is_facade_corner=
                                        self.reference.origin_is_facade_corner)
        self.scale = self.model = None

    def set_span_mm(self, value: float | None) -> None:
        self.reference.span_mm = value
        self.scale = self.model = None

    def set_base_options(self, *, span_sigma_mm: float | None, scale_source: str,
                         origin_is_facade_corner: bool) -> None:
        """Все три значения задаются явно: `span_sigma_mm = None` — погрешность длины
        не задана (а не «оставить прежнюю»)."""
        self.reference.span_sigma_mm = span_sigma_mm
        self.reference.scale_source = scale_source
        self.reference.origin_is_facade_corner = bool(origin_is_facade_corner)
        self.scale = self.model = None

    def operator_reference(self, sigma_of: SigmaOf) -> run.OperatorReference:
        a, b = self.reference.ends
        r = self.reference
        return run.OperatorReference(origin_px=a.xy, span_px=(a.xy, b.xy),
                                     span_mm=float(r.span_mm),
                                     sigma_px=max(sigma_of(a), sigma_of(b)),
                                     end_sigma_px=(sigma_of(a), sigma_of(b)),
                                     span_sigma_mm=r.span_sigma_mm,
                                     origin_is_facade_corner=r.origin_is_facade_corner)

    def compute_scale(self, sigma_of: SigmaOf):
        """Фаза масштаба (миллисекунды). Оценку плоскости не повторяет."""
        self.scale = self.model = None
        self.error = None
        if self.frame is None or len(self.reference.ends) != 2 or not self._span_ok():
            self.error = "; ".join(self.ready_to_measure(require_marks=False)[1])
            return None
        try:
            self.scale = run.scale_stage(self.frame, self.operator_reference(sigma_of),
                                         scale_source=self.reference.scale_source)
        except ValueError as error:
            self.error = str(error)
        return self.scale

    def _span_ok(self) -> bool:
        v = self.reference.span_mm
        return v is not None and math.isfinite(v) and v > 0

    # --- разметка ----------------------------------------------------------------

    def new_mark(self, class_: str = "window") -> MarkDraft:
        """Черновик с идентификатором, который не переиспользуется никогда: удаление
        разметки не сдвигает идентификаторы остальных (план 3, задача 6)."""
        mark = MarkDraft(id=f"{_ID_PREFIX.get(class_, 'e')}_{self._next_mark:03d}",
                         class_=class_)
        self._next_mark += 1
        self.marks.append(mark)
        self.model = None
        return mark

    def mark(self, mark_id: str) -> MarkDraft:
        for mark in self.marks:
            if mark.id == mark_id:
                return mark
        raise KeyError(mark_id)

    def delete_mark(self, mark_id: str) -> None:
        self.marks.remove(self.mark(mark_id))
        self.model = None

    def add_corner(self, mark: MarkDraft, point: ClickedPoint) -> None:
        if len(mark.corners) >= 4:
            raise ValueError(f"{mark.id}: у проёма четыре угла, все уже указаны")
        mark.corners.append(point)
        self.model = None

    def add_reveal_point(self, mark: MarkDraft, side: str, point: ClickedPoint) -> None:
        if mark.reveal_side not in (None, side):
            mark.reveal_points.clear()
        mark.reveal_side = side
        if len(mark.reveal_points) >= 2:
            raise ValueError(f"{mark.id}: у внутренней кромки грани откоса две точки")
        mark.reveal_points.append(point)
        self.model = None

    def element_marks(self, sigma_of: SigmaOf) -> list[ElementMark]:
        """Черновики → `ElementMark` через ЕГО проверку, а не свою.

        σ разметки — худшая по её точкам: тем же правилом ядро берёт худшее
        локальное разрешение по контуру (`elements._gsd_near`).
        """
        out = []
        for mark in self.marks:
            payload = {
                "class": mark.class_, "mounting": mark.mounting,
                "edge_type": mark.edge_type, "id": mark.id,
                "corners_px": [list(map(float, p))
                               for p in order_corners([c.xy for c in mark.corners])],
                "sigma_px": max(sigma_of(c) for c in mark.corners),
            }
            if mark.reveal_side is not None and mark.reveal_points:
                payload["reveal"] = {
                    "side": mark.reveal_side,
                    "inner_edge_px": [list(p.xy) for p in mark.reveal_points],
                    "sigma_px": max(sigma_of(p) for p in mark.reveal_points),
                }
            out.append(ElementMark.model_validate(payload))
        return out

    def compute_elements(self, sigma_of: SigmaOf) -> FacadeModel | None:
        """Модель фасада: масштаб (если его нет) и разметка — без оценки плоскости."""
        self.model = None
        ready, reasons = self.ready_to_measure()
        if not ready:
            self.error = "; ".join(reasons)
            return None
        if self.scale is None and self.compute_scale(sigma_of) is None:
            return None
        self.error = None
        try:
            geometry = run.raster_geometry_for(self.frame, self.scale)
            self.model = run.elements_stage(self.frame, self.scale, geometry,
                                            self.element_marks(sigma_of) or None)
        except ValueError as error:
            self.error = str(error)
        return self.model

    # --- готовность --------------------------------------------------------------

    def ready_to_measure(self, *, require_marks: bool = True) -> tuple[bool, list[str]]:
        """Можно ли считать, и если нет — чего не хватает, каждое своей причиной."""
        if self.image_path is None:
            return False, ["снимок не открыт"]
        reasons = []
        if self.frame is None:
            reasons.append(self.error or "кадр не обработан")
        elif self.frame.plane.needs_operator:
            reasons.append("плоскость фасада не восстановлена: укажите её вручную "
                           "(четыре угла прямоугольника и одно доопределение)")
        ends = len(self.reference.ends)
        if ends != 2:
            reasons.append(f"опорная база: указано концов {ends} из 2")
        if self.reference.span_mm is None:
            reasons.append("длина опорной базы не введена")
        elif not self._span_ok():
            reasons.append("длина опорной базы должна быть положительной")
        if require_marks:
            for mark in self.marks:
                if len(mark.corners) != 4:
                    reasons.append(f"{mark.id}: углов {len(mark.corners)} из 4")
                if mark.reveal_side is not None and len(mark.reveal_points) != 2:
                    reasons.append(f"{mark.id}: точек внутренней кромки откоса "
                                   f"{len(mark.reveal_points)} из 2")
        return not reasons, reasons
