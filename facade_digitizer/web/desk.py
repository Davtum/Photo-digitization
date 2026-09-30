"""Рабочий стол оператора над одним снимком. Веб-интерфейс, задачи 3–5.

Перенос логики окна Qt (`MainWindow` плана 3) без Qt и без HTTP: `Desk` держит
`OperatorSession` и состояние экрана — шаг, постановку точек, выбранный проём,
выровненный вид, байты кадра — и отдаёт всё, что рисует страница, одним словарем
`state()`. Страница ничего не вычисляет: она рисует `state()` и шлёт действия
(`act`). Так путь «клик → миллиметры» проверяется тестами без браузера.

**Постановка точек вместо режимов.** Что делает клик по кадру, определяет
постановка текущего шага (`placing`): углы плоскости, область оценки, концы базы,
углы проёма, кромка откоса. Постановку включает кнопка шага (или сам шаг: база без
концов, проём без углов), и она всегда названа подсказкой над кадром. Вне
постановки клик на шаге «Проёмы» выбирает проём, а точки можно перетаскивать; во
время постановки захват точек выключен — как в окне Qt, где правка была только в
«Навигации»: иначе первая точка кромки откоса у самого угла «схватила» бы угол.

**Пересчёт — сам и полный.** Масштаб (миллисекунды) считается, как только есть оба
конца базы и длина; модель — как только разметка готова; после новой плоскости —
всё по цепочке. Покомпонентного кэша нет (задача 16 плана 3).

**Ревизия.** Каждое действие поднимает `rev`; страница шлёт её с действием, и
сервер отвергает действие по устаревшему состоянию (`web.app`).
"""
import json
import math
import threading
from collections.abc import Callable

import cv2
import numpy as np

from facade_digitizer.pipeline import quality as quality_gate
from facade_digitizer.pipeline import run
from facade_digitizer.ui import session_file, texts
from facade_digitizer.ui.session import ClickedPoint, OperatorSession, order_corners
from facade_digitizer.ui.zoom import (
    RECOMMENDED_MIN_SCALE,
    SIGMA_EDGE_PX,
    SIGMA_SCREEN_PX,
    click_sigma,
)
from facade_digitizer.web import images
from facade_digitizer.web.jobs import JobRunner

STEPS = (("frame", "Снимок"), ("scale", "Масштаб"), ("marks", "Проёмы"),
         ("result", "Результат"))
STEP_KEYS = tuple(k for k, _ in STEPS)

#: Цвета разметки — те же, что в окне Qt.
COLORS = {"plane": "#fdd835", "roi": "#1e88e5", "base": "#43a047", "mark": "#8e24aa",
          "selected": "#ff7043", "reveal": "#00acc1"}

CONSTRAINTS = (("aspect", "Отношение сторон (ширина / высота)"),
               ("sizes", "Два размера, мм"),
               ("calibrated", "Камера откалибрована, стороны ортогональны"))

#: Действия страницы. Только они доступны через `act` — не произвольные методы.
ACTIONS = frozenset({
    "set_step", "click", "drop", "stop_placing",
    "start_manual_plane", "set_plane_constraint", "apply_manual_plane", "start_roi",
    "apply_roi", "reset_plane", "set_profile",
    "restart_base", "set_base",
    "new_mark", "select_mark", "set_mark_attrs", "set_reveal_side", "start_reveal",
    "undo_point", "delete_mark", "export",
})
#: Пока идёт фаза кадра, кадр на странице устарел: разрешён только переход по шагам.
ALLOWED_WHILE_BUSY = frozenset({"set_step"})

_ERRORS = (ValueError, KeyError, IndexError, FileNotFoundError)


class DeskBusy(Exception):
    """Действие во время фазы кадра: точки легли бы на кадр, который сейчас сменится."""


def _num(value):
    """Число для JSON: NaN и бесконечность — `None` (JSON их не знает)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _error_text(error: Exception) -> str:
    text = str(error)
    if isinstance(error, KeyError) and error.args:
        text = f"не найдено: {error.args[0]}"
    return text or type(error).__name__


class Desk:
    def __init__(self, session: OperatorSession, *, runner: JobRunner,
                 save: Callable[[OperatorSession], None] | None = None,
                 lock=None, desk_id: str = "desk"):
        self.session = session
        self.runner = runner
        self._save = save
        self.lock = lock or threading.RLock()
        self.desk_id = desk_id
        self.rev = 0
        self.step = "frame"
        #: Постановка по шагам: незавершённая сохраняется при переходе и продолжается.
        self.placings: dict[str, str | None] = {"frame": None, "scale": None, "marks": None}
        self.current_mark: str | None = session.marks[-1].id if session.marks else None
        self.side_choice: dict[str, str] = {}
        self.plane_constraint = {"kind": "sizes", "aspect": 1.0, "width_mm": 1000.0,
                                 "height_mm": 1000.0}
        self.busy: str | None = None
        self.frame_version = 0
        self.frame_png: bytes | None = None
        self.usable_png: bytes | None = None
        self.preview = None
        self.frame_error: str | None = None
        self.scale_error: str | None = None
        self.base_result: str | None = None
        self.warning: str | None = None
        self.notice: dict | None = None
        self.export_info: dict | None = None
        self._export_snapshot: str | None = None
        self._rect: dict = {"status": "none", "scale": None}
        self.rect_version = 0
        self.rect_jpg: bytes | None = None
        self.rect_mask_png: bytes | None = None
        self.save_status = {"ok": True, "error": None}
        self._saved_snapshot = self._snapshot()
        self._first_frame = True
        self._state_cache: tuple[int, dict] | None = None

    # --- действия ---------------------------------------------------------------

    @property
    def placing(self) -> str | None:
        return self.placings.get(self.step)

    def _set_placing(self, value: str | None, step: str | None = None) -> None:
        self.placings[step or self.step] = value

    def act(self, name: str, **args):
        """Действие страницы. Ошибка оператора — текстом в `notice`, а не исключением."""
        if name not in ACTIONS:
            raise KeyError(f"неизвестное действие: {name}")
        with self.lock:
            if self.busy is not None and name not in ALLOWED_WHILE_BUSY:
                raise DeskBusy(self.busy)
            self.notice = None
            result = None
            try:
                result = getattr(self, f"_a_{name}")(**args)
            except _ERRORS as error:
                self.notice = {"kind": "error", "text": _error_text(error)}
            finally:
                self._touch()
            return result

    def _bump(self) -> None:
        self.rev += 1
        self._state_cache = None

    def _touch(self) -> None:
        """После действия: новая ревизия и автосохранение, если сессия изменилась."""
        self._bump()
        snapshot = self._snapshot()
        if snapshot == self._saved_snapshot or self._save is None:
            return
        try:
            self._save(self.session)
        except (OSError, ValueError) as error:
            self.save_status = {"ok": False, "error": _error_text(error)}
        else:
            self.save_status = {"ok": True, "error": None}
            self._saved_snapshot = snapshot

    def _snapshot(self) -> str | None:
        """Сессия без хронометража (но с меткой оператора) — для «изменилось ли»
        (сохранение, выгрузка): секунды меняются сами, метка — решение оператора."""
        if self.session.image_path is None:
            return None
        raw = session_file.session_to_dict(self.session)
        raw["operator"] = raw.pop("timing", {}).get("operator")
        return json.dumps(raw, sort_keys=True, ensure_ascii=False)

    # --- кадр ---------------------------------------------------------------------

    def start(self) -> None:
        """Фаза кадра для открытого (или восстановленного) снимка — в фоне."""
        with self.lock:
            s = self.session
            path, profile, plane, roi = s.image_path, s.profile_path, s.plane_override, s.roi
            self._run_frame(lambda: run.frame_stage(path, profile_path=profile, with_color=True,
                                                    plane_override=plane, roi=roi),
                            f"{path.name}: обрабатывается — кадр, калибровка, плоскость…",
                            plane_override=plane, roi=roi)

    def _run_frame(self, fn, message: str, *, plane_override=None, roi=None) -> None:
        old = self.session.frame
        self.busy = message
        self._bump()

        def job():
            fs = fn()
            new_image = old is None or fs.frame is not old.frame
            png = images.png(fs.frame.color) if new_image else None
            preview, usable = self._preview(fs)
            return fs, png, preview, usable

        def done(result):
            with self.lock:
                self._frame_ready(*result, plane_override=plane_override, roi=roi)

        def failed(message_text: str):
            with self.lock:
                self.busy = None
                self.frame_error = message_text
                self.session.error = message_text
                self.notice = {"kind": "error", "text": f"Кадр не обработан: {message_text}"}
                self._bump()

        self.runner.submit("frame", job, done, failed)

    @staticmethod
    def _preview(fs):
        if fs.plane.needs_operator:
            return None, None
        preview = run.theta_preview(fs)
        return preview, images.usable_png(preview.usable_mask)

    def _frame_ready(self, fs, png, preview, usable, *, plane_override=None, roi=None) -> None:
        s = self.session
        s.accept_frame(fs, plane_override=plane_override, roi=roi)
        if png is not None:
            self.frame_png = png
            self.frame_version += 1
        self.preview, self.usable_png = preview, usable
        self.busy = None
        self.frame_error = None
        if fs.plane.needs_operator:
            self.step = "frame"
            if self.placings["frame"] != "plane":
                self.placings["frame"] = "plane"
        elif self.placings["frame"] in ("plane", "roi"):
            self.placings["frame"] = None
        if self._first_frame:
            self._first_frame = False
            if not fs.plane.needs_operator:
                self.step = ("marks" if s.marks else
                             "scale" if s.reference.ends else "frame")
            if self.step == "scale" and len(s.reference.ends) < 2:
                self.placings["scale"] = "base"
        self._recompute_scale()
        self._bump()

    # --- шаги и постановки -----------------------------------------------------------

    def _a_set_step(self, step: str) -> None:
        if step not in STEP_KEYS:
            raise ValueError(f"нет такого шага: {step}")
        if step != "frame" and self.session.frame is None:
            raise ValueError("Сначала дождитесь обработки снимка.")
        self.step = step
        if step == "scale" and self.placings["scale"] is None \
                and len(self.session.reference.ends) < 2:
            self.placings["scale"] = "base"

    def _a_stop_placing(self) -> None:
        placing = self.placing
        if placing == "corners" and self.current_mark is not None:
            mark = self.session.mark(self.current_mark)
            if not mark.corners:
                self.session.delete_mark(mark.id)
                self.current_mark = self.session.marks[-1].id if self.session.marks else None
        if placing == "reveal" and self.current_mark is not None:
            mark = self.session.mark(self.current_mark)
            if not mark.reveal_points:
                mark.reveal_side = None
        self._set_placing(None)
        self._recompute_model()

    def _point(self, x, y, view_scale) -> ClickedPoint:
        x, y, s = float(x), float(y), float(view_scale)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(s) and s > 0):
            raise ValueError("точка: координаты и масштаб должны быть конечными, "
                             "масштаб — положительным")
        fs = self.session.frame
        if fs is None:
            raise ValueError("кадр ещё не обработан")
        w, h = fs.image_size
        if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
            raise ValueError("точка вне снимка")
        return ClickedPoint(x, y, s)

    def _a_click(self, x, y, view_scale) -> None:
        point = self._point(x, y, view_scale)
        s = self.session
        placing = self.placing
        if placing == "plane":
            s.add_plane_point(point)
        elif placing == "roi":
            s.add_roi_point(point)
        elif placing == "base":
            s.add_reference_end(point)
            if len(s.reference.ends) == 2:
                self._set_placing(None)
            self._recompute_scale()
        elif placing == "corners":
            mark = self._current()
            s.add_corner(mark, point)
            self.warning = texts.scale_warning(point, s.local_gsd(point))
            if len(mark.corners) == 4:
                self._set_placing(None)
                self.notice = {"kind": "info", "text": (
                    f"{mark.id}: четыре угла указаны — проверьте установку и тип кромки: "
                    "от типа кромки зависит, выпускается ли признак допуска.")}
            self._recompute_model()
        elif placing == "reveal":
            mark = self._current()
            s.add_reveal_point(mark, self._side_for(mark), point)
            self.warning = texts.scale_warning(point, s.local_gsd(point))
            if len(mark.reveal_points) == 2:
                self._set_placing(None)
            self._recompute_model()
        elif self.step == "marks":
            self._select_at(point)

    def _select_at(self, point: ClickedPoint) -> None:
        for mark in reversed(self.session.marks):
            if len(mark.corners) != 4:
                continue
            quad = order_corners([c.xy for c in mark.corners]).astype(np.float32)
            if cv2.pointPolygonTest(quad.reshape(-1, 1, 2), point.xy, False) >= 0:
                self.current_mark = mark.id
                return

    # --- плоскость -------------------------------------------------------------------

    def _a_start_manual_plane(self) -> None:
        self.step = "frame"
        self.session.plane_points.clear()
        self._set_placing("plane")

    def _a_start_roi(self) -> None:
        self.step = "frame"
        self.session.roi_points.clear()
        self._set_placing("roi")

    def _a_set_plane_constraint(self, kind: str, aspect=None, width_mm=None,
                                height_mm=None) -> None:
        if kind not in dict(CONSTRAINTS):
            raise ValueError(f"нет такого доопределения: {kind}")
        values = {"aspect": aspect, "width_mm": width_mm, "height_mm": height_mm}
        for key, value in values.items():
            if value is None:
                continue
            value = float(value)
            if not (math.isfinite(value) and value > 0):
                raise ValueError("доопределение плоскости: значения должны быть положительными")
            self.plane_constraint[key] = value
        self.plane_constraint["kind"] = kind

    def _constraint_args(self) -> dict:
        c = self.plane_constraint
        if c["kind"] == "aspect":
            return {"aspect_ratio": float(c["aspect"])}
        if c["kind"] == "sizes":
            return {"size_mm": (float(c["width_mm"]), float(c["height_mm"]))}
        return {"assume_calibrated": True}

    def _a_apply_manual_plane(self) -> None:
        """Ручная плоскость — мгновенно: снимок не перечитывается (`run.replace_plane`)."""
        fs = self._frame()
        try:
            override = self.session.manual_plane(**self._constraint_args())
            new_fs = run.replace_plane(fs, plane_override=override)
        except ValueError as error:
            raise ValueError(f"Ручная плоскость не применена: {error}") from error
        preview, usable = self._preview(new_fs)
        self._frame_ready(new_fs, None, preview, usable, plane_override=override)
        self._set_placing(None, "frame")

    def _a_apply_roi(self) -> None:
        fs = self._frame()
        roi = self.session.roi_polygon()
        self._set_placing(None, "frame")
        self._run_frame(lambda: run.replace_plane(fs, roi=roi),
                        "точки схода оцениваются по области…", roi=roi)

    def _a_reset_plane(self) -> None:
        fs = self._frame()
        self.session.plane_points.clear()
        self.session.roi_points.clear()
        self._set_placing(None, "frame")
        self._run_frame(lambda: run.replace_plane(fs),
                        "точки схода оцениваются по всему кадру…")

    def _a_set_profile(self, path=None, discard: bool = False) -> None:
        """Профиль сдвигает кадр (снятие дисторсии): точки прежнего кадра теряют смысл.
        Без `discard` при указанных точках — вопрос оператору, точки остаются."""
        s = self.session
        try:
            s.set_profile(path, discard_clicks=bool(discard))
        except ValueError as error:
            self.notice = {"kind": "confirm", "text": (
                f"Профиль сдвигает кадр (снятие дисторсии), и указанные точки к нему не "
                f"относятся: {error}. Сбросить разметку и продолжить?"),
                "confirm": {"name": "set_profile", "args": {"path": path, "discard": True}}}
            return
        self.current_mark = None
        self.side_choice.clear()
        self.placings = {"frame": None, "scale": None, "marks": None}
        self.step = "frame"
        self.preview = self.usable_png = None
        self.export_info = None
        self._drop_raster()
        name = s.profile_path.name if s.profile_path is not None else "без профиля"
        self._run_frame(lambda: run.frame_stage(s.image_path, profile_path=s.profile_path,
                                                with_color=True),
                        f"Профиль {name}: кадр пересчитывается…")

    def _frame(self):
        if self.session.frame is None:
            raise ValueError("кадр ещё не обработан")
        return self.session.frame

    # --- опорная база ---------------------------------------------------------------

    def _a_restart_base(self) -> None:
        self.session.clear_reference()
        self.step = "scale"
        self._set_placing("base")
        self._recompute_scale()

    def _a_set_base(self, span_mm=None, span_sigma_mm=None,
                    scale_source: str = "operator_reference",
                    origin_is_facade_corner: bool = False) -> None:
        if scale_source not in dict(texts.SCALE_SOURCES):
            raise ValueError(f"нет такого происхождения длины: {scale_source}")
        span = _num(span_mm) if span_mm is not None else None
        sigma = _num(span_sigma_mm) if span_sigma_mm is not None else None
        self.session.set_span_mm(span)
        self.session.set_base_options(span_sigma_mm=sigma if sigma and sigma > 0 else None,
                                      scale_source=scale_source,
                                      origin_is_facade_corner=bool(origin_is_facade_corner))
        self._recompute_scale()

    def _recompute_scale(self) -> None:
        """Масштаб — миллисекунды: плоскость не переоценивается. Затем растр и модель."""
        s = self.session
        s.scale = s.model = None
        self.base_result = None
        ready = (s.frame is not None and not s.frame.plane.needs_operator
                 and len(s.reference.ends) == 2 and s._span_ok())
        if not ready:
            self.scale_error = ("; ".join(s.ready_to_measure(require_marks=False)[1])
                                if s.image_path is not None else None)
            self._drop_raster()
            return
        ss = s.compute_scale(click_sigma)
        if ss is None:
            self.scale_error = s.error
            self._drop_raster()
            return
        self.scale_error = None
        self.base_result = texts.base_result_text(ss, s.reference)
        self._start_raster(ss)
        self._recompute_model()

    # --- выровненный вид -----------------------------------------------------------

    def _drop_raster(self) -> None:
        self.runner.cancel("raster")
        self._rect = {"status": "none", "scale": None}

    def _start_raster(self, ss) -> None:
        """Пиксели растра — в фоне (до 4 с); геометрия та же, что уйдёт в выходной файл."""
        fs = self.session.frame
        self._rect = {"status": "pending", "scale": ss}

        def job():
            r = run.raster_stage(fs, run.raster_geometry_for(fs, ss), color=True)
            return r, images.jpeg(r.image, 88), images.invalid_png(r.valid_mask)

        def done(result):
            with self.lock:
                if self.session.scale is not ss:
                    return
                raster, jpg, mask = result
                self.rect_jpg, self.rect_mask_png = jpg, mask
                self.rect_version += 1
                self._rect = {"status": "ready", "scale": ss, "raster": raster,
                              "coverage": float(np.asarray(raster.valid_mask).mean())}
                self._bump()

        def failed(message: str):
            with self.lock:
                if self.session.scale is ss:
                    self._rect = {"status": "error", "scale": ss,
                                  "text": f"Выровненный вид не построен: {message}"}
                    self._bump()

        self.runner.submit("raster", job, done, failed)

    # --- проёмы ---------------------------------------------------------------------

    def _current(self):
        if self.current_mark is None:
            raise ValueError("Сначала создайте проём: «+ Окно» или «+ Дверь».")
        return self.session.mark(self.current_mark)

    def _side_for(self, mark) -> str:
        if mark.id in self.side_choice:
            return self.side_choice[mark.id]
        if mark.reveal_side is not None:
            return mark.reveal_side
        sides = self.session.suggested_reveal_sides(mark)
        return sides[0] if sides is not None else "left"

    def _a_new_mark(self, class_: str = "window") -> None:
        if class_ not in dict(texts.CLASSES):
            raise ValueError(f"нет такого класса проёма: {class_}")
        s = self.session
        previous = s.marks[-1] if s.marks else None
        mark = s.new_mark(class_)
        if previous is not None:
            # Установка и тип кромки — как у предыдущего: в окне Qt списки тоже
            # сохраняли выбор между проёмами.
            mark.mounting, mark.edge_type = previous.mounting, previous.edge_type
        self.current_mark = mark.id
        self.step = "marks"
        self._set_placing("corners")
        self.warning = (f"{mark.id}: укажите четыре угла, порядок не важен. Рекомендуемое "
                        f"увеличение у углов — не менее {RECOMMENDED_MIN_SCALE:g}:1.")
        self._recompute_model()

    def _a_select_mark(self, id: str) -> None:
        mark = self.session.mark(id)
        self.current_mark = mark.id
        if self.placing in ("corners", "reveal"):
            self._set_placing("corners" if len(mark.corners) < 4 else None)

    def _a_set_mark_attrs(self, id: str, mounting: str | None = None,
                          edge_type: str | None = None) -> None:
        mark = self.session.mark(id)
        if mounting is not None:
            if mounting not in dict(texts.MOUNTINGS):
                raise ValueError(f"нет такого способа установки: {mounting}")
            mark.mounting = mounting
        if edge_type is not None:
            if edge_type not in dict(texts.EDGES):
                raise ValueError(f"нет такого типа кромки: {edge_type}")
            mark.edge_type = edge_type
        self.session.model = None
        self._recompute_model()

    def _a_set_reveal_side(self, side: str) -> None:
        if side not in dict(texts.SIDES):
            raise ValueError(f"нет такой грани: {side}")
        mark = self._current()
        if mark.reveal_points and mark.reveal_side != side:
            old = dict(texts.SIDES)[mark.reveal_side]
            mark.reveal_points.clear()
            mark.reveal_side = None
            self.notice = {"kind": "info", "text": (
                f"{mark.id}: точки откоса ({old} грань) стёрты — выбрана другая грань.")}
        self.side_choice[mark.id] = side
        self._recompute_model()

    def _a_start_reveal(self) -> None:
        mark = self._current()
        if len(mark.corners) < 4:
            raise ValueError(f"{mark.id}: сначала укажите четыре угла проёма.")
        side = self._side_for(mark)
        self.side_choice[mark.id] = side
        mark.reveal_points.clear()
        mark.reveal_side = None
        self.step = "marks"
        self._set_placing("reveal")
        self._recompute_model()

    def _a_undo_point(self) -> None:
        mark = self._current()
        self.session.undo_last_point(mark)
        if len(mark.corners) < 4:
            self._set_placing("corners", "marks")
        elif mark.reveal_side is not None and len(mark.reveal_points) < 2:
            self._set_placing("reveal", "marks")
        self._recompute_model()

    def _a_delete_mark(self, id: str | None = None) -> None:
        mark_id = id or self._current().id
        self.session.delete_mark(mark_id)
        self.side_choice.pop(mark_id, None)
        if self.current_mark == mark_id:
            self.current_mark = self.session.marks[-1].id if self.session.marks else None
            self._set_placing(None, "marks")
        self._recompute_model()

    def _recompute_model(self) -> None:
        """Модель — по готовой разметке, полным пересчётом через фазу элементов."""
        s = self.session
        s.model = None
        if s.scale is None or not s.ready_to_measure()[0]:
            return
        s.compute_elements(click_sigma)

    # --- правка ---------------------------------------------------------------------

    def _a_drop(self, key: str, x, y, view_scale, moved: bool = True) -> None:
        """Отпускание перенесённой точки. Во время переноса точка движется только на
        странице; отпускание без смещения — не правка: ни события `edit`, ни пересчёта."""
        if not moved:
            return
        if self.placing is not None:
            raise ValueError("Во время постановки точек правка выключена — нажмите Esc.")
        point = self._point(x, y, view_scale)
        if key.startswith("base:"):
            self.session.move_reference_end(int(key.split(":", 1)[1]), point)
            self._recompute_scale()
            return
        mark_id, kind, index = key.rsplit(":", 2)
        mark = self.session.mark(mark_id)
        points = mark.corners if kind == "corner" else mark.reveal_points
        if kind not in ("corner", "reveal") or not 0 <= int(index) < len(points):
            raise ValueError(f"нет такой точки: {key}")
        self.session.move_point(key, point)
        self.current_mark = mark_id
        self.warning = texts.scale_warning(point, self.session.local_gsd(point))
        self._recompute_model()

    # --- выгрузка -------------------------------------------------------------------

    def _a_export(self):
        s = self.session
        out_dir = s.image_path.parent / "экспорт"
        try:
            out = session_file.export(s, out_dir, click_sigma)
        except ValueError as error:
            raise ValueError(f"Экспорт не выполнен: {error}") from error
        files = [p for p in (out.json_path, out.dxf_path, out.marks_path,
                             out.json_path.with_suffix(".command.txt")) if p is not None]
        self.export_info = {"dir": str(out.json_path.parent),
                            "files": [{"name": p.name, "url": f"api/export/{p.name}"}
                                      for p in files],
                            "cli": out.cli_line}
        self._export_snapshot = self._snapshot()
        self.notice = {"kind": "info", "text": f"Выгружено в {out.json_path.parent}"}
        return out

    # --- состояние ------------------------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            if self._state_cache is not None and self._state_cache[0] == self.rev:
                return self._state_cache[1]
            state = self._build_state()
            self._state_cache = (self.rev, state)
            return state

    def _build_state(self) -> dict:
        s = self.session
        return {
            "desk": self.desk_id,
            "rev": self.rev,
            "image": {"name": s.image_path.name if s.image_path else None},
            "busy": self.busy,
            "step": self.step,
            "steps": self._steps(),
            "placing": self.placing,
            "hint": self._hint(),
            "frame": self._frame_state(),
            "quality": self._quality(),
            "overlays": self._overlays(),
            "draggable": self._draggable(),
            "scale": self._scale_state(),
            "marks": self._marks_state(),
            "result": self._result_state(),
            "rectified": self._rect_state(),
            "sigma": {"screen_px": SIGMA_SCREEN_PX, "edge_px": SIGMA_EDGE_PX,
                      "recommended_min_scale": RECOMMENDED_MIN_SCALE},
            "choices": {"classes": texts.CLASSES, "mountings": texts.MOUNTINGS,
                        "edges": texts.EDGES, "sides": texts.SIDES,
                        "scale_sources": texts.SCALE_SOURCES, "constraints": CONSTRAINTS},
            "notice": self.notice,
            "save": dict(self.save_status),
        }

    def _steps(self) -> list[dict]:
        s = self.session
        fs = s.frame
        frame = ("todo" if fs is None and self.frame_error is None else
                 "attention" if fs is None or fs.plane.needs_operator else "done")
        scale = "done" if s.scale is not None else "todo"
        if s.marks and s.model is not None:
            marks = "done"
        elif s.marks:
            marks = "attention"
        else:
            marks = "todo"
        if self.export_info is None:
            result = "todo"
        else:
            result = "done" if self._export_snapshot == self._snapshot() else "attention"
        status = {"frame": frame, "scale": scale, "marks": marks, "result": result}
        return [{"key": k, "title": t, "status": status[k]} for k, t in STEPS]

    def _hint(self) -> dict:
        s = self.session
        placing = self.placing
        if self.busy:
            return {"text": self.busy, "placing": False}
        if placing == "plane":
            n = len(s.plane_points)
            text = (f"Кликните угол заведомого прямоугольника в плоскости стены: {n} из 4. "
                    "Порядок не важен." if n < 4 else
                    "Четыре угла указаны — выберите доопределение и нажмите "
                    "«Применить плоскость».")
            return {"text": text, "placing": True, "count": n, "total": 4}
        if placing == "roi":
            n = len(s.roi_points)
            return {"text": f"Обведите область оценки: точек {n} (нужно не менее 3), затем "
                            "«Оценить по области».", "placing": True, "count": n, "total": None}
        if placing == "base":
            n = len(s.reference.ends)
            text = ("Кликните первый конец опорной базы — он же начало отсчёта." if n == 0
                    else "Кликните второй конец опорной базы — как можно дальше от первого.")
            return {"text": text, "placing": True, "count": n, "total": 2}
        if placing == "corners":
            mark = s.mark(self.current_mark)
            return {"text": f"Кликните угол проёма {mark.id}: {len(mark.corners)} из 4. "
                            f"Порядок не важен; увеличение у углов — не менее "
                            f"{RECOMMENDED_MIN_SCALE:g}:1.",
                    "placing": True, "count": len(mark.corners), "total": 4}
        if placing == "reveal":
            mark = s.mark(self.current_mark)
            side = dict(texts.SIDES)[self._side_for(mark)]
            return {"text": f"Кликните внутреннюю кромку откоса ({side} грань) проёма "
                            f"{mark.id}: {len(mark.reveal_points)} из 2.",
                    "placing": True, "count": len(mark.reveal_points), "total": 2}
        text = {
            "frame": "Колесо — масштаб, перетаскивание — панорама.",
            "scale": "Концы базы можно перетаскивать. Колесо — масштаб.",
            "marks": "Кликните внутри проёма, чтобы выбрать его; точки можно перетаскивать.",
            "result": "Наведите на строку таблицы — проём подсветится на снимке.",
        }[self.step]
        return {"text": text, "placing": False}

    def _frame_state(self) -> dict | None:
        s = self.session
        fs = s.frame
        profile = ({"path": str(s.profile_path), "name": s.profile_path.name}
                   if s.profile_path is not None else None)
        if fs is None:
            return {"ready": False, "error": self.frame_error, "profile": profile,
                    "version": self.frame_version}
        plane = fs.plane
        manual = plane.method == "manual_four_point"
        w, h = fs.image_size
        info = [f"{w} × {h} px", f"поворот EXIF {fs.frame.orientation}",
                f"камера: {texts.CALIBRATION_TEXT.get(fs.camera_record.calibration, '')}"]
        if plane.needs_operator:
            reasons = "; ".join(plane.confidence.reasons) or "доверие ниже порога"
            info.append(f"плоскость не восстановлена ({reasons}) — укажите её вручную")
            plane_text = ("Плоскость не восстановлена автоматически. Укажите четыре угла "
                          "заведомого прямоугольника в плоскости стены и одно доопределение.")
        else:
            info.append("плоскость задана оператором" if manual
                        else f"доверие к плоскости {plane.confidence.value:.2f}")
            plane_text = (("Плоскость задана оператором." if manual else
                           f"Плоскость оценена по точкам схода, доверие "
                           f"{plane.confidence.value:.2f}.")
                          + " Переопределить её можно в любой момент.")
        if self.preview is not None:
            info.append(f"пригодно по углу {self.preview.usable_fraction:.0%} кадра")
            half = self.preview.half_turn
            info.append("поворот на полкруга: " + (
                "задан оператором" if half.resolved
                else "не разрешён, принят «снимок не перевёрнут»"))
        return {
            "ready": True, "version": self.frame_version, "width": w, "height": h,
            "name": fs.path.name, "info": info, "profile": profile,
            "usable": self.usable_png is not None, "error": None,
            "plane": {"needs_operator": bool(plane.needs_operator), "manual": manual,
                      "confidence": (None if plane.needs_operator or manual
                                     else _num(plane.confidence.value)),
                      "text": plane_text, "points": len(s.plane_points),
                      "roi_points": len(s.roi_points), "roi": s.roi is not None,
                      "constraint": dict(self.plane_constraint)},
        }

    def _quality(self) -> dict | None:
        s = self.session
        if s.scale is not None:
            q = s.scale.quality
            return {"stage": "final", "stage_text": texts.QUALITY_STAGE_TEXT["final"],
                    "verdict": q.verdict, "verdict_text": texts.VERDICT_TEXT[q.verdict],
                    "reasons": list(q.reasons), "note": texts.quality_note("final", q.verdict)}
        fs = s.frame
        if fs is None or self.preview is None:
            return None
        manual = fs.plane.method == "manual_four_point"
        confidence = ("плоскость задана оператором" if manual
                      else f"доверие к плоскости {fs.plane.confidence.value:.2f}")
        try:
            pre = quality_gate.assess_preliminary(fs.sharpness, self.preview.usable_fraction)
        except ValueError as error:
            return {"stage": "preliminary",
                    "stage_text": texts.QUALITY_STAGE_TEXT["preliminary"],
                    "verdict": None, "verdict_text": "не определён",
                    "reasons": [str(error)], "note": ""}
        return {"stage": "preliminary", "stage_text": texts.QUALITY_STAGE_TEXT["preliminary"],
                "verdict": pre.verdict, "verdict_text": texts.VERDICT_TEXT[pre.verdict],
                "reasons": list(pre.reasons),
                "note": texts.quality_note("preliminary", pre.verdict, confidence=confidence),
                "usable_fraction": _num(self.preview.usable_fraction)}

    def _overlays(self) -> list[dict]:
        s = self.session
        out = []

        def add(key, points, color, closed=False, kind="line"):
            if points:
                out.append({"key": key, "points": [[float(x), float(y)] for x, y in points],
                            "closed": bool(closed), "color": color, "kind": kind})

        if self.step == "frame":
            pts = [p.xy for p in s.plane_points]
            add("plane", pts, COLORS["plane"], closed=len(pts) == 4)
            add("roi", [p.xy for p in s.roi_points], COLORS["roi"], closed=True)
        if self.step in ("scale", "marks", "result"):
            add("base", [p.xy for p in s.reference.ends], COLORS["base"])
        if self.step in ("marks", "result"):
            for m in s.marks:
                color = COLORS["selected"] if m.id == self.current_mark else COLORS["mark"]
                pts = [c.xy for c in m.corners]
                if len(pts) == 4:
                    pts = [tuple(p) for p in order_corners(pts)]
                add(f"mark:{m.id}", pts, color, closed=len(pts) == 4)
                add(f"reveal:{m.id}", [p.xy for p in m.reveal_points], COLORS["reveal"])
        return out

    def _draggable(self) -> list[dict]:
        if self.placing is not None or self.busy:
            return []
        s = self.session
        if self.step == "scale":
            return [{"key": f"base:{i}", "x": p.x, "y": p.y}
                    for i, p in enumerate(s.reference.ends)]
        if self.step == "marks":
            return [{"key": key, "x": x, "y": y} for key, x, y in s.editable_points()]
        return []

    def _scale_state(self) -> dict:
        r = self.session.reference
        return {"ends": len(r.ends), "end_lines": texts.end_sigma_lines(r.ends),
                "values": {"span_mm": _num(r.span_mm), "span_sigma_mm": _num(r.span_sigma_mm),
                           "scale_source": r.scale_source,
                           "origin_is_facade_corner": r.origin_is_facade_corner},
                "result": self.base_result,
                "tolerance_ok": (self.session.scale.sigma_rel <= run.SIGMA_REL_TOLERANCE
                                 if self.session.scale is not None else None),
                "error": self.scale_error if self.session.scale is None else None}

    def _marks_state(self) -> dict:
        s = self.session
        sizes = {}
        if s.model is not None:
            for e in s.model.elements:
                z = e.size_mm
                sizes[e.id] = (f"{z.width:.0f} ± {z.sigma_width:.0f} × "
                               f"{z.height:.0f} ± {z.sigma_height:.0f} мм")
        classes, sides = dict(texts.CLASSES), dict(texts.SIDES)
        items = [{"id": m.id, "class": m.class_, "class_text": classes.get(m.class_, m.class_),
                  "mounting": m.mounting, "edge_type": m.edge_type,
                  "corners": len(m.corners), "reveal_side": m.reveal_side,
                  "reveal_side_text": sides.get(m.reveal_side) if m.reveal_side else None,
                  "reveal_points": len(m.reveal_points), "size": sizes.get(m.id)}
                 for m in s.marks]
        selected = suggestion = side_choice = None
        if self.current_mark is not None:
            mark = s.mark(self.current_mark)
            selected = mark.id
            side_choice = self._side_for(mark) if len(mark.corners) == 4 else None
            suggestion = (texts.reveal_suggestion(s.suggested_reveal_sides(mark))
                          if len(mark.corners) == 4 else None)
        reasons = [] if s.image_path is None else [
            r for r in s.ready_to_measure()[1] if s.marks]
        return {"items": items, "selected": selected, "side_choice": side_choice,
                "suggestion": suggestion, "warning": self.warning,
                "reasons": reasons}

    def _result_state(self) -> dict:
        model = self.session.model
        return {"columns": list(texts.RESULT_COLUMNS),
                "header": texts.result_header(model) if model is not None else None,
                "rows": texts.result_rows(model) if model is not None else [],
                "export": self.export_info}

    def _rect_state(self) -> dict:
        s = self.session
        rect = self._rect
        if s.scale is None or rect.get("scale") is not s.scale:
            return {"status": "none" if s.scale is None else "pending",
                    "text": "Выровненный вид появится после опорной базы"}
        if rect["status"] == "pending":
            return {"status": "pending", "text": "Выровненный вид строится…"}
        if rect["status"] == "error":
            return {"status": "error", "text": rect["text"]}
        raster = rect["raster"]
        H = np.asarray(raster.H, dtype=float)
        h, w = np.asarray(raster.valid_mask).shape
        coverage = rect["coverage"]
        return {"status": "ready", "version": self.rect_version, "width": int(w),
                "height": int(h), "H": H.tolist(), "H_inv": np.linalg.inv(H).tolist(),
                "mm_per_px": _num(raster.mm_per_px), "coverage": coverage,
                "text": (f"Выровненный вид, {raster.mm_per_px:.1f} мм/px; охват растра "
                         f"{coverage:.0%} (штриховка — вне снимка). Только для просмотра: "
                         "клик центрирует снимок, измерение — на снимке (п. 6.5).")}
