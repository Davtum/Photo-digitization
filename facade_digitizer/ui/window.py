"""Главное окно приложения оператора. План 3, задачи 1, 9, 10.

Окно — тонкий слой над `ui.session.OperatorSession`: оно показывает сессию и
посылает ей события. Клик по холсту уходит в сессию по текущему РЕЖИМУ инструмента
(`handle_click`); тот же метод вызывают тесты, поэтому путь от клика до миллиметров
проверяется без мыши. Долгие фазы идут в фоне (`ui.worker`), окно при этом не замирает.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from facade_digitizer.pipeline import quality as quality_gate
from facade_digitizer.pipeline import run
from facade_digitizer.ui.canvas import FrameCanvas
from facade_digitizer.ui.panels import SidePanel
from facade_digitizer.ui.rectified import RectifiedView
from facade_digitizer.ui.session import ClickedPoint, OperatorSession
from facade_digitizer.ui.worker import run_in_background
from facade_digitizer.ui.zoom import (
    RECOMMENDED_MIN_SCALE,
    click_sigma,
    localisation_cost_mm,
    sigma_image_px,
)

TITLE = "Оцифровка фасада"
IMAGE_FILTER = "Снимки (*.jpg *.jpeg *.png *.tif *.tiff);;Все файлы (*)"

#: Режимы инструмента: что значит клик по холсту.
MODES = {
    "navigate": "Навигация",
    "plane": "Углы ручной плоскости",
    "roi": "Область оценки точек схода",
    "base": "Концы опорной базы",
    "opening": "Углы проёма",
    "reveal": "Внутренняя кромка откоса",
}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.session = OperatorSession()
        self.canvas = FrameCanvas(self)
        # Выровненный вид — рядом с кадром (задача 14): только просмотр и навигация.
        self.rect_view = RectifiedView(self)
        self.rect_label = QLabel("Выровненный вид появится после опорной базы")
        self.rect_label.setWordWrap(True)
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.rect_label)
        right_layout.addWidget(self.rect_view, 1)
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(right)
        self.splitter.setSizes([700, 500])
        self.setCentralWidget(self.splitter)
        self.rect_view.frameTarget.connect(self.center_frame_on)
        self._raster_job = None
        self.rectified = None
        # Прежние имена каркаса задачи 1.
        self.view, self.scene = self.canvas, self.canvas.scene()

        self.side = SidePanel(self)
        # Прокрутка, а не сжатие: колонка выше экрана ноутбука (около 1180 px), и без
        # прокрутки на 1366×768 Qt ужимал бы поля ввода до нечитаемых.
        scroll = QScrollArea(self)
        scroll.setWidget(self.side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(self.side.sizeHint().width() + 24)
        dock = QDockWidget("Оператор", self)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.status_label = QLabel("Снимок не открыт")
        self.scale_label = QLabel("")
        self.mode_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.mode_label)
        self.statusBar().addPermanentWidget(self.scale_label)
        self.canvas.scaleChanged.connect(self._show_scale)
        self.canvas.pointClicked.connect(self.handle_click)
        self._job = None
        self.current_mark: str | None = None
        self.mode = "navigate"
        self._build_actions()
        self._connect_panels()
        self.set_mode("navigate")

    # --- действия и режимы -------------------------------------------------------

    def _build_actions(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        view = self.menuBar().addMenu("Вид")
        for parent, text, keys, slot in (
                (menu, "Открыть снимок…", QKeySequence.Open, self._ask_open),
                (view, "Масштаб 1:1", "Ctrl+1", self.canvas.one_to_one),
                (view, "Вписать в окно", "Ctrl+0", self.canvas.fit_to_window)):
            action = QAction(text, self)
            action.setShortcut(keys)
            action.triggered.connect(slot)
            parent.addAction(action)
            self.addAction(action)
        tools = self.addToolBar("Инструмент")
        self.mode_group = QActionGroup(self)
        self.mode_actions = {}
        for key, text in MODES.items():
            action = QAction(text, self, checkable=True)
            action.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            self.mode_group.addAction(action)
            tools.addAction(action)
            self.mode_actions[key] = action
        self.mode_actions["navigate"].setShortcut("Esc")

    def _connect_panels(self) -> None:
        plane = self.side.plane
        plane.pickCorners.connect(lambda: self.set_mode("plane"))
        plane.applyManual.connect(self.apply_manual_plane)
        plane.pickRoi.connect(lambda: self.set_mode("roi"))
        plane.applyRoi.connect(self.apply_roi)
        plane.resetAuto.connect(self.reset_plane)
        base = self.side.base
        base.pickEnds.connect(lambda: self.set_mode("base"))
        base.apply.connect(self.apply_base)
        marks = self.side.marks
        marks.newMark.connect(self.new_mark)
        marks.pickReveal.connect(self.pick_reveal)
        marks.undo.connect(self.undo_point)
        marks.delete.connect(self.delete_mark)
        marks.selected.connect(self.select_mark)
        marks.compute.connect(self.compute)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"неизвестный режим инструмента: {mode}")
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        self.mode_label.setText(f"Режим: {MODES[mode]}")
        if mode == "plane":
            self.session.plane_points.clear()
            self._draw_plane_points()
        if mode == "roi":
            self.session.roi_points.clear()
            self.canvas.clear_overlay("roi")
        if mode == "base":
            self.session.clear_reference()
            self._draw_base()

    def handle_click(self, x: float, y: float, view_scale: float) -> None:
        """Клик по кадру (координаты конвейера) — в сессию по текущему режиму."""
        point = ClickedPoint(x, y, view_scale)
        if self.mode in ("opening", "reveal") and self.current_mark is None:
            self.status_label.setText("Сначала создайте проём: «Новый проём» на панели.")
            return
        try:
            if self.mode == "plane":
                self.session.add_plane_point(point)
                self._draw_plane_points()
            elif self.mode == "roi":
                self.session.add_roi_point(point)
                self.canvas.set_overlay("roi", [p.xy for p in self.session.roi_points],
                                        closed=True, color="#1e88e5")
            elif self.mode == "opening":
                mark = self.session.mark(self.current_mark)
                self.session.add_corner(mark, point)
                self._warn_scale(point)
                self._draw_marks()
                if len(mark.corners) == 4:
                    self.set_mode_keep("navigate")
                    self._suggest_reveal(mark)
            elif self.mode == "reveal":
                mark = self.session.mark(self.current_mark)
                self.session.add_reveal_point(mark, self.side.marks.side.currentData(), point)
                self._warn_scale(point)
                self._draw_marks()
                if len(mark.reveal_points) == 2:
                    self.set_mode_keep("navigate")
            elif self.mode == "base":
                if len(self.session.reference.ends) == 2:
                    self.session.clear_reference()
                self.session.add_reference_end(point)
                self._draw_base()
                if len(self.session.reference.ends) == 2:
                    self.set_mode_keep("navigate")
        except ValueError as error:
            self.status_label.setText(str(error))

    def _draw_plane_points(self) -> None:
        pts = [p.xy for p in self.session.plane_points]
        self.canvas.set_overlay("plane", pts, closed=len(pts) == 4, color="#fdd835")
        self.side.plane.corners_label.setText(f"Углов указано: {len(pts)} из 4")

    # --- снимок ---------------------------------------------------------------

    def _ask_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть снимок фасада", "", IMAGE_FILTER)
        if path:
            self.open_image(path)

    def _run(self, fn, on_done, message: str) -> None:
        if self._job is not None:
            self._job.cancel()
        self.status_label.setText(message)
        self._job = run_in_background(fn, on_done=on_done, on_error=self._frame_failed)

    def open_image(self, path) -> None:
        """Открыть снимок: фаза кадра (секунды) — в фоне, прежняя задача отменяется."""
        path = Path(path)
        if self._job is not None:
            self._job.cancel()
        self.session.open_image(path)
        self.canvas.scene().clear()
        self.canvas.frame_item = self.canvas.usable_item = None
        self.canvas.overlays = {}
        if self._raster_job is not None:
            self._raster_job.cancel()
        self.rect_view.scene().clear()
        self.rect_view.to_raster = self.rect_view.H = None
        self.rectified = None
        self.rect_label.setText("Выровненный вид появится после опорной базы")
        self.side.plane.set_frame_loaded(False)
        session = self.session
        self._run(lambda: run.frame_stage(session.image_path,
                                          profile_path=session.profile_path,
                                          with_color=True),
                  self._frame_ready,
                  f"{path.name}: обрабатывается — кадр, калибровка, плоскость…")

    def _frame_ready(self, fs, *, plane_override=None, roi=None) -> None:
        first = self.session.frame is None or self.session.frame.frame is not fs.frame
        self.session.accept_frame(fs, plane_override=plane_override, roi=roi)
        if first:
            self.canvas.set_frame(fs.frame.color)
        self.side.plane.set_frame_loaded(True)
        self._show_frame_state(fs)

    def _show_frame_state(self, fs) -> None:
        plane = fs.plane
        parts = [f"{fs.path.name}: {fs.image_size[0]}×{fs.image_size[1]}",
                 f"поворот EXIF {fs.frame.orientation}",
                 f"K: {fs.camera_record.calibration}"]
        if plane.needs_operator:
            reasons = "; ".join(plane.confidence.reasons) or "доверие ниже порога"
            parts.append(f"плоскость не восстановлена ({reasons}) — укажите её вручную")
            self.canvas.set_usable_mask(None)
            self.side.plane.info.setText(
                "Плоскость не восстановлена автоматически. Укажите четыре угла "
                "заведомого прямоугольника в плоскости стены и одно доопределение.")
            self.set_mode("plane")
        else:
            preview = run.theta_preview(fs)
            self.canvas.set_usable_mask(preview.usable_mask)
            pre = quality_gate.assess_preliminary(fs.sharpness, preview.usable_fraction)
            self.side.quality.show_preliminary(
                pre.verdict, pre.reasons,
                "плоскость задана оператором" if plane.method == "manual_four_point"
                else f"доверие к плоскости {plane.confidence.value:.2f}")
            half = preview.half_turn
            manual = plane.method == "manual_four_point"
            parts.append("плоскость задана оператором" if manual
                         else f"доверие к плоскости {plane.confidence.value:.2f}")
            parts.append(f"пригодно по углу {preview.usable_fraction:.0%} кадра")
            parts.append("поворот на полкруга: " + (
                "задан оператором" if half.resolved
                else "не разрешён, принят «снимок не перевёрнут»"))
            self.side.plane.info.setText(
                ("Плоскость задана оператором." if manual else
                 f"Плоскость оценена по точкам схода, доверие {plane.confidence.value:.2f}.")
                + " Переопределить её можно в любой момент.")
        self.status_label.setText("; ".join(parts))
        self._show_scale()

    def _frame_failed(self, message: str) -> None:
        self.session.error = message
        self.status_label.setText(f"Кадр не обработан: {message}")

    # --- плоскость (задача 10) ----------------------------------------------------

    def apply_manual_plane(self) -> None:
        """Ручная плоскость — мгновенно: снимок не перечитывается (`run.replace_plane`)."""
        fs = self.session.frame
        if fs is None:
            return
        try:
            override = self.session.manual_plane(**self.side.plane.constraint())
            new_fs = run.replace_plane(fs, plane_override=override)
        except ValueError as error:
            self.status_label.setText(f"Ручная плоскость не применена: {error}")
            return
        self._frame_ready(new_fs, plane_override=override)
        self.set_mode("navigate")
        self._draw_plane_points()

    def apply_roi(self) -> None:
        """Точки схода по области — секунды, в фоне."""
        fs = self.session.frame
        if fs is None:
            return
        try:
            roi = self.session.roi_polygon()
        except ValueError as error:
            self.status_label.setText(str(error))
            return
        self._run(lambda: run.replace_plane(fs, roi=roi),
                  lambda new_fs: self._frame_ready(new_fs, roi=roi),
                  "точки схода оцениваются по области…")
        self.set_mode("navigate")

    def reset_plane(self) -> None:
        fs = self.session.frame
        if fs is None:
            return
        self.canvas.clear_overlays("plane")
        self.canvas.clear_overlays("roi")
        self._run(lambda: run.replace_plane(fs), self._frame_ready,
                  "точки схода оцениваются по всему кадру…")

    def set_mode_keep(self, mode: str) -> None:
        """Сменить режим, НЕ сбрасывая сделанного в прежнем (в отличие от `set_mode`)."""
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        self.mode_label.setText(f"Режим: {MODES[mode]}")

    # --- опорная база (задача 11) --------------------------------------------------

    def _draw_base(self) -> None:
        ends = self.session.reference.ends
        self.canvas.set_overlay("base", [p.xy for p in ends], color="#43a047")
        lines = [f"Концов указано: {len(ends)} из 2"]
        for i, p in enumerate(ends, 1):
            note = (f" — масштаб {self._scale_name(p.view_scale)} грубее 1:1; при "
                    f"{self._scale_name(RECOMMENDED_MIN_SCALE)} было бы "
                    f"{sigma_image_px(RECOMMENDED_MIN_SCALE):.2f} px"
                    if p.view_scale < 1.0 else "")
            lines.append(f"конец {i}: σ клика {click_sigma(p):.2f} px{note}")
        self.side.base.ends_label.setText("\n".join(lines))

    @staticmethod
    def _scale_name(scale: float) -> str:
        return f"{scale:g}:1" if scale >= 1 else f"1:{1 / scale:.3g}"

    def apply_base(self) -> None:
        """Масштаб — миллисекунды: плоскость не переоценивается (фаза масштаба)."""
        v = self.side.base.values()
        self.session.set_span_mm(v["span_mm"] if v["span_mm"] > 0 else None)
        self.session.set_base_options(span_sigma_mm=v["span_sigma_mm"],
                                      scale_source=v["scale_source"],
                                      origin_is_facade_corner=v["origin_is_facade_corner"])
        ss = self.session.compute_scale(click_sigma)
        if ss is None:
            self.side.base.result.setText(f"Масштаб не вычислен: {self.session.error}")
            return
        origin = ("левый нижний угол фасада (охват полный)"
                  if v["origin_is_facade_corner"] else "точка оператора (охват частичный)")
        tol = "достигает" if ss.sigma_rel <= run.SIGMA_REL_TOLERANCE else "НЕ достигает"
        self.side.base.result.setText(
            f"σ масштаба {ss.sigma_rel:.3%} — {tol} допуска п. 6.3; "
            f"на базе {v['span_mm']:.0f} мм это {ss.sigma_rel * v['span_mm']:.1f} мм. "
            f"Начало отсчёта — {origin}. Более длинная база даёт меньшую σ.")
        self.scale_ready(ss)

    # --- разметка проёмов (задача 13) ---------------------------------------------

    def new_mark(self) -> None:
        attrs = self.side.marks.attributes()
        mark = self.session.new_mark(attrs["class_"])
        mark.mounting, mark.edge_type = attrs["mounting"], attrs["edge_type"]
        self.current_mark = mark.id
        self.set_mode_keep("opening")
        self._draw_marks()
        self.side.marks.warning.setText(
            f"{mark.id}: укажите четыре угла, порядок не важен. Рекомендуемое увеличение "
            f"у углов — не менее {RECOMMENDED_MIN_SCALE:g}:1.")

    def select_mark(self, mark_id: str) -> None:
        self.current_mark = mark_id
        self._draw_marks()

    def pick_reveal(self) -> None:
        if self.current_mark is None:
            self.side.marks.warning.setText("Сначала выберите или создайте проём.")
            return
        self.session.mark(self.current_mark).reveal_points.clear()
        self.set_mode_keep("reveal")
        self._draw_marks()

    def undo_point(self) -> None:
        if self.current_mark is not None:
            self.session.undo_last_point(self.session.mark(self.current_mark))
            self._draw_marks()

    def delete_mark(self) -> None:
        if self.current_mark is not None:
            self.session.delete_mark(self.current_mark)
            self.current_mark = self.session.marks[-1].id if self.session.marks else None
            self._draw_marks()

    def _suggest_reveal(self, mark) -> None:
        sides = self.session.suggested_reveal_sides(mark)
        label = self.side.marks.suggestion
        if sides is None:
            label.setText("Видимую грань откоса подскажет поза камеры — задайте опорную базу.")
            return
        vertical, horizontal = sides
        names = dict(self.side.marks.SIDES)
        label.setText(f"Камере видны грани: {names[vertical]} и {names[horizontal]}. "
                      "Ближние закрыты собственной стеной — их не размечают (п. 5.5).")
        box = self.side.marks.side
        box.setCurrentIndex(box.findData(vertical))

    def _warn_scale(self, point) -> None:
        """Цена масштаба клика — числом (σ в px и слагаемое бюджета в мм)."""
        label = self.side.marks.warning
        sigma = click_sigma(point)
        gsd = self.session.local_gsd(point)
        cost = (f", слагаемое бюджета {localisation_cost_mm(point.view_scale, gsd):.1f} мм "
                f"при разрешении {gsd:.1f} мм/px" if gsd is not None else "")
        if point.view_scale < RECOMMENDED_MIN_SCALE:
            best = sigma_image_px(RECOMMENDED_MIN_SCALE)
            label.setText(f"Масштаб {self._scale_name(point.view_scale)} грубее "
                          f"рекомендованного {RECOMMENDED_MIN_SCALE:g}:1: σ клика "
                          f"{sigma:.2f} px вместо {best:.2f}{cost}.")
        else:
            label.setText(f"σ клика {sigma:.2f} px{cost}.")

    def _draw_marks(self) -> None:
        self.canvas.clear_overlays("mark:")
        for m in self.session.marks:
            color = "#ff7043" if m.id == self.current_mark else "#8e24aa"
            self.canvas.set_overlay(f"mark:{m.id}", [p.xy for p in m.corners],
                                    closed=len(m.corners) == 4, color=color)
            if m.reveal_points:
                self.canvas.set_overlay(f"mark:{m.id}:reveal",
                                        [p.xy for p in m.reveal_points], color="#00acc1")
        self.side.marks.set_marks(self.session.marks, self.current_mark)
        self.rect_view.set_polygons({
            m.id: ([p.xy for p in m.corners], len(m.corners) == 4,
                   "#ff7043" if m.id == self.current_mark else "#8e24aa")
            for m in self.session.marks})

    def compute(self):
        """Модель фасада по текущей разметке — через фазу элементов, без переоценки."""
        model = self.session.compute_elements(click_sigma)
        if model is None:
            self.status_label.setText(f"Не посчитано: {self.session.error}")
            return None
        self.model_ready(model)
        return model

    def model_ready(self, model) -> None:
        """Точка расширения для задачи 15: модель посчитана."""
        self.status_label.setText(f"Посчитано элементов: {len(model.elements)}")

    def scale_ready(self, ss) -> None:
        """Масштаб посчитан: окончательный вердикт (задача 12), растр (задача 14)."""
        self.side.quality.show_final(ss.quality.verdict, ss.quality.reasons)
        self._start_raster(ss)

    # --- выровненный вид (задача 14) ------------------------------------------------

    def _start_raster(self, ss) -> None:
        """Пиксели растра — в фоне (до 4 с на наибольшем разрешении); геометрия та же,
        что уйдёт в выходной файл (`run.raster_geometry_for`)."""
        fs = self.session.frame
        if self._raster_job is not None:
            self._raster_job.cancel()
        self.rect_label.setText("Выровненный вид строится…")
        self._raster_job = run_in_background(
            lambda: run.raster_stage(fs, run.raster_geometry_for(fs, ss), color=True),
            on_done=self._raster_ready,
            on_error=lambda m: self.rect_label.setText(f"Выровненный вид не построен: {m}"))

    def _raster_ready(self, rectified) -> None:
        self.rectified = rectified
        self.rect_view.set_raster(rectified)
        self.rect_label.setText(
            f"Выровненный вид, {rectified.mm_per_px:.1f} мм/px; охват растра "
            f"{self.rect_view.coverage:.0%} (штриховка — вне снимка). Только для просмотра: "
            "клик центрирует кадр, измерение — в кадре (п. 6.5).")
        self._draw_marks()

    def center_frame_on(self, x: float, y: float) -> None:
        self.canvas.centerOn(x + 0.5, y + 0.5)

    def _show_scale(self, *_):
        self.scale_label.setText(self.canvas.scale_text())
