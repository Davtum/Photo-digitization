"""Главное окно приложения оператора. План 3, задачи 1, 9, 10.

Окно — тонкий слой над `ui.session.OperatorSession`: оно показывает сессию и
посылает ей события. Клик по холсту уходит в сессию по текущему РЕЖИМУ инструмента
(`handle_click`); тот же метод вызывают тесты, поэтому путь от клика до миллиметров
проверяется без мыши. Долгие фазы идут в фоне (`ui.worker`), окно при этом не замирает.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QDockWidget, QFileDialog, QLabel, QMainWindow

from facade_digitizer.pipeline import run
from facade_digitizer.ui.canvas import FrameCanvas
from facade_digitizer.ui.panels import SidePanel
from facade_digitizer.ui.session import ClickedPoint, OperatorSession
from facade_digitizer.ui.worker import run_in_background
from facade_digitizer.ui.zoom import RECOMMENDED_MIN_SCALE, click_sigma, sigma_image_px

TITLE = "Оцифровка фасада"
IMAGE_FILTER = "Снимки (*.jpg *.jpeg *.png *.tif *.tiff);;Все файлы (*)"

#: Режимы инструмента: что значит клик по холсту.
MODES = {
    "navigate": "Навигация",
    "plane": "Углы ручной плоскости",
    "roi": "Область оценки точек схода",
    "base": "Концы опорной базы",
}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.session = OperatorSession()
        self.canvas = FrameCanvas(self)
        self.setCentralWidget(self.canvas)
        # Прежние имена каркаса задачи 1.
        self.view, self.scene = self.canvas, self.canvas.scene()

        self.side = SidePanel(self)
        dock = QDockWidget("Оператор", self)
        dock.setWidget(self.side)
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
        try:
            if self.mode == "plane":
                self.session.add_plane_point(point)
                self._draw_plane_points()
            elif self.mode == "roi":
                self.session.add_roi_point(point)
                self.canvas.set_overlay("roi", [p.xy for p in self.session.roi_points],
                                        closed=True, color="#1e88e5")
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

    def scale_ready(self, ss) -> None:
        """Точка расширения для задач 12–15: масштаб посчитан."""

    def _show_scale(self, *_):
        self.scale_label.setText(self.canvas.scale_text())
