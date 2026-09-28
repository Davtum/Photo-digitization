"""Главное окно приложения оператора. План 3, задачи 1 и 9.

Окно — тонкий слой над `ui.session.OperatorSession`: оно показывает сессию и
посылает ей события. Долгая фаза кадра идёт в фоне (`ui.worker`), окно при этом не
замирает; остальные инструменты добавляются задачами 10–18.
"""
from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow

from facade_digitizer.pipeline import run
from facade_digitizer.ui.canvas import FrameCanvas
from facade_digitizer.ui.session import OperatorSession
from facade_digitizer.ui.worker import run_in_background

TITLE = "Оцифровка фасада"
IMAGE_FILTER = "Снимки (*.jpg *.jpeg *.png *.tif *.tiff);;Все файлы (*)"


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.session = OperatorSession()
        self.canvas = FrameCanvas(self)
        self.setCentralWidget(self.canvas)
        # Прежние имена каркаса задачи 1.
        self.view, self.scene = self.canvas, self.canvas.scene()

        self.status_label = QLabel("Снимок не открыт")
        self.scale_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.scale_label)
        self.canvas.scaleChanged.connect(self._show_scale)
        self._job = None
        self._build_actions()

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

    # --- снимок ---------------------------------------------------------------

    def _ask_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть снимок фасада", "", IMAGE_FILTER)
        if path:
            self.open_image(path)

    def open_image(self, path) -> None:
        """Открыть снимок: фаза кадра (секунды) — в фоне, прежняя задача отменяется."""
        path = Path(path)
        if self._job is not None:
            self._job.cancel()
        self.session.open_image(path)
        self.canvas.scene().clear()
        self.canvas.frame_item = self.canvas.usable_item = None
        self.status_label.setText(f"{path.name}: обрабатывается — кадр, калибровка, "
                                  "плоскость…")
        session = self.session
        self._job = run_in_background(
            lambda: run.frame_stage(session.image_path, profile_path=session.profile_path,
                                    with_color=True, plane_override=session.plane_override,
                                    roi=session.roi),
            on_done=self._frame_ready, on_error=self._frame_failed)

    def _frame_ready(self, fs) -> None:
        self.session.frame, self.session.error = fs, None
        self.canvas.set_frame(fs.frame.color)
        name = fs.path.name
        plane = fs.plane
        parts = [f"{name}: {fs.image_size[0]}×{fs.image_size[1]}",
                 f"поворот EXIF {fs.frame.orientation}",
                 f"K: {fs.camera_record.calibration}"]
        if plane.needs_operator:
            reasons = "; ".join(plane.confidence.reasons) or "доверие ниже порога"
            parts.append(f"плоскость не восстановлена ({reasons}) — укажите её вручную")
        else:
            preview = run.theta_preview(fs)
            self.canvas.set_usable_mask(preview.usable_mask)
            half = preview.half_turn
            parts.append(f"доверие к плоскости {plane.confidence.value:.2f}")
            parts.append(f"пригодно по углу {preview.usable_fraction:.0%} кадра")
            parts.append("поворот на полкруга: " + ("задан оператором" if half.resolved
                                                    else "не разрешён, принят «снимок не "
                                                         "перевёрнут»"))
        self.status_label.setText("; ".join(parts))
        self._show_scale()

    def _frame_failed(self, message: str) -> None:
        self.session.error = message
        self.status_label.setText(f"Кадр не обработан: {message}")

    def _show_scale(self, *_):
        self.scale_label.setText(self.canvas.scale_text())
