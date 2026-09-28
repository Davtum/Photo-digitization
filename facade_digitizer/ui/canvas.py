"""Просмотрщик кадра. План 3, задача 9.

Показывает `Frame.color` — тот же кадр, повёрнутый по EXIF и исправленный от
дисторсии, по которому считает конвейер (задача 3), — а не файл: иначе клики
ложились бы не туда без единой ошибки.

**Соглашение о пикселях.** У Qt пиксель `i` занимает `[i, i+1)` сцены; у конвейера
(OpenCV) центр пикселя `i` лежит в целой координате `i`, и допустимая точка — в
`[0, w−1]`. Перевод сцена → кадр — `x − 0.5`: без него все точки смещены на
полпикселя. Клик в любую часть пикселя снимка даёт точку этого пикселя — края
прижимаются к `[0, w−1]`, потому что правая половина последнего пикселя тоже
снимок; клик вне снимка точкой не становится.

**Масштаб — число, а не только ощущение.** От него зависит σ клика (`ui.zoom`), и
оператор должен его видеть: «1:4», «1:1», «2:1».
"""
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

#: Шаг колеса мыши: один щелчок меняет масштаб в 1.25 раза.
WHEEL_STEP = 1.25
MIN_SCALE, MAX_SCALE = 1.0 / 64.0, 32.0


def to_qimage(image: np.ndarray) -> QImage:
    """BGR (OpenCV) или серый массив → QImage с собственной копией данных."""
    arr = np.ascontiguousarray(image)
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8).copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = np.ascontiguousarray(arr[:, :, ::-1])
        h, w, _ = rgb.shape
        return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
    raise ValueError(f"кадр: ожидался серый или трёхканальный массив, получено {arr.shape}")


class FrameCanvas(QGraphicsView):
    """Кадр с масштабом, панорамой и кликами в координатах конвейера."""

    #: x, y — пиксели кадра (соглашение OpenCV); последний — масштаб просмотра.
    pointClicked = Signal(float, float, float)
    #: Масштаб изменился: новое значение.
    scaleChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.frame_item: QGraphicsPixmapItem | None = None
        self.usable_item: QGraphicsPixmapItem | None = None
        self.frame_size: tuple[int, int] | None = None
        self._press_pos = None

    # --- кадр -----------------------------------------------------------------

    def set_frame(self, image: np.ndarray) -> None:
        self.scene().clear()
        self.usable_item = None
        pixmap = QPixmap.fromImage(to_qimage(image))
        self.frame_item = self.scene().addPixmap(pixmap)
        self.frame_item.setPos(0, 0)
        self.frame_size = (pixmap.width(), pixmap.height())
        self.scene().setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fit_to_window()

    def set_usable_mask(self, mask: np.ndarray | None) -> None:
        """Пригодная зона θ ≤ 30° (п. 2.4) поверх кадра; `None` — скрыть.

        Непригодная часть затемняется, пригодная остаётся как есть: оператор видит,
        где проём мерить можно, до того как начал разметку.
        """
        if self.usable_item is not None:
            self.scene().removeItem(self.usable_item)
            self.usable_item = None
        if mask is None or self.frame_size is None:
            return
        m = np.asarray(mask, dtype=bool)
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[~m] = (0, 0, 0, 110)
        h, w = m.shape
        qimg = QImage(np.ascontiguousarray(rgba).data, w, h, w * 4,
                      QImage.Format_RGBA8888).copy()
        # Сглаживание при растяжении: маска посчитана на сетке узлов (64×64), и без
        # него граница зоны на экране — ступеньки, которых в поле углов нет.
        pix = QPixmap.fromImage(qimg).scaled(*self.frame_size, Qt.IgnoreAspectRatio,
                                             Qt.SmoothTransformation)
        self.usable_item = self.scene().addPixmap(pix)
        self.usable_item.setZValue(1)

    # --- масштаб --------------------------------------------------------------

    def view_scale(self) -> float:
        """Экранных пикселей на пиксель кадра — ровно то, что стоит в преобразовании."""
        return float(self.transform().m11())

    def set_view_scale(self, scale: float) -> None:
        scale = float(min(max(scale, MIN_SCALE), MAX_SCALE))
        self.setTransform(QTransform.fromScale(scale, scale))
        self.scaleChanged.emit(scale)

    def one_to_one(self) -> None:
        self.resetTransform()
        self.scaleChanged.emit(1.0)

    def fit_to_window(self) -> None:
        if self.frame_item is not None:
            self.fitInView(self.frame_item, Qt.KeepAspectRatio)
            self.scaleChanged.emit(self.view_scale())

    def scale_text(self) -> str:
        s = self.view_scale()
        if s >= 1.0:
            return f"{s:g}:1" if abs(s - round(s, 1)) < 1e-9 else f"{s:.2f}:1"
        inv = 1.0 / s
        return f"1:{inv:g}" if abs(inv - round(inv, 1)) < 1e-9 else f"1:{inv:.2f}"

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            factor = WHEEL_STEP ** steps
            target = min(max(self.view_scale() * factor, MIN_SCALE), MAX_SCALE)
            factor = target / self.view_scale()
            self.scale(factor, factor)            # якорь — под курсором
            self.scaleChanged.emit(self.view_scale())
        event.accept()

    # --- клики ----------------------------------------------------------------

    def scene_to_frame(self, scene_pt: QPointF) -> tuple[float, float] | None:
        """Точка сцены Qt → точка кадра конвейера; `None` — вне снимка."""
        if self.frame_size is None:
            return None
        w, h = self.frame_size
        sx, sy = scene_pt.x(), scene_pt.y()
        if not (0.0 <= sx < w and 0.0 <= sy < h):
            return None
        return (min(max(sx - 0.5, 0.0), w - 1.0), min(max(sy - 0.5, 0.0), h - 1.0))

    def frame_point_at(self, viewport_pos) -> tuple[float, float] | None:
        return self.scene_to_frame(self.mapToScene(viewport_pos))

    def mousePressEvent(self, event) -> None:
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        pos = event.position().toPoint()
        moved = (self._press_pos is not None
                 and (pos - self._press_pos).manhattanLength() > 3)
        if event.button() == Qt.LeftButton and not moved:
            point = self.frame_point_at(pos)
            if point is not None:
                self.pointClicked.emit(point[0], point[1], self.view_scale())
        self._press_pos = None
