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
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

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
    #: Перетаскивание точки разметки: ключ точки, x, y кадра, масштаб (задача 16).
    pointDragged = Signal(str, float, float, float)
    pointDropped = Signal(str, float, float, float)

    #: Радиус захвата точки разметки, экранных пикселей.
    GRAB_RADIUS_PX = 8.0

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
        self.overlays: dict[str, list] = {}
        #: Точки, которые можно перетаскивать: функция → [(ключ, x, y)] кадра.
        self.editable_points = list
        self._dragging: str | None = None

    # --- кадр -----------------------------------------------------------------

    def set_frame(self, image: np.ndarray) -> None:
        self.scene().clear()
        self.usable_item = None
        self.overlays = {}
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

    # --- наложения ------------------------------------------------------------

    def set_overlay(self, key: str, points, *, closed: bool = False,
                    color: str = "#e53935", markers: bool = True) -> None:
        """Точки и ломаная в координатах КАДРА (соглашение OpenCV) поверх снимка.

        Толщина линий и размер меток — в экранных пикселях (косметическое перо и
        метки, не масштабируемые с видом): при 1:8 и при 8:1 разметку видно
        одинаково, и она не закрывает кромку, по которой оператор целится.
        """
        self.clear_overlay(key)
        pts = [(float(x) + 0.5, float(y) + 0.5) for x, y in points]
        items = []
        pen = QPen(QColor(color))
        pen.setCosmetic(True)
        pen.setWidthF(2.0)
        if len(pts) >= 2:
            path = QPainterPath(QPointF(*pts[0]))
            for pt in pts[1:]:
                path.lineTo(QPointF(*pt))
            if closed and len(pts) >= 3:
                path.closeSubpath()
            line = QGraphicsPathItem(path)
            line.setPen(pen)
            line.setZValue(2)
            self.scene().addItem(line)
            items.append(line)
        if markers:
            for x, y in pts:
                dot = QGraphicsEllipseItem(-4, -4, 8, 8)
                dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                dot.setPos(x, y)
                dot.setPen(pen)
                dot.setBrush(QBrush(QColor(color)))
                dot.setZValue(3)
                self.scene().addItem(dot)
                items.append(dot)
        self.overlays[key] = items

    def clear_overlay(self, key: str) -> None:
        for item in self.overlays.pop(key, []):
            if item.scene() is self.scene():
                self.scene().removeItem(item)

    def clear_overlays(self, prefix: str = "") -> None:
        for key in [k for k in self.overlays if k.startswith(prefix)]:
            self.clear_overlay(key)

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

    def _grab_key(self, viewport_pos) -> str | None:
        """Ключ ближайшей точки разметки в радиусе захвата (в ЭКРАННЫХ пикселях)."""
        best, best_d = None, self.GRAB_RADIUS_PX
        for key, x, y in self.editable_points():
            p = self.mapFromScene(QPointF(x + 0.5, y + 0.5))
            d = ((p.x() - viewport_pos.x()) ** 2 + (p.y() - viewport_pos.y()) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = key, d
        return best

    def mousePressEvent(self, event) -> None:
        self._press_pos = event.position().toPoint()
        if event.button() == Qt.LeftButton:
            self._dragging = self._grab_key(self._press_pos)
            if self._dragging is not None:
                self.setDragMode(QGraphicsView.NoDrag)     # не панорама, а правка
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None:
            point = self.frame_point_at(event.position().toPoint())
            if point is not None:
                self.pointDragged.emit(self._dragging, point[0], point[1], self.view_scale())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging is not None:
            key, self._dragging = self._dragging, None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            point = self.frame_point_at(event.position().toPoint())
            if point is not None:
                self.pointDropped.emit(key, point[0], point[1], self.view_scale())
            self._press_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
        pos = event.position().toPoint()
        moved = (self._press_pos is not None
                 and (pos - self._press_pos).manhattanLength() > 3)
        if event.button() == Qt.LeftButton and not moved:
            point = self.frame_point_at(pos)
            if point is not None:
                self.pointClicked.emit(point[0], point[1], self.view_scale())
        self._press_pos = None
