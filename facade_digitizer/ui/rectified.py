"""Выровненный вид. План 3, задача 14.

Главный видимый результат ядра — трапеция кадра становится прямоугольником фасада.
Вид служит ТОЛЬКО просмотру и навигации (п. 6.5): измерение выполняется в кадре, а
клик по выровненному растру центрирует кадр на соответствующей точке и точкой
разметки не становится. Разрешить клики здесь значило бы переводить масштаб
просмотра в σ через якобиан H⁻¹, меняющийся по кадру в 2.5–13 раз (п. 2.2), —
план этого не делает.

**Перевод кадр → растр.** Растр задан в пикселях `Rectified.H` (с масштабом и
сдвигом растра), а не в единицах `H_units`. `QTransform` строится из
ТРАНСПОНИРОВАННОЙ H: Qt умножает на вектор-строку. Построчная H даёт промах в сотни
пикселей, а проверка «туда и обратно» при этом проходит (рецензия плана 3) — поэтому
тест сверяет перевод с опорными значениями выходного файла.
"""
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)

from facade_digitizer.ui.canvas import to_qimage


def frame_to_raster_transform(H) -> QTransform:
    """Кадр (OpenCV) → растр (OpenCV) как `QTransform`. Транспонирование обязательно."""
    m = np.asarray(H, dtype=float)
    return QTransform(m[0, 0], m[1, 0], m[2, 0],
                      m[0, 1], m[1, 1], m[2, 1],
                      m[0, 2], m[1, 2], m[2, 2])


class RectifiedView(QGraphicsView):
    """Выровненный растр с наложениями и навигацией обратно в кадр."""

    #: Клик по растру: точка КАДРА (соглашение OpenCV), на которую центрировать кадр.
    frameTarget = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.H = None
        self.to_raster: QTransform | None = None
        self.raster_item = None
        self.invalid_item = None
        self.coverage = None
        self._overlays = []
        self._press = None

    def set_raster(self, rectified) -> None:
        """Растр `pipeline.rectify.Rectified` (цветной либо серый) и его H."""
        self.scene().clear()
        self._overlays = []
        self.H = np.asarray(rectified.H, dtype=float)
        self.to_raster = frame_to_raster_transform(self.H)
        pix = QPixmap.fromImage(to_qimage(rectified.image))
        self.raster_item = self.scene().addPixmap(pix)
        mask = np.asarray(rectified.valid_mask, dtype=bool)
        self.coverage = float(mask.mean())
        # Непокрытая часть растра — явная штриховка, а не серый фон: серое поле
        # читается как измеренная часть фасада, которой там нет.
        rgba = np.zeros(mask.shape + (4,), np.uint8)
        stripes = ((np.add.outer(np.arange(mask.shape[0]), np.arange(mask.shape[1])) // 6)
                   % 2).astype(bool)
        rgba[~mask & stripes] = (200, 60, 60, 150)
        rgba[~mask & ~stripes] = (255, 255, 255, 120)
        h, w = mask.shape
        qimg = QImage(np.ascontiguousarray(rgba).data, w, h, w * 4,
                      QImage.Format_RGBA8888).copy()
        self.invalid_item = self.scene().addPixmap(QPixmap.fromImage(qimg))
        self.invalid_item.setZValue(1)
        self.scene().setSceneRect(0, 0, pix.width(), pix.height())
        self.fitInView(self.raster_item, Qt.KeepAspectRatio)

    def frame_to_scene(self, x: float, y: float) -> QPointF:
        """Точка кадра → точка сцены растра (центр пикселя растра = +0.5)."""
        p = self.to_raster.map(QPointF(float(x), float(y)))
        return QPointF(p.x() + 0.5, p.y() + 0.5)

    def scene_to_frame(self, scene_pt: QPointF) -> tuple[float, float] | None:
        if self.to_raster is None:
            return None
        inverted, ok = self.to_raster.inverted()
        if not ok:
            return None
        p = inverted.map(QPointF(scene_pt.x() - 0.5, scene_pt.y() - 0.5))
        return (p.x(), p.y())

    def set_polygons(self, polygons: dict) -> None:
        """Контуры в координатах КАДРА: {ключ: (точки, замкнут, цвет)}."""
        for item in self._overlays:
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self._overlays = []
        if self.to_raster is None:
            return
        for pts, closed, color in polygons.values():
            if not pts:
                continue
            pen = QPen(QColor(color))
            pen.setCosmetic(True)
            pen.setWidthF(2.0)
            scene_pts = [self.frame_to_scene(x, y) for x, y in pts]
            if len(scene_pts) >= 2:
                path = QPainterPath(scene_pts[0])
                for p in scene_pts[1:]:
                    path.lineTo(p)
                if closed:
                    path.closeSubpath()
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setZValue(2)
                self.scene().addItem(item)
                self._overlays.append(item)
            for p in scene_pts:
                dot = QGraphicsEllipseItem(-3, -3, 6, 6)
                dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                dot.setPos(p)
                dot.setPen(pen)
                dot.setBrush(QBrush(QColor(color)))
                dot.setZValue(3)
                self.scene().addItem(dot)
                self._overlays.append(dot)

    def mousePressEvent(self, event):
        self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        pos = event.position().toPoint()
        moved = self._press is not None and (pos - self._press).manhattanLength() > 3
        if event.button() == Qt.LeftButton and not moved:
            target = self.scene_to_frame(self.mapToScene(pos))
            if target is not None:
                self.frameTarget.emit(*target)
        self._press = None
