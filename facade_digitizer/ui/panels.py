"""Панели окна оператора. План 3, задачи 10–15.

Панели только показывают сессию и собирают ввод; решения принимает сессия и ядро.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


def _spin(value: float, lo: float, hi: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


class PlanePanel(QGroupBox):
    """Плоскость фасада: доверие, ручное задание (п. 4.2), область оценки, сброс.

    Ручное задание доступно ВСЕГДА, а не только при отказе автоматики: доверие к
    плоскости «предсказателем точности не является» (п. 4.2), и уверенно, но неверно
    оценённую плоскость (типично — снимок угла здания) оператор обязан мочь
    переопределить.
    """

    pickCorners = Signal()        # перейти к указанию четырёх углов
    applyManual = Signal()
    pickRoi = Signal()
    applyRoi = Signal()
    resetAuto = Signal()

    CONSTRAINTS = ("aspect", "sizes", "calibrated")

    def __init__(self, parent=None):
        super().__init__("Плоскость фасада", parent)
        self.info = QLabel("Снимок не открыт")
        self.info.setWordWrap(True)

        self.pick_corners = QPushButton("Указать четыре угла прямоугольника…")
        self.pick_corners.setToolTip(
            "Углы заведомого прямоугольника в плоскости стены: проём, панель, край "
            "фасада. Порядок кликов не важен.")
        self.corners_label = QLabel("Углов указано: 0 из 4")

        self.group = QButtonGroup(self)
        self.r_aspect = QRadioButton("Отношение сторон (ширина / высота)")
        self.r_sizes = QRadioButton("Два размера, мм")
        self.r_calibrated = QRadioButton("Камера откалибрована, стороны ортогональны")
        for i, r in enumerate((self.r_aspect, self.r_sizes, self.r_calibrated)):
            self.group.addButton(r, i)
        self.r_sizes.setChecked(True)
        self.aspect = _spin(1.0, 0.01, 100.0, 4)
        self.width_mm = _spin(1000.0, 1.0, 1e6, 1, " мм")
        self.height_mm = _spin(1000.0, 1.0, 1e6, 1, " мм")
        sizes = QHBoxLayout()
        sizes.addWidget(self.width_mm)
        sizes.addWidget(QLabel("×"))
        sizes.addWidget(self.height_mm)

        form = QFormLayout()
        form.addRow(self.r_aspect, self.aspect)
        form.addRow(self.r_sizes, sizes)
        form.addRow(self.r_calibrated)

        self.apply_manual = QPushButton("Применить ручную плоскость")
        self.pick_roi = QPushButton("Обвести область оценки…")
        self.pick_roi.setToolTip("Для снимка угла здания: точки схода оцениваются только "
                                 "по отрезкам внутри обведённой области.")
        self.apply_roi = QPushButton("Оценить по области")
        self.reset = QPushButton("Автоматически по всему кадру")

        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.pick_corners)
        layout.addWidget(self.corners_label)
        layout.addLayout(form)
        layout.addWidget(self.apply_manual)
        row = QHBoxLayout()
        row.addWidget(self.pick_roi)
        row.addWidget(self.apply_roi)
        layout.addLayout(row)
        layout.addWidget(self.reset)

        self.pick_corners.clicked.connect(self.pickCorners)
        self.apply_manual.clicked.connect(self.applyManual)
        self.pick_roi.clicked.connect(self.pickRoi)
        self.apply_roi.clicked.connect(self.applyRoi)
        self.reset.clicked.connect(self.resetAuto)
        self.set_frame_loaded(False)

    def set_frame_loaded(self, loaded: bool) -> None:
        for w in (self.pick_corners, self.apply_manual, self.pick_roi, self.apply_roi,
                  self.reset):
            w.setEnabled(loaded)

    def constraint(self) -> dict:
        """Выбранное доопределение — ровно одно, в аргументах `ManualPlane`."""
        which = self.CONSTRAINTS[self.group.checkedId()]
        if which == "aspect":
            return {"aspect_ratio": float(self.aspect.value())}
        if which == "sizes":
            return {"size_mm": (float(self.width_mm.value()), float(self.height_mm.value()))}
        return {"assume_calibrated": True}

    def set_constraint(self, which: str, **values) -> None:
        {"aspect": self.r_aspect, "sizes": self.r_sizes,
         "calibrated": self.r_calibrated}[which].setChecked(True)
        if "aspect_ratio" in values:
            self.aspect.setValue(values["aspect_ratio"])
        if "size_mm" in values:
            self.width_mm.setValue(values["size_mm"][0])
            self.height_mm.setValue(values["size_mm"][1])


class SidePanel(QWidget):
    """Правая колонка окна: панели сверху вниз в порядке работы оператора."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plane = PlanePanel(self)
        self.layout_ = QVBoxLayout(self)
        self.layout_.addWidget(self.plane)
        self.layout_.addStretch(1)
