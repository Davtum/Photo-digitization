"""Панели окна оператора. План 3, задачи 10–15.

Панели только показывают сессию и собирают ввод; решения принимает сессия и ядро.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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


class BasePanel(QGroupBox):
    """Опорная база и начало отсчёта. План 3, задача 11; спецификация, п. 6.3 и п. 7.

    σ масштаба показывается СРАЗУ после ввода: фаза масштаба — миллисекунды, и
    оператор видит, что короткая база даёт худший масштаб, до разметки проёмов.
    """

    pickEnds = Signal()
    apply = Signal()

    SOURCES = (("operator_reference", "Длина измерена (рулетка, дальномер)"),
               ("assumed_floor_height", "Типовая высота этажа (допуск не достигается)"))

    def __init__(self, parent=None):
        super().__init__("Опорная база", parent)
        self.pick_ends = QPushButton("Указать два конца базы…")
        self.pick_ends.setToolTip("Две точки в плоскости стены по острой кромке, как можно "
                                  "дальше друг от друга: ошибка масштаба общая для всех "
                                  "проёмов.")
        self.ends_label = QLabel("Концов указано: 0 из 2")
        self.ends_label.setWordWrap(True)
        self.span_mm = _spin(0.0, 0.0, 1e6, 1, " мм")
        self.span_sigma_mm = _spin(0.0, 0.0, 1e4, 1, " мм")
        self.span_sigma_mm.setSpecialValueText("не задана")
        self.span_sigma_mm.setToolTip("Погрешность самой измеренной длины: рулетка, "
                                      "дальномер. 0 — не задана.")
        self.source = QComboBox()
        for key, text in self.SOURCES:
            self.source.addItem(text, key)
        self.corner_origin = QCheckBox("Первый конец — левый нижний угол фасада")
        self.corner_origin.setToolTip("Спецификация, п. 7: тогда начало координат — угол "
                                      "фасада и охват полный; иначе — точка оператора.")
        self.apply_button = QPushButton("Применить масштаб")
        self.result = QLabel("Масштаб не задан")
        self.result.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Длина базы", self.span_mm)
        form.addRow("Погрешность длины", self.span_sigma_mm)
        form.addRow("Происхождение", self.source)
        layout = QVBoxLayout(self)
        layout.addWidget(self.pick_ends)
        layout.addWidget(self.ends_label)
        layout.addLayout(form)
        layout.addWidget(self.corner_origin)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.result)
        self.pick_ends.clicked.connect(self.pickEnds)
        self.apply_button.clicked.connect(self.apply)

    def values(self) -> dict:
        sigma = float(self.span_sigma_mm.value())
        return {"span_mm": float(self.span_mm.value()),
                "span_sigma_mm": sigma if sigma > 0 else None,
                "scale_source": self.source.currentData(),
                "origin_is_facade_corner": self.corner_origin.isChecked()}

    def set_values(self, *, span_mm=None, span_sigma_mm=None, scale_source=None,
                   origin_is_facade_corner=None) -> None:
        if span_mm is not None:
            self.span_mm.setValue(span_mm)
        if span_sigma_mm is not None:
            self.span_sigma_mm.setValue(span_sigma_mm)
        if scale_source is not None:
            self.source.setCurrentIndex(self.source.findData(scale_source))
        if origin_is_facade_corner is not None:
            self.corner_origin.setChecked(origin_is_facade_corner)


class QualityPanel(QGroupBox):
    """Шлюз качества в две ступени. План 3, задача 12; спецификация, п. 4.1.

    Причины показываются ТЕМИ ЖЕ строками, что попадают в `quality.reasons`
    выходного файла: вторая формулировка одной причины разошлась бы с первой.
    """

    def __init__(self, parent=None):
        super().__init__("Качество снимка", parent)
        self.stage = QLabel("—")
        self.verdict = QLabel("—")
        self.reasons = QLabel("")
        self.note = QLabel("")
        for label in (self.stage, self.verdict, self.reasons, self.note):
            label.setWordWrap(True)
        layout = QVBoxLayout(self)
        for label in (self.stage, self.verdict, self.reasons, self.note):
            layout.addWidget(label)
        self.reason_list: list[str] = []

    def _show(self, verdict: str, reasons: list[str]) -> None:
        self.verdict.setText(f"Вердикт: {verdict}")
        self.reason_list = list(reasons)
        self.reasons.setText("\n".join(f"— {r}" for r in reasons) or "причин нет")

    def show_preliminary(self, verdict: str, reasons: list[str], confidence: str) -> None:
        self.stage.setText("Предварительный: до опорной базы")
        self._show(verdict, reasons)
        self.note.setText(f"{confidence}. Вердикт по разрешению будет после ввода "
                          "опорной базы: разрешение считается от её длины.")

    def show_final(self, verdict: str, reasons: list[str]) -> None:
        self.stage.setText("Окончательный: зависит от введённой длины опорной базы")
        self._show(verdict, reasons)
        if verdict == "reject":
            self.note.setText("Кадр отбракован по разрешению. Разметка разрешена, но "
                              "признак соответствия допуску у элементов выпущен не будет. "
                              "Проверьте длину базы: опечатка в ней меняет вердикт.")
        else:
            self.note.setText("Вердикт зависит от длины опорной базы: опечатка в ней "
                              "меняет его и признак допуска у всех элементов.")


class SidePanel(QWidget):
    """Правая колонка окна: панели сверху вниз в порядке работы оператора."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plane = PlanePanel(self)
        self.base = BasePanel(self)
        self.quality = QualityPanel(self)
        self.layout_ = QVBoxLayout(self)
        self.layout_.addWidget(self.plane)
        self.layout_.addWidget(self.quality)
        self.layout_.addWidget(self.base)
        self.layout_.addStretch(1)
