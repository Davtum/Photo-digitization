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
    QListWidget,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
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


class MarksPanel(QGroupBox):
    """Разметка проёмов и граней откоса. План 3, задача 13.

    Класс, способ установки и тип кромки выбираются из значений схемы, а не
    вводятся текстом: `mounting` — обязательное поле `ElementMark`, и от него вместе
    с типом кромки зависит, вправе ли элемент нести признак допуска (п. 5.3, 5.6).
    """

    newMark = Signal()
    pickReveal = Signal()
    undo = Signal()
    delete = Signal()
    selected = Signal(str)
    compute = Signal()

    CLASSES = (("window", "окно"), ("door", "дверь"))
    MOUNTINGS = (("embedded", "заглублён в проём"), ("flush", "заподлицо"),
                 ("protruding", "выступает"))
    EDGES = (("sharp_wall_edge", "острая кромка стены"), ("surround", "обрамление"),
             ("cladding_edge", "кромка облицовки"), ("unknown", "не известно"))
    SIDES = (("left", "левая"), ("right", "правая"), ("top", "верхняя"),
             ("bottom", "нижняя"))

    def __init__(self, parent=None):
        super().__init__("Проёмы", parent)
        self.class_ = QComboBox()
        self.mounting = QComboBox()
        self.edge = QComboBox()
        for box, items in ((self.class_, self.CLASSES), (self.mounting, self.MOUNTINGS),
                           (self.edge, self.EDGES)):
            for key, text in items:
                box.addItem(text, key)
        self.new_mark = QPushButton("Новый проём: указать четыре угла…")
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.side = QComboBox()
        for key, text in self.SIDES:
            self.side.addItem(text, key)
        self.pick_reveal = QPushButton("Указать внутреннюю кромку откоса…")
        self.suggestion = QLabel("")
        self.warning = QLabel("")
        for label in (self.suggestion, self.warning):
            label.setWordWrap(True)
        self.undo_button = QPushButton("Отменить точку")
        self.delete_button = QPushButton("Удалить проём")
        self.compute_button = QPushButton("Посчитать")

        form = QFormLayout()
        form.addRow("Класс", self.class_)
        form.addRow("Установка", self.mounting)
        form.addRow("Кромка", self.edge)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.new_mark)
        layout.addWidget(self.list)
        row = QHBoxLayout()
        row.addWidget(QLabel("Грань откоса"))
        row.addWidget(self.side)
        layout.addLayout(row)
        layout.addWidget(self.pick_reveal)
        layout.addWidget(self.suggestion)
        layout.addWidget(self.warning)
        row2 = QHBoxLayout()
        row2.addWidget(self.undo_button)
        row2.addWidget(self.delete_button)
        layout.addLayout(row2)
        layout.addWidget(self.compute_button)

        self.new_mark.clicked.connect(self.newMark)
        self.pick_reveal.clicked.connect(self.pickReveal)
        self.undo_button.clicked.connect(self.undo)
        self.delete_button.clicked.connect(self.delete)
        self.compute_button.clicked.connect(self.compute)
        self.list.currentTextChanged.connect(
            lambda text: self.selected.emit(text.split(" ")[0]) if text else None)

    def attributes(self) -> dict:
        return {"class_": self.class_.currentData(), "mounting": self.mounting.currentData(),
                "edge_type": self.edge.currentData()}

    def set_marks(self, marks, current: str | None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for m in marks:
            reveal = f", откос {m.reveal_side} {len(m.reveal_points)}/2" if m.reveal_side else ""
            self.list.addItem(f"{m.id} ({len(m.corners)}/4{reveal})")
        ids = [m.id for m in marks]
        if current in ids:
            self.list.setCurrentRow(ids.index(current))
        self.list.blockSignals(False)


#: Почему у элемента нет σ положения: положение сводится с общей ошибкой масштаба
#: в одном месте (`assemble`, п. 4.4), и до задачи 21 плана 3 оно не считается.
POSITION_SIGMA_PENDING = "σ не рассчитывается до сведения погрешностей (задача 21)"

#: Происхождение глубины по схеме → слова для оператора.
RECESS_ORIGIN_TEXT = {
    "unavailable": ("не измерена: грань откоса не видна камере, размечена не та сторона, "
                    "угол визирования на грань ниже порога либо кромка вне плоскости стены"),
    "assumed_class_default": "принята типовой для класса — не измерение",
    "operator": "введена оператором — не измерение",
}

CALIBRATION_TEXT = {
    "target": "калибровка по мишени — условие п. 2.2 выполнено",
    "exif": "K из EXIF — условие п. 2.2 «камера откалибрована» не выполнено",
    "database": "K из таблицы моделей — условие п. 2.2 «камера откалибрована» не выполнено",
}


def tolerance_text(element, verdict: str) -> str:
    """Признак соответствия допуску — словами, и при `null` — С ПРИЧИНОЙ.

    `null` значит «утверждать нечем», а пустая ячейка выдала бы отсутствие
    утверждения за его отсутствие в интерфейсе, а не в измерении.
    """
    if element.meets_tolerance is True:
        return "соответствует допуску п. 2.2"
    if element.meets_tolerance is False:
        return "не соответствует: σ габарита больше допуска 10 мм"
    if element.edge_reference != "wall_plane":
        return ("не выпущено: кромка вне плоскости стены или не известна "
                "(п. 5.3, 5.6), вынос не измерен")
    if verdict == "reject":
        return "не выпущено: кадр отбракован по разрешению (п. 2.2)"
    return "не выпущено"


class ResultPanel(QGroupBox):
    """Миллиметры с погрешностью. План 3, задача 15.

    Правило панели: величины без σ не бывает. Где σ ещё не считается (положение —
    до задачи 21), ячейка называет причину, а не пустует.
    """

    COLUMNS = ("id", "ширина, мм", "высота, мм", "положение X / Y, мм", "заглубление, мм",
               "θ", "допуск")

    def __init__(self, parent=None):
        super().__init__("Результат", parent)
        self.header = QLabel("Не посчитано")
        self.header.setWordWrap(True)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setMinimumHeight(140)
        self.table.verticalHeader().setVisible(False)
        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.table)

    def show_model(self, model) -> None:
        image = model.images[0]
        verdict = image.quality.verdict
        origin = ("от левого нижнего угла фасада" if model.facade.origin == "bottom_left"
                  else "от точки оператора (первый конец базы), не от угла здания")
        self.header.setText(
            f"Координаты — {origin}. Камера: "
            f"{CALIBRATION_TEXT.get(image.camera.calibration, image.camera.calibration)}. "
            f"Вердикт качества «{verdict}» зависит от введённой длины опорной базы.")
        self.table.setRowCount(len(model.elements))
        for row, e in enumerate(model.elements):
            s = e.size_mm
            xs = [p[0] for p in e.contour_mm]
            ys = [p[1] for p in e.contour_mm]
            if e.position_sigma_mm is not None:
                position = (f"{min(xs):.0f} / {min(ys):.0f} ± {e.position_sigma_mm:.1f}")
            else:
                position = f"{min(xs):.0f} / {min(ys):.0f}; {POSITION_SIGMA_PENDING}"
            if e.recess is None:
                recess = "грань откоса не размечена"
            elif e.recess.value_mm is not None and e.recess.sigma_mm is not None:
                recess = f"{e.recess.value_mm:.1f} ± {e.recess.sigma_mm:.1f}"
            else:
                recess = RECESS_ORIGIN_TEXT.get(e.recess.origin, e.recess.origin)
            theta = f"{e.theta.full_deg:.1f}°" if e.theta is not None else "—"
            cells = (e.id, f"{s.width:.1f} ± {s.sigma_width:.1f}",
                     f"{s.height:.1f} ± {s.sigma_height:.1f}", position, recess, theta,
                     tolerance_text(e, verdict))
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def cell(self, row: int, column: str) -> str:
        return self.table.item(row, self.COLUMNS.index(column)).text()


class SidePanel(QWidget):
    """Правая колонка окна: панели сверху вниз в порядке работы оператора.

    Колонка выше типового экрана ноутбука, поэтому окно кладёт её в прокрутку
    (`MainWindow`): без прокрутки на экране 1366×768 Qt ужимал бы поля ввода.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plane = PlanePanel(self)
        self.base = BasePanel(self)
        self.quality = QualityPanel(self)
        self.marks = MarksPanel(self)
        self.result = ResultPanel(self)
        self.layout_ = QVBoxLayout(self)
        self.layout_.addWidget(self.plane)
        self.layout_.addWidget(self.quality)
        self.layout_.addWidget(self.base)
        self.layout_.addWidget(self.marks)
        self.layout_.addWidget(self.result)
        self.layout_.addStretch(1)
