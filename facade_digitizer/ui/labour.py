"""Метрика трудозатрат оператора. План 3, задача 22; спецификация, раздел 10 и п. 14.

Что считается:

* время на снимок и на элемент — сумма интервалов между последовательными
  действиями оператора, каждый интервал отнесён к элементу, над которым сделано
  завершающее его действие; перерыв длиннее `IDLE_CAP_S` засчитывается как
  `IDLE_CAP_S` — оператор ушёл, а не размечал;
* правки (перенос точки) и отмены — по элементу и всего;
* время при масштабе грубее рекомендованного — интервалы, завершённые кликом
  с таким масштабом (`zoom.RECOMMENDED_MIN_SCALE`);
* время на кадрах с вердиктом `reject` — отдельно: разметка там идёт, но
  признака допуска не будет, и смешивать это время с годными кадрами нельзя.

**Часы монотонные** (`time.monotonic`): перевод системного времени, в том числе
назад, не делает интервал отрицательным и не дарит оператору час. Часы
подставляемые — так это проверяется без ожидания.

**Метка оператора непрозрачна.** Она назначается исследованием, а не вводится
именем: `op-` и восемь шестнадцатеричных знаков (`opaque_label`). Любая иная
строка — отказ: имя в выходном файле — персональные данные (п. 14), и для
разбора эффекта обучения оно не нужно.
"""
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from facade_digitizer.ui.zoom import RECOMMENDED_MIN_SCALE

#: Наибольший засчитываемый интервал между действиями, с.
IDLE_CAP_S = 60.0
_LABEL = re.compile(r"^op-[0-9a-f]{8}$")

#: Виды действий оператора.
KINDS = ("new_mark", "click", "edit", "undo", "delete")


def opaque_label() -> str:
    """Новая непрозрачная метка оператора: `op-` и 8 случайных hex-знаков."""
    return f"op-{secrets.token_hex(4)}"


def checked_label(label: str) -> str:
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise ValueError(
            f"метка оператора {label!r} не непрозрачна: ожидается «op-» и восемь "
            "шестнадцатеричных знаков (назначается исследованием, не имя; п. 14)")
    return label


@dataclass
class ElementLabour:
    edits: int = 0
    undos: int = 0
    seconds: float = 0.0


@dataclass
class LabourLog:
    operator: str = field(default_factory=opaque_label)
    clock: Callable[[], float] = time.monotonic
    elements: dict[str, ElementLabour] = field(default_factory=dict)
    total_s: float = 0.0
    coarse_scale_s: float = 0.0
    reject_s: float = 0.0
    edits: int = 0
    undos: int = 0
    _last: float | None = None

    def __post_init__(self):
        checked_label(self.operator)

    def event(self, kind: str, mark_id: str | None, *, view_scale: float | None = None,
              reject: bool = False) -> None:
        """Действие оператора. Интервал от предыдущего действия — этому элементу."""
        if kind not in KINDS:
            raise ValueError(f"неизвестное действие оператора: {kind}")
        now = self.clock()
        dt = 0.0 if self._last is None else min(max(now - self._last, 0.0), IDLE_CAP_S)
        self._last = now
        self.total_s += dt
        if view_scale is not None and view_scale < RECOMMENDED_MIN_SCALE:
            self.coarse_scale_s += dt
        if reject:
            self.reject_s += dt
        if mark_id is None:
            return
        entry = self.elements.setdefault(mark_id, ElementLabour())
        entry.seconds += dt
        if kind == "edit":
            entry.edits += 1
            self.edits += 1
        elif kind == "undo":
            entry.undos += 1
            self.undos += 1

    def start(self) -> None:
        """Начало работы со снимком: первый интервал отсчитывается отсюда."""
        self._last = self.clock()

    def operator_record(self, mark_id: str) -> dict:
        """Поле `Element.operator`: правки — переносы и отмены вместе."""
        entry = self.elements.get(mark_id, ElementLabour())
        return {"id": self.operator, "edits": entry.edits + entry.undos,
                "seconds": round(entry.seconds, 3)}

    def summary(self) -> dict:
        """Сводка для файла сессии."""
        return {
            "operator": self.operator,
            "total_s": round(self.total_s, 3),
            "coarse_scale_s": round(self.coarse_scale_s, 3),
            "coarse_scale_fraction": (round(self.coarse_scale_s / self.total_s, 4)
                                      if self.total_s > 0 else 0.0),
            "reject_s": round(self.reject_s, 3),
            "edits": self.edits,
            "undos": self.undos,
            "elements": {k: {"edits": v.edits, "undos": v.undos,
                             "seconds": round(v.seconds, 3)}
                         for k, v in self.elements.items()},
        }

    @classmethod
    def from_summary(cls, raw: dict, clock: Callable[[], float] = time.monotonic):
        log = cls(operator=raw["operator"], clock=clock)
        log.total_s = float(raw["total_s"])
        log.coarse_scale_s = float(raw["coarse_scale_s"])
        log.reject_s = float(raw["reject_s"])
        log.edits, log.undos = int(raw["edits"]), int(raw["undos"])
        log.elements = {k: ElementLabour(int(v["edits"]), int(v["undos"]),
                                         float(v["seconds"]))
                        for k, v in raw["elements"].items()}
        return log
