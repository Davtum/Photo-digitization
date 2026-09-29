"""Долгие фазы — в фоне, с отменой. План 3, задача 9.

Фаза кадра стоит до 4 с на снимке 20 Мп, растр наибольшего разрешения — до 4 с;
окно при этом не замирает. Результат доставляется в поток интерфейса очередью
сигналов Qt.

**Что значит «отмена».** Прервать расчёт OpenCV изнутри нельзя: отмена отказывается
от результата, а не останавливает вычисление. Окно освобождается сразу, а поздний
результат отменённой задачи не доставляется — иначе он перезаписал бы результат
задачи, запущенной после отмены.
"""
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Runnable(QRunnable):
    def __init__(self, fn, signals):
        super().__init__()
        self.fn, self.signals = fn, signals
        self.setAutoDelete(True)

    def run(self):
        try:
            result = self.fn()
        except Exception as error:  # noqa: BLE001 — доходит до окна текстом
            self._emit(self.signals.failed, str(error))
        else:
            self._emit(self.signals.done, result)

    @staticmethod
    def _emit(signal, value) -> None:
        # Окно могло быть закрыто, пока шёл расчёт (растр — до 4 с): тогда объект
        # сигналов уже удалён, и доставлять результат некому. Прежде это печатало
        # трассировку из фонового потока («Signal source has been deleted»).
        try:
            signal.emit(value)
        except RuntimeError:
            pass


#: Задачи, результат которых ещё не доставлен. Модуль держит на них ссылку сам:
#: иначе задача, которую вызывающий не сохранил, уходит в сборщик мусора вместе с
#: объектом сигналов, и результат теряется молча — окно ждало бы фазу кадра вечно.
#: Поймано тестом `test_background_job_reports_errors_as_text`.
_ACTIVE: set = set()


class Job:
    """Задача в фоне. `cancel()` — отказ от результата; см. докстринг модуля."""

    def __init__(self, on_done, on_error):
        self.cancelled = False
        self.finished = False
        self._on_done, self._on_error = on_done, on_error
        self.signals = _Signals()
        self.signals.done.connect(self._deliver)
        self.signals.failed.connect(self._fail)

    def cancel(self) -> None:
        self.cancelled = True

    def _deliver(self, result) -> None:
        self.finished = True
        _ACTIVE.discard(self)
        if not self.cancelled:
            self._on_done(result)

    def _fail(self, message: str) -> None:
        self.finished = True
        _ACTIVE.discard(self)
        if not self.cancelled:
            self._on_error(message)


def run_in_background(fn, *, on_done, on_error) -> Job:
    """Выполнить `fn()` в пуле потоков; результат или текст ошибки — в поток окна."""
    job = Job(on_done, on_error)
    _ACTIVE.add(job)
    QThreadPool.globalInstance().start(_Runnable(fn, job.signals))
    return job
