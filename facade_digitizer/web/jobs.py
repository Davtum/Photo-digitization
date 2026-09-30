"""Долгие фазы — в фоне, с поколениями. Веб-интерфейс, задача 2.

Фаза кадра стоит до 4 с на снимке 20 Мп (с кодированием PNG — немного больше),
растр наибольшего разрешения — до 4 с; сервер при этом отвечает на запросы.

**Что значит «отмена»** — то же, что у `ui.worker` окна Qt: прервать расчёт OpenCV
изнутри нельзя, отмена отказывается от результата. У каждого вида задач (`kind`) —
номер поколения; `submit` и `cancel` его поднимают, и результат доставляется, только
если его поколение всё ещё последнее. Иначе поздний результат старой задачи
перезаписал бы результат задачи, запущенной после неё.

`on_done` и `on_error` вызываются в потоке пула: вызывающий сам берёт свою
блокировку.
"""
import threading
from concurrent.futures import ThreadPoolExecutor


class JobRunner:
    def __init__(self, max_workers: int = 2):
        self._pool = (ThreadPoolExecutor(max_workers=max_workers,
                                         thread_name_prefix="facade-job")
                      if max_workers else None)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._generation: dict[str, int] = {}
        #: Незавершённые задачи: (вид, поколение).
        self._unfinished: set[tuple[str, int]] = set()

    def submit(self, kind: str, fn, on_done, on_error) -> int:
        with self._lock:
            generation = self._generation.get(kind, 0) + 1
            self._generation[kind] = generation
            self._unfinished.add((kind, generation))
        self._start(kind, generation, fn, on_done, on_error)
        return generation

    def _start(self, kind, generation, fn, on_done, on_error) -> None:
        self._pool.submit(self._run, kind, generation, fn, on_done, on_error)

    def cancel(self, kind: str) -> None:
        with self._lock:
            self._generation[kind] = self._generation.get(kind, 0) + 1

    def pending(self, kind: str) -> bool:
        """Идёт ли задача этого вида, чей результат ещё будет доставлен."""
        with self._lock:
            return (kind, self._generation.get(kind)) in self._unfinished

    def _current(self, kind: str, generation: int) -> bool:
        with self._lock:
            return self._generation.get(kind) == generation

    def _run(self, kind, generation, fn, on_done, on_error) -> None:
        try:
            try:
                result = fn()
            except Exception as error:  # noqa: BLE001 — доходит до страницы текстом
                if self._current(kind, generation):
                    on_error(str(error) or type(error).__name__)
            else:
                if self._current(kind, generation):
                    on_done(result)
        finally:
            with self._lock:
                self._unfinished.discard((kind, generation))
                self._idle.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        """Дождаться, пока все задачи закончатся (тесты, демо)."""
        with self._lock:
            return self._idle.wait_for(lambda: not self._unfinished, timeout)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


class InlineRunner(JobRunner):
    """Выполняет задачу сразу в вызывающем потоке: тесты `Desk` и демо без пула."""

    def __init__(self):
        super().__init__(max_workers=0)

    def _start(self, kind, generation, fn, on_done, on_error) -> None:
        self._run(kind, generation, fn, on_done, on_error)
