"""Фоновые задачи и кодирование изображений веб-интерфейса. Задача 2."""
import threading

import cv2
import numpy as np

from facade_digitizer.web import images
from facade_digitizer.web.jobs import InlineRunner, JobRunner


def test_stale_generation_is_not_delivered():
    """Поздний результат устаревшей задачи не перезаписывает свежий (семантика ui.worker)."""
    runner = JobRunner()
    release = threading.Event()
    delivered = []

    def slow():
        release.wait(5)
        return "первая"

    runner.submit("frame", slow, delivered.append, delivered.append)
    runner.submit("frame", lambda: "вторая", delivered.append, delivered.append)
    release.set()
    assert runner.wait_idle(5)
    assert delivered == ["вторая"]
    assert not runner.pending("frame")
    runner.shutdown()


def test_cancel_drops_the_result():
    runner = JobRunner()
    release = threading.Event()
    delivered = []
    runner.submit("raster", lambda: release.wait(5), delivered.append, delivered.append)
    assert runner.pending("raster")
    runner.cancel("raster")
    release.set()
    assert runner.wait_idle(5)
    assert delivered == [] and not runner.pending("raster")
    runner.shutdown()


def test_errors_are_delivered_as_text():
    runner = JobRunner()
    errors = []

    def fail():
        raise ValueError("кадр не читается")

    runner.submit("frame", fail, lambda _r: None, errors.append)
    assert runner.wait_idle(5)
    assert errors == ["кадр не читается"]
    runner.shutdown()


def test_inline_runner_runs_now():
    runner = InlineRunner()
    got = []
    runner.submit("frame", lambda: 42, got.append, got.append)
    assert got == [42] and not runner.pending("frame")


def test_png_roundtrip_is_lossless():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(images.png(bgr), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded, bgr)


def test_usable_png_marks_unusable():
    mask = np.array([[True, False], [False, True]])
    rgba = cv2.imdecode(np.frombuffer(images.usable_png(mask), np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 110


def test_invalid_png_hatches_only_outside_the_raster():
    valid = np.ones((20, 20), bool)
    valid[:, :5] = False
    rgba = cv2.imdecode(np.frombuffer(images.invalid_png(valid), np.uint8),
                        cv2.IMREAD_UNCHANGED)
    assert (rgba[:, 5:, 3] == 0).all() and (rgba[:, :5, 3] > 0).all()


def test_thumbnail_is_small_jpeg(tmp_path):
    path = tmp_path / "снимок.png"
    cv2.imencode(".png", np.full((1200, 1600, 3), 128, np.uint8))[1].tofile(str(path))
    thumb = cv2.imdecode(np.frombuffer(images.thumbnail(path, width=320), np.uint8),
                         cv2.IMREAD_COLOR)
    assert thumb.shape[1] == 320 and thumb.shape[0] == 240
