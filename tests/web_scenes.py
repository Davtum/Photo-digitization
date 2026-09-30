"""Синтетические сцены и помощники тестов веб-интерфейса.

Сцены — те же, что у тестов окна Qt плана 3 (`test_ui_marks`, `test_ui_result`,
`test_ui_plane`): ближняя (вердикт не `reject`, глубина измерима) и дальняя
(`reject` по разрешению). Клики — проекции ИСТИННЫХ углов сцены.

Фаза кадра стоит секунды, а тестов, открывающих один и тот же снимок, десятки:
`cached_frame_stage` запоминает результат по аргументам. Кэш — тестовый помощник, он
не подменяет путь `Desk` → `run.frame_stage` (вызов тот же), а только не повторяет
одинаковую работу.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facade_digitizer.pipeline import run

NEAR = {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0, "depth": 150.0}
FAR = {"dx": 3000.0, "dy": -2000.0, "dist": 20000.0, "depth": 150.0}
FACADE_W, FACADE_H = 20000.0, 15000.0


@dataclass
class Scene:
    sc: object
    path: Path
    base_px: np.ndarray
    span_mm: float
    corners: np.ndarray          # Н-Л, Н-П, В-П, В-Л
    side: str                    # видимая грань откоса
    inner: np.ndarray            # внутренняя кромка откоса: низ, верх
    facade_quad: np.ndarray      # углы фасада Н-Л, Н-П, В-П, В-Л


def build(directory: Path, view: dict, name: str = "facade.png") -> Scene:
    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline.io import save_image
    from tests.test_synth import make_scene

    if "scripts" not in sys.path:
        sys.path.insert(0, "scripts")
    from demo_synthetic import operator_base

    sc = make_scene(**view)
    path = Path(directory) / name
    save_image(path, sc.render())
    base_px, span_mm, _origin = operator_base(sc)
    op = sc.openings[0]
    side, _h = visible_reveal_side(sc.camera_on_plane(), op.x, op.x + op.width,
                                   op.y, op.y + op.height)
    x = op.x if side == "left" else op.x + op.width
    inner = sc.project(np.array([[x, op.y], [x, op.y + op.height]]), depth=op.depth)
    quad = sc.project(np.array([[0.0, 0.0], [FACADE_W, 0.0], [FACADE_W, FACADE_H],
                                [0.0, FACADE_H]]))
    return Scene(sc, path, base_px, span_mm, sc.project(op.corners_mm()), side, inner, quad)


def profile_for(directory: Path, sc, model: str = "unknown") -> Path:
    """Профиль под синтетический кадр: без EXIF модель камеры читается как «unknown»."""
    from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile

    path = Path(directory) / f"профиль_{model}.json"
    save_profile(CalibrationProfile(model=model, K=sc.K.tolist(), dist=[0.0] * 5,
                                    rms_px=0.2, image_size=tuple(sc.image_size)), path)
    return path


#: Кэш фаз кадра — последние несколько: кадр 20 Мп в цвете и в сером — около 85 МБ.
_FRAMES: dict = {}
_FRAMES_KEPT = 4
_REAL_FRAME_STAGE = run.frame_stage


def cached_frame_stage(image_path, *, profile_path=None, with_color=False,
                       operator_reference=None, plane_override=None, roi=None):
    if operator_reference is not None or plane_override is not None or roi is not None:
        return _REAL_FRAME_STAGE(image_path, profile_path=profile_path,
                                 with_color=with_color, operator_reference=operator_reference,
                                 plane_override=plane_override, roi=roi)
    key = (str(Path(image_path).resolve()), Path(image_path).stat().st_mtime_ns,
           str(profile_path), with_color)
    if key not in _FRAMES:
        while len(_FRAMES) >= _FRAMES_KEPT:
            _FRAMES.pop(next(iter(_FRAMES)))
        _FRAMES[key] = _REAL_FRAME_STAGE(image_path, profile_path=profile_path,
                                         with_color=with_color)
    return _FRAMES[key]


def test_runner(run_raster: bool = False):
    """`InlineRunner`, который растр строит только по просьбе теста.

    Растр (до секунд на кадр 20 Мп) нужен тестам выровненного вида, а десяткам
    остальных — нет; задача растра при этом отменяется так же, как отменённая.
    """
    from facade_digitizer.web.jobs import InlineRunner

    class TestRunner(InlineRunner):
        def _start(self, kind, generation, fn, on_done, on_error):
            if kind == "raster" and not self.run_raster:
                with self._lock:
                    self._unfinished.discard((kind, generation))
                return
            super()._start(kind, generation, fn, on_done, on_error)

    runner = TestRunner()
    runner.run_raster = run_raster
    return runner


def new_desk(path, *, profile=None, save=None, run_raster=False):
    """`Desk` над открытым снимком: фазы выполняются сразу, в вызывающем потоке."""
    from facade_digitizer.ui.session import OperatorSession
    from facade_digitizer.web.desk import Desk

    session = OperatorSession()
    session.open_image(path)
    if profile is not None:
        session.set_profile(profile)
    desk = Desk(session, runner=test_runner(run_raster), save=save, desk_id="test")
    desk.start()
    return desk


def set_base(desk, scene: Scene, *, scale=1.0, span_mm=None, **options):
    desk.act("set_step", step="scale")
    for x, y in scene.base_px:
        desk.act("click", x=float(x), y=float(y), view_scale=scale)
    desk.act("set_base", span_mm=scene.span_mm if span_mm is None else span_mm,
             span_sigma_mm=options.get("span_sigma_mm"),
             scale_source=options.get("scale_source", "operator_reference"),
             origin_is_facade_corner=options.get("origin_is_facade_corner", False))


def add_opening(desk, scene: Scene, *, scale=4.0, order=(0, 1, 2, 3), reveal=True,
                class_="window"):
    desk.act("set_step", step="marks")
    desk.act("new_mark", class_=class_)
    mark_id = desk.current_mark
    for i in order:
        desk.act("click", x=float(scene.corners[i][0]), y=float(scene.corners[i][1]),
                 view_scale=scale)
    if reveal:
        desk.act("set_reveal_side", side=scene.side)
        desk.act("start_reveal")
        for x, y in scene.inner:
            desk.act("click", x=float(x), y=float(y), view_scale=scale)
    return mark_id
