"""Файл сессии оператора и экспорт результата. План 3, задача 17.

Два формата, которые НЕ смешиваются:

* **Сессия** — внутренний формат интерфейса со своей версией (`SESSION_VERSION`):
  снимок и профиль с хэшами, ручная плоскость, область оценки, опорная база,
  черновики разметки с масштабом просмотра КАЖДОГО клика, хронометраж. Из неё работа
  продолжается с того места, где оператор остановился. Qt здесь не импортируется.
* **Выход** — `FacadeModel` схемы 1.2, тот же, что пишет CLI.

Экспорт — ОДНА функция (`export`), дающая три вещи сразу: выходной JSON, `marks.json`
(σ и идентификатор у каждого элемента) и строку команды CLI, которая по этим двум
файлам воспроизводит тот же выходной JSON байт в байт. Пути приводятся `resolve()`:
`images[].path` пишется как задан, и относительный путь у CLI против абсолютного из
диалога дал бы разный файл при одной и той же разметке.
"""
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facade_digitizer.pipeline import run
from facade_digitizer.pipeline.plane import ManualPlane
from facade_digitizer.ui.session import (
    ClickedPoint,
    MarkDraft,
    OperatorSession,
    ReferenceDraft,
    SigmaOf,
    _file_hash,
)

SESSION_FORMAT = "facade-digitizer-operator-session"
SESSION_VERSION = 1


# --- сессия ------------------------------------------------------------------------

def _points(points) -> list[dict]:
    return [{"x": p.x, "y": p.y, "view_scale": p.view_scale} for p in points]


def _clicked(raw) -> list[ClickedPoint]:
    return [ClickedPoint(float(p["x"]), float(p["y"]), float(p["view_scale"])) for p in raw]


def _plane(override: ManualPlane | None) -> dict | None:
    if override is None:
        return None
    return {"image_pts": np.asarray(override.image_pts, dtype=float).tolist(),
            "aspect_ratio": override.aspect_ratio,
            "size_mm": list(override.size_mm) if override.size_mm is not None else None,
            "assume_calibrated": bool(override.assume_calibrated)}


def session_to_dict(session: OperatorSession) -> dict:
    if session.image_path is None:
        raise ValueError("сессию без открытого снимка сохранять нечего")
    r = session.reference
    return {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "image": {"path": str(session.image_path), "sha256": session.image_hash},
        "profile": (None if session.profile_path is None else
                    {"path": str(session.profile_path), "sha256": session.profile_hash}),
        "plane_points": _points(session.plane_points),
        "manual_plane": _plane(session.plane_override),
        "roi_points": _points(session.roi_points),
        "roi": None if session.roi is None else np.asarray(session.roi, float).tolist(),
        "reference": {"ends": _points(r.ends), "span_mm": r.span_mm,
                      "scale_source": r.scale_source, "span_sigma_mm": r.span_sigma_mm,
                      "origin_is_facade_corner": r.origin_is_facade_corner},
        "next_mark": session._next_mark,
        "marks": [{"id": m.id, "class": m.class_, "mounting": m.mounting,
                   "edge_type": m.edge_type, "corners": _points(m.corners),
                   "reveal_side": m.reveal_side, "reveal_points": _points(m.reveal_points)}
                  for m in session.marks],
        "timing": session.timing,
    }


def save_session(session: OperatorSession, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(session_to_dict(session), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def _checked_file(label: str, raw: dict) -> Path:
    """Файл, на который ссылается сессия, обязан быть ТЕМ ЖЕ: клики привязаны к
    пикселям именно этого снимка, а кадр — к именно этому профилю."""
    path = Path(raw["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} сессии не найден: {path}")
    actual = _file_hash(path)
    if actual != raw["sha256"]:
        raise ValueError(
            f"{label} {path.name} не тот, по которому сделана сессия (sha256 "
            f"{actual[:12]}… вместо {raw['sha256'][:12]}…): точки сессии к нему "
            "не относятся")
    return path


def load_session(path) -> OperatorSession:
    """Сессия из файла. Чужой снимок или профиль — отказ с именем файла."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format") != SESSION_FORMAT:
        raise ValueError(f"{Path(path).name}: не файл сессии оператора")
    if raw.get("version") != SESSION_VERSION:
        raise ValueError(f"{Path(path).name}: версия сессии {raw.get('version')}, "
                         f"поддерживается {SESSION_VERSION}")
    image = _checked_file("снимок", raw["image"])
    profile = (_checked_file("профиль калибровки", raw["profile"])
               if raw["profile"] is not None else None)
    session = OperatorSession()
    session.open_image(image)
    if profile is not None:
        session.set_profile(profile)
    session.plane_points = _clicked(raw["plane_points"])
    plane = raw["manual_plane"]
    if plane is not None:
        session.plane_override = ManualPlane(
            image_pts=np.asarray(plane["image_pts"], dtype=float),
            aspect_ratio=plane["aspect_ratio"],
            size_mm=tuple(plane["size_mm"]) if plane["size_mm"] is not None else None,
            assume_calibrated=plane["assume_calibrated"])
    session.roi_points = _clicked(raw["roi_points"])
    session.roi = None if raw["roi"] is None else np.asarray(raw["roi"], dtype=float)
    ref = raw["reference"]
    session.reference = ReferenceDraft(
        ends=_clicked(ref["ends"]), span_mm=ref["span_mm"],
        scale_source=ref["scale_source"], span_sigma_mm=ref["span_sigma_mm"],
        origin_is_facade_corner=ref["origin_is_facade_corner"])
    session.marks = [MarkDraft(id=m["id"], class_=m["class"], mounting=m["mounting"],
                               edge_type=m["edge_type"], corners=_clicked(m["corners"]),
                               reveal_side=m["reveal_side"],
                               reveal_points=_clicked(m["reveal_points"]))
                     for m in raw["marks"]]
    session._next_mark = int(raw["next_mark"])
    session.timing = dict(raw.get("timing") or {})
    return session


# --- экспорт -----------------------------------------------------------------------

@dataclass(frozen=True)
class Exported:
    json_path: Path
    marks_path: Path | None
    cli_args: list[str]
    dxf_path: Path | None = None

    @property
    def cli_line(self) -> str:
        return shlex.join(["facade-digitize", *self.cli_args])


def _num(value: float) -> str:
    """`repr` вещественного числа восстанавливается `float()` без потерь."""
    return repr(float(value))


def cli_args(session: OperatorSession, sigma_of: SigmaOf, raster_mm_per_px: float,
             out_dir: Path, marks_path: Path | None) -> list[str]:
    ref = session.operator_reference(sigma_of)
    (x1, y1), (x2, y2) = ref.span_px
    args = [str(session.image_path),
            "--raster-mm-per-px", _num(raster_mm_per_px),
            "--origin-px", _num(ref.origin_px[0]), _num(ref.origin_px[1]),
            "--span-px", _num(x1), _num(y1), _num(x2), _num(y2),
            "--span-mm", _num(ref.span_mm),
            "--sigma-px", _num(ref.sigma_px),
            "--end-sigma-px", *map(_num, ref.end_sigma_px),
            "--scale-source", session.reference.scale_source]
    if ref.span_sigma_mm is not None:
        args += ["--span-sigma-mm", _num(ref.span_sigma_mm)]
    if ref.origin_is_facade_corner:
        args.append("--origin-is-facade-corner")
    if session.profile_path is not None:
        args += ["--profile", str(session.profile_path)]
    if marks_path is not None:
        args += ["--marks", str(marks_path)]
    plane = session.plane_override
    if plane is not None:
        args += ["--manual-plane",
                 *map(_num, np.asarray(plane.image_pts, dtype=float).ravel())]
        if plane.aspect_ratio is not None:
            args += ["--manual-aspect", _num(plane.aspect_ratio)]
        elif plane.size_mm is not None:
            args += ["--manual-size-mm", *map(_num, plane.size_mm)]
        else:
            args.append("--manual-calibrated")
    elif session.roi is not None:
        args += ["--roi", *map(_num, np.asarray(session.roi, dtype=float).ravel())]
    args += ["--out-dir", str(out_dir)]
    return args


def export(session: OperatorSession, out_dir, sigma_of: SigmaOf, *,
           dxf: bool = True) -> Exported:
    """Выходной JSON, `marks.json` и строка CLI — из одного и того же состояния;
    с `dxf` — и чертёж (задача 18), и тогда строка CLI несёт `--dxf`.

    Разрешение растра берётся тем же автоподбором, каким его берёт окно
    (`run.auto_raster_mm_per_px`), и передаётся CLI явно: у CLI оно обязательно.
    """
    out_dir = Path(out_dir).resolve()
    model = session.compute_elements(sigma_of)
    if model is None:
        raise ValueError(f"экспортировать нечего: {session.error}")
    mm_per_px = run.auto_raster_mm_per_px(session.frame, session.scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = session.image_path.stem
    marks = session.element_marks(sigma_of)
    marks_path = None
    if marks:
        marks_path = out_dir / f"{stem}.marks.json"
        marks_path.write_text(json.dumps(
            [m.model_dump(by_alias=True, exclude_none=True, mode="json") for m in marks],
            ensure_ascii=False, indent=2), encoding="utf-8")
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    args = cli_args(session, sigma_of, mm_per_px, out_dir, marks_path)
    dxf_path = None
    if dxf:
        from facade_digitizer.pipeline.export_dxf import to_dxf

        dxf_path = to_dxf(model, json_path.with_suffix(".dxf"))
        args.insert(args.index("--out-dir"), "--dxf")
    (out_dir / f"{stem}.command.txt").write_text(
        shlex.join(["facade-digitize", *args]) + "\n", encoding="utf-8")
    return Exported(json_path=json_path, marks_path=marks_path, cli_args=args,
                    dxf_path=dxf_path)
