"""Рабочая папка оператора. Веб-интерфейс, задача 6.

Всё, что окно Qt делало диалогами файлов, здесь делается без них:

* **библиотека** — снимки рабочей папки со статусом (новый / в работе / выгружен /
  изменён после выгрузки) и миниатюрой; снимок загружается перетаскиванием в
  `снимки/`, профиль камеры — в `профили/`;
* **сессия** пишется сама после каждого изменения — `<основа>.session.json` рядом со
  снимком (имя по умолчанию окна Qt; там же её ищет протокол σ клика) — и сама
  продолжается при открытии снимка;
* **метка оператора** (п. 14): явная (`--operator`) — сессия чужой метки не
  продолжается молча; случайная — продолжается метка файла.

Пути из запросов страницы принимаются только ВНУТРИ рабочей папки (`resolve`).
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from facade_digitizer.pipeline.calib import load_profile
from facade_digitizer.ui import session_file
from facade_digitizer.ui.labour import checked_label, opaque_label
from facade_digitizer.ui.session import OperatorSession, _file_hash
from facade_digitizer.web import images
from facade_digitizer.web.desk import Desk
from facade_digitizer.web.jobs import JobRunner

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
UPLOAD_DIR, PROFILE_DIR, EXPORT_DIR, THUMB_DIR = "снимки", "профили", "экспорт", ".thumbs"
#: Папки, где снимков оператора не бывает: выгрузка, кэш, профили.
SERVICE_DIRS = {EXPORT_DIR, THUMB_DIR, PROFILE_DIR}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
SESSION_SUFFIX = ".session.json"


class Stale(Exception):
    """Действие по устаревшему состоянию: другой снимок или ревизия ниже текущей."""


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_name(name: str) -> str:
    """Имя файла без пути и без символов, недопустимых в Windows."""
    base = Path(str(name).replace("\\", "/")).name
    base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base).strip(" .")
    if not base:
        raise ValueError("пустое имя файла")
    return base


def session_path_for(image: Path) -> Path:
    return image.with_suffix(SESSION_SUFFIX)


class Workbench:
    def __init__(self, data_dir, *, operator: str | None = None,
                 runner: JobRunner | None = None):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.explicit_operator = operator is not None
        self.operator = checked_label(operator) if operator is not None else opaque_label()
        self.runner = runner or JobRunner()
        self.lock = threading.RLock()
        self.desk: Desk | None = None
        self.image_rel: str | None = None
        self.question: dict | None = None
        self._pending: tuple[str, OperatorSession] | None = None
        self._opened = 0

    # --- пути -------------------------------------------------------------------

    def resolve(self, rel) -> Path:
        path = (self.data_dir / str(rel)).resolve()
        if path != self.data_dir and self.data_dir not in path.parents:
            raise ValueError(f"путь вне рабочей папки: {rel}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.data_dir).as_posix()

    # --- библиотека ------------------------------------------------------------------

    def _image_files(self) -> list[Path]:
        out = []
        for root, dirs, files in os.walk(self.data_dir):
            dirs[:] = sorted(d for d in dirs
                             if not d.startswith(".") and d not in SERVICE_DIRS)
            out += [Path(root) / f for f in sorted(files)
                    if Path(f).suffix.lower() in IMAGE_SUFFIXES and not f.startswith(".")]
        return out

    @staticmethod
    def _collisions(files: list[Path]) -> set[Path]:
        """Снимки одной папки с одной основой имени: сессия и выгрузка строятся по
        основе и перезаписали бы друг друга."""
        seen: dict[tuple, list[Path]] = {}
        for f in files:
            seen.setdefault((f.parent, f.stem.lower()), []).append(f)
        return {f for group in seen.values() if len(group) > 1 for f in group}

    @staticmethod
    def status(image: Path) -> str:
        session = session_path_for(image)
        exported = image.parent / EXPORT_DIR / f"{image.stem}.json"
        if not session.is_file():
            return "exported" if exported.is_file() else "new"
        if not exported.is_file():
            return "in_progress"
        return ("exported" if exported.stat().st_mtime_ns >= session.stat().st_mtime_ns
                else "changed_after_export")

    def library(self) -> dict:
        files = self._image_files()
        collisions = self._collisions(files)
        folders: dict[str, list] = {}
        for f in files:
            folder = f.parent.relative_to(self.data_dir).as_posix()
            folders.setdefault("" if folder == "." else folder, []).append({
                "path": self.rel(f), "name": f.name, "status": self.status(f),
                "collision": f in collisions, "current": self.rel(f) == self.image_rel})
        return {"data_dir": str(self.data_dir), "operator": self.operator,
                "folders": [{"path": k, "images": v} for k, v in sorted(folders.items())],
                "profiles": self.profiles()}

    def thumbnail(self, rel: str) -> bytes:
        path = self.resolve(rel)
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            raise FileNotFoundError(rel)
        st = path.stat()
        key = hashlib.sha1(f"{self.rel(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        cache = self.data_dir / THUMB_DIR / f"{key}.jpg"
        if cache.is_file():
            return cache.read_bytes()
        data = images.thumbnail(path)
        cache.parent.mkdir(exist_ok=True)
        try:
            cache.write_bytes(data)
        except OSError:
            pass                                  # без кэша — медленнее, но работает
        return data

    # --- загрузки -------------------------------------------------------------------

    def upload_image(self, name: str, data: bytes) -> str:
        """Снимок в `снимки/`. Тот же файл (sha256), уже лежащий в рабочей папке, —
        не копия, а он сам: открывается прежняя работа."""
        name = _safe_name(name)
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{name}: не снимок (jpg, jpeg, png, tif, tiff)")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{name}: больше {MAX_UPLOAD_BYTES // 2 ** 20} МБ")
        digest = hashlib.sha256(data).hexdigest()
        for f in self._image_files():
            if f.stat().st_size == len(data) and _file_hash(f) == digest:
                return self.rel(f)
        target_dir = self.data_dir / UPLOAD_DIR
        target_dir.mkdir(exist_ok=True)
        stem, suffix = Path(name).stem, Path(name).suffix
        taken = {p.stem.lower() for p in target_dir.iterdir()
                 if p.suffix.lower() in IMAGE_SUFFIXES}
        candidate, n = stem, 1
        while candidate.lower() in taken:
            n += 1
            candidate = f"{stem}_{n}"
        target = target_dir / f"{candidate}{suffix}"
        target.write_bytes(data)
        return self.rel(target)

    @staticmethod
    def _is_profile(path: Path) -> bool:
        try:
            profile = load_profile(path)
            return len(profile.K) == 3 and all(len(r) == 3 for r in profile.K)
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
            return False

    def upload_profile(self, name: str, data: bytes) -> str:
        name = _safe_name(name)
        if Path(name).suffix.lower() != ".json":
            raise ValueError(f"{name}: профиль калибровки — файл .json")
        target_dir = self.data_dir / PROFILE_DIR
        target_dir.mkdir(exist_ok=True)
        for existing in target_dir.glob("*.json"):
            if existing.read_bytes() == data:
                return self.rel(existing)
        stem, n, target = Path(name).stem, 1, target_dir / name
        while target.exists():
            n += 1
            target = target_dir / f"{stem}_{n}.json"
        target.write_bytes(data)
        if not self._is_profile(target):
            target.unlink()
            raise ValueError(f"{name}: не профиль калибровки (нужны model, K, dist, "
                             "rms_px, image_size — как пишет facade-calibrate)")
        return self.rel(target)

    def profiles(self) -> list[dict]:
        found = sorted((self.data_dir / PROFILE_DIR).glob("*.json"))
        if self.desk is not None and self.desk.session.image_path is not None:
            here = self.desk.session.image_path.parent
            found += sorted(p for p in here.glob("*.json")
                            if not p.name.endswith((SESSION_SUFFIX, ".marks.json")))
        out, seen = [], set()
        for p in found:
            if p in seen or not self._is_profile(p):
                continue
            seen.add(p)
            try:
                out.append({"name": p.name, "path": self.rel(p)})
            except ValueError:
                continue
        return out

    # --- открытие и сессии ---------------------------------------------------------

    def open(self, rel: str) -> dict:
        with self.lock:
            path = self.resolve(rel)
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                raise FileNotFoundError(f"снимок не найден: {rel}")
            siblings = [p for p in path.parent.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem.lower() == path.stem.lower()]
            if len(siblings) > 1:
                raise ValueError(f"{path.name}: в папке есть снимок с тем же именем и другим "
                                 "расширением (одинаковое имя) — переименуйте один: файлы "
                                 "сессии и выгрузки перезаписали бы друг друга")
            self.close()
            rel = self.rel(path)
            session_path = session_path_for(path)
            notice = None
            session = None
            if session_path.is_file():
                try:
                    session = session_file.load_session(
                        session_path, search_dirs=[self.data_dir / PROFILE_DIR])
                except (ValueError, FileNotFoundError, KeyError, TypeError,
                        json.JSONDecodeError) as error:
                    bak = self._archive(session_path)
                    notice = {"kind": "error", "text": (
                        f"Прежняя сессия не открылась ({error}) и сохранена как {bak.name}; "
                        "начата новая.")}
            if session is not None and self.explicit_operator \
                    and session.labour.operator != self.operator:
                self._pending = (rel, session)
                self.question = {"image": rel, "file_operator": session.labour.operator,
                                 "operator": self.operator, "text": (
                                     f"Сессия этого снимка начата оператором "
                                     f"{session.labour.operator}. Продолжить под меткой "
                                     f"{self.operator} (время {session.labour.operator} "
                                     "останется в сумме) или начать заново?")}
                return self.state()
            if session is None:
                session = self._fresh(path)
            self._make_desk(rel, session, notice)
            return self.state()

    def answer_operator(self, choice: str) -> dict:
        with self.lock:
            if self._pending is None:
                raise ValueError("вопроса о метке оператора нет")
            rel, session = self._pending
            self._pending = self.question = None
            if choice == "continue":
                self._make_desk(rel, session, None)
                # Метка меняется после создания стола: смена — изменение сессии, и
                # автосохранение запишет её сразу, а не с первым кликом.
                session.labour.operator = self.operator
                self.desk._touch()
            elif choice == "restart":
                path = self.resolve(rel)
                self._archive(session_path_for(path))
                self._make_desk(rel, self._fresh(path), None)
            else:
                raise ValueError(f"нет такого ответа: {choice}")
            return self.state()

    def _fresh(self, path: Path) -> OperatorSession:
        session = OperatorSession()
        session.labour.operator = self.operator
        session.open_image(path)
        return session

    def _make_desk(self, rel: str, session: OperatorSession, notice) -> None:
        self._opened += 1
        target = session_path_for(self.resolve(rel))
        desk = Desk(session, runner=self.runner, lock=self.lock,
                    save=lambda s: session_file.save_session(s, target),
                    desk_id=f"{session.image_hash[:12]}-{self._opened}")
        self.desk, self.image_rel = desk, rel
        desk.start()
        if notice is not None:
            desk.notice = notice
            desk._bump()

    @staticmethod
    def _archive(session_path: Path) -> Path:
        """Прежняя сессия — в `<основа>.session.json.<время>.bak`: не теряется и не
        попадает под `*.session.json`, по которому собирает протокол σ клика."""
        bak = session_path.with_name(f"{session_path.name}.{_stamp()}.bak")
        n = 1
        while bak.exists():
            n += 1
            bak = session_path.with_name(f"{session_path.name}.{_stamp()}-{n}.bak")
        if session_path.is_dir():
            return bak
        os.replace(session_path, bak)
        return bak

    def restart(self) -> dict:
        with self.lock:
            desk = self._need_desk()
            path = desk.session.image_path
            self.runner.cancel("frame")
            self.runner.cancel("raster")
            session_path = session_path_for(self.resolve(self.image_rel))
            if session_path.is_file():
                self._archive(session_path)
            self._make_desk(self.image_rel, self._fresh(path), None)
            return self.state()

    def save_copy(self, folder: str | None = None) -> str:
        """Копия сессии с временем в имени — повторы протокола σ клика
        (`--repeat-dir повторы/<метка>`)."""
        with self.lock:
            desk = self._need_desk()
            target_dir = self.resolve(folder or f"повторы/{desk.session.labour.operator}")
            target_dir.mkdir(parents=True, exist_ok=True)
            stem = desk.session.image_path.stem
            target = target_dir / f"{stem}.{_stamp()}{SESSION_SUFFIX}"
            n = 1
            while target.exists():
                n += 1
                target = target_dir / f"{stem}.{_stamp()}-{n}{SESSION_SUFFIX}"
            session_file.save_session(desk.session, target)
            return self.rel(target)

    def close(self) -> None:
        with self.lock:
            self.runner.cancel("frame")
            self.runner.cancel("raster")
            self.desk = self.image_rel = None
            self.question = self._pending = None

    def _need_desk(self) -> Desk:
        if self.desk is None:
            raise ValueError("снимок не открыт")
        return self.desk

    # --- действия и состояние ------------------------------------------------------

    def act(self, desk_id: str, rev: int, name: str, args: dict | None = None):
        """Действие страницы по состоянию `rev` рабочего стола `desk_id`."""
        with self.lock:
            desk = self.desk
            if desk is None or desk.desk_id != desk_id:
                raise Stale("открыт другой снимок")
            if int(rev) != desk.rev:
                raise Stale("состояние изменилось")
            args = dict(args or {})
            if name == "set_profile" and args.get("path"):
                args["path"] = str(self.resolve(args["path"]))
            return desk.act(name, **args)

    def state(self) -> dict:
        with self.lock:
            if self.desk is None:
                return {"desk": None, "operator": self.operator, "question": self.question}
            state = dict(self.desk.state())
            state["image"] = {**state["image"], "path": self.image_rel}
            state["operator"] = self.desk.session.labour.operator
            frame = state.get("frame")
            if frame and frame.get("profile"):
                try:
                    rel = self.rel(Path(frame["profile"]["path"]))
                except ValueError:
                    rel = None
                state["frame"] = {**frame, "profile": {**frame["profile"], "path": rel}}
            return state

    def export_file(self, name: str) -> Path:
        with self.lock:
            desk = self._need_desk()
            if _safe_name(name) != name:
                raise ValueError(f"недопустимое имя файла: {name}")
            path = desk.session.image_path.parent / EXPORT_DIR / name
            if not path.is_file():
                raise FileNotFoundError(name)
            return path

    def session_bytes(self) -> tuple[str, bytes]:
        with self.lock:
            desk = self._need_desk()
            raw = session_file.session_to_dict(desk.session)
            return (f"{desk.session.image_path.stem}{SESSION_SUFFIX}",
                    json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8"))
