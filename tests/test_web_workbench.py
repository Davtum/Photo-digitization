"""Рабочая папка веб-интерфейса: библиотека, загрузки, сессии, метка. Задача 6."""
import json
import os
import shutil
import time

import pytest

from facade_digitizer.pipeline import run
from tests import web_scenes as ws

LABEL_A, LABEL_B = "op-aaaaaaaa", "op-bbbbbbbb"


@pytest.fixture(autouse=True)
def _cached_frames(monkeypatch):
    monkeypatch.setattr(run, "frame_stage", ws.cached_frame_stage)


@pytest.fixture(scope="module")
def near(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("исходник"), ws.NEAR)


@pytest.fixture
def data(tmp_path, near):
    d = tmp_path / "data"
    (d / "примеры").mkdir(parents=True)
    shutil.copy(near.path, d / "примеры" / "01_фасад.png")
    return d


def _bench(data, operator=None):
    from facade_digitizer.web.workbench import Workbench

    return Workbench(data, operator=operator, runner=ws.test_runner())


def _images(library):
    return {i["path"]: i for f in library["folders"] for i in f["images"]}


def _work(wb, near):
    """Базa и проём через рабочий стол открытого снимка."""
    desk = wb.desk
    ws.set_base(desk, near)
    ws.add_opening(desk, near, reveal=False)
    return desk


def test_library_lists_images_with_statuses_and_skips_service_dirs(data, near):
    (data / "примеры" / "экспорт").mkdir()
    shutil.copy(near.path, data / "примеры" / "экспорт" / "не_снимок.png")
    (data / ".thumbs").mkdir()
    shutil.copy(near.path, data / ".thumbs" / "x.png")
    (data / "примеры" / "заметки.txt").write_text("-", encoding="utf-8")
    wb = _bench(data)
    images = _images(wb.library())
    assert list(images) == ["примеры/01_фасад.png"]
    assert images["примеры/01_фасад.png"]["status"] == "new"

    wb.open("примеры/01_фасад.png")
    _work(wb, near)
    assert _images(wb.library())["примеры/01_фасад.png"]["status"] == "in_progress"
    wb.desk.act("export")
    assert _images(wb.library())["примеры/01_фасад.png"]["status"] == "exported"
    time.sleep(0.02)
    wb.desk.act("set_mark_attrs", id=wb.desk.current_mark, mounting="flush")
    assert _images(wb.library())["примеры/01_фасад.png"]["status"] == "changed_after_export"


def test_stem_collision_is_flagged_and_refused(data, near):
    shutil.copy(near.path, data / "примеры" / "01_фасад.jpg")
    wb = _bench(data)
    images = _images(wb.library())
    assert images["примеры/01_фасад.png"]["collision"]
    assert images["примеры/01_фасад.jpg"]["collision"]
    with pytest.raises(ValueError, match="одинаковое имя"):
        wb.open("примеры/01_фасад.png")


def test_upload_dedupes_by_sha256(data, near):
    wb = _bench(data)
    rel = wb.upload_image("новый.png", near.path.read_bytes())
    assert rel == "примеры/01_фасад.png"


def test_upload_renames_on_name_clash(data, near):
    wb = _bench(data)
    first = wb.upload_image("фото.jpg", b"\xff\xd8 first")
    second = wb.upload_image("фото.png", b"\x89PNG second")
    third = wb.upload_image("../../фото.jpg", b"\xff\xd8 third")
    assert first == "снимки/фото.jpg"
    assert second == "снимки/фото_2.png"          # та же основа — перезаписала бы сессию
    assert third == "снимки/фото_3.jpg"           # путь из имени отброшен
    with pytest.raises(ValueError, match="не снимок"):
        wb.upload_image("вирус.exe", b"MZ")


def test_resolve_refuses_paths_outside(data, tmp_path):
    wb = _bench(data)
    for bad in ("../x.png", str(tmp_path / "x.png"), "примеры/../../x.png"):
        with pytest.raises(ValueError, match="вне рабочей папки"):
            wb.resolve(bad)
    assert wb.resolve("примеры/01_фасад.png") == (data / "примеры" / "01_фасад.png").resolve()


def test_autosave_after_each_change(data, near):
    wb = _bench(data)
    wb.open("примеры/01_фасад.png")
    session_path = data / "примеры" / "01_фасад.session.json"
    assert not session_path.exists()             # открытие само ничего не пишет
    wb.desk.act("set_step", step="scale")
    assert not session_path.exists()             # переход по шагам — не изменение
    wb.desk.act("click", x=float(near.base_px[0][0]), y=float(near.base_px[0][1]),
                view_scale=1.0)
    saved = json.loads(session_path.read_text(encoding="utf-8"))
    assert len(saved["reference"]["ends"]) == 1
    assert wb.desk.state()["save"] == {"ok": True, "error": None}


def test_open_restores_session_next_to_image(data, near):
    wb = _bench(data, operator=LABEL_A)
    wb.open("примеры/01_фасад.png")
    desk = _work(wb, near)
    marks = [(m.id, [c.xy for c in m.corners]) for m in desk.session.marks]
    wb.close()
    again = _bench(data, operator=LABEL_A)
    again.open("примеры/01_фасад.png")
    restored = again.desk.session
    assert [(m.id, [c.xy for c in m.corners]) for m in restored.marks] == marks
    assert restored.scale is not None and restored.model is not None
    assert again.desk.step == "marks"


def test_autosave_failure_is_visible(data, near):
    wb = _bench(data)
    (data / "примеры" / "01_фасад.session.json").mkdir()      # на месте файла — папка
    # Папка на месте файла сессии — не файл сессии: открытие начинает новую сессию.
    wb.open("примеры/01_фасад.png")
    wb.desk.act("set_step", step="scale")
    wb.desk.act("click", x=float(near.base_px[0][0]), y=float(near.base_px[0][1]),
                view_scale=1.0)
    save = wb.desk.state()["save"]
    assert save["ok"] is False and save["error"]
    assert len(wb.desk.session.reference.ends) == 1              # работа продолжается


def test_restore_keeps_file_label_when_server_label_is_random(data, near):
    wb = _bench(data, operator=LABEL_A)
    wb.open("примеры/01_фасад.png")
    _work(wb, near)
    wb.close()
    random_label = _bench(data)
    random_label.open("примеры/01_фасад.png")
    assert random_label.desk.session.labour.operator == LABEL_A
    assert random_label.question is None


def test_explicit_other_label_asks_before_opening(data, near):
    wb = _bench(data, operator=LABEL_A)
    wb.open("примеры/01_фасад.png")
    _work(wb, near)
    marks = len(wb.desk.session.marks)
    wb.close()
    other = _bench(data, operator=LABEL_B)
    state = other.open("примеры/01_фасад.png")
    assert other.desk is None
    assert state["question"]["file_operator"] == LABEL_A
    assert state["question"]["operator"] == LABEL_B
    other.answer_operator("continue")
    assert other.desk.session.labour.operator == LABEL_B
    assert len(other.desk.session.marks) == marks
    other.close()
    third = _bench(data, operator=LABEL_A)
    third.open("примеры/01_фасад.png")               # теперь в файле — метка B
    third.answer_operator("restart")
    assert third.desk.session.marks == []
    assert third.desk.session.labour.operator == LABEL_A
    assert list((data / "примеры").glob("01_фасад.session.json.*.bak"))


def test_restart_archives_to_bak(data, near):
    wb = _bench(data)
    wb.open("примеры/01_фасад.png")
    _work(wb, near)
    wb.restart()
    assert wb.desk.session.marks == [] and wb.desk.session.reference.ends == []
    baks = list((data / "примеры").glob("01_фасад.session.json.*.bak"))
    assert len(baks) == 1
    assert json.loads(baks[0].read_text(encoding="utf-8"))["marks"]
    # Архив не попадает под `*.session.json` — сбор протокола σ клика его не увидит.
    assert not [p for p in (data / "примеры").glob("*.session.json") if p.is_file()]


def test_save_copy_names_with_timestamp(data, near):
    wb = _bench(data, operator=LABEL_A)
    wb.open("примеры/01_фасад.png")
    _work(wb, near)
    rel = wb.save_copy()
    assert rel.startswith(f"повторы/{LABEL_A}/01_фасад.") and rel.endswith(".session.json")
    copy = json.loads((data / rel).read_text(encoding="utf-8"))
    assert copy["marks"]
    with pytest.raises(ValueError, match="вне рабочей папки"):
        wb.save_copy("../наружу")


def test_thumbnail_is_cached(data):
    wb = _bench(data)
    first = wb.thumbnail("примеры/01_фасад.png")
    cached = list((data / ".thumbs").iterdir())
    assert len(cached) == 1 and cached[0].read_bytes() == first
    assert wb.thumbnail("примеры/01_фасад.png") == first


def test_upload_profile_checks_format(data, near, tmp_path):
    wb = _bench(data)
    good = ws.profile_for(tmp_path, near.sc)
    rel = wb.upload_profile("камера.json", good.read_bytes())
    assert rel == "профили/камера.json"
    assert [p["path"] for p in wb.profiles()] == ["профили/камера.json"]
    with pytest.raises(ValueError, match="не профиль калибровки"):
        wb.upload_profile("мусор.json", b'{"a": 1}')
    assert not (data / "профили" / "мусор.json").exists()


def test_act_checks_desk_and_revision(data):
    from facade_digitizer.web.workbench import Stale

    wb = _bench(data)
    wb.open("примеры/01_фасад.png")
    state = wb.state()
    wb.act(state["desk"], state["rev"], "set_step", {"step": "scale"})
    with pytest.raises(Stale):
        wb.act(state["desk"], state["rev"], "set_step", {"step": "frame"})
    fresh = wb.state()
    with pytest.raises(Stale):
        wb.act("другой", fresh["rev"], "set_step", {"step": "frame"})


def test_set_profile_path_is_resolved_inside_data(data, near, tmp_path):
    wb = _bench(data)
    rel = wb.upload_profile("камера.json", ws.profile_for(tmp_path, near.sc).read_bytes())
    wb.open("примеры/01_фасад.png")
    state = wb.state()
    wb.act(state["desk"], state["rev"], "set_profile", {"path": rel, "discard": True})
    assert wb.desk.session.profile_path == (data / rel).resolve()
    assert wb.state()["frame"]["profile"]["path"] == rel


def test_export_file_is_served_only_from_export_dir(data, near):
    wb = _bench(data)
    wb.open("примеры/01_фасад.png")
    _work(wb, near)
    wb.desk.act("export")
    assert wb.export_file("01_фасад.json").read_bytes().startswith(b"{")
    for bad in ("../01_фасад.session.json", "нет.json"):
        with pytest.raises((ValueError, FileNotFoundError)):
            wb.export_file(bad)


def test_state_carries_image_path_and_operator(data):
    wb = _bench(data, operator=LABEL_A)
    assert wb.state() == {"desk": None, "operator": LABEL_A, "question": None}
    wb.open("примеры/01_фасад.png")
    state = wb.state()
    assert state["image"]["path"] == "примеры/01_фасад.png"
    assert state["operator"] == LABEL_A
    assert os.path.isabs(str(wb.data_dir))
