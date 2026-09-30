"""HTTP-слой веб-интерфейса. Задача 7.

Сервер слушает только loopback, и всё равно его видит любой сайт, открытый в
браузере оператора: `POST` на 127.0.0.1 форма чужого сайта отправить может, а через
DNS rebinding — и прочитать ответ. Отсюда две проверки: `Host` и токен.
"""
import shutil

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from facade_digitizer.pipeline import run
from tests import web_scenes as ws

#: Предупреждение starlette об httpx относится к тестовому клиенту, не к серверу.
pytestmark = pytest.mark.filterwarnings("ignore:Using .httpx. with .starlette.testclient")

TOKEN = "t" * 32
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(autouse=True)
def _cached_frames(monkeypatch):
    monkeypatch.setattr(run, "frame_stage", ws.cached_frame_stage)


@pytest.fixture(scope="module")
def near(tmp_path_factory):
    return ws.build(tmp_path_factory.mktemp("апи"), ws.NEAR)


@pytest.fixture
def client(tmp_path, near):
    from facade_digitizer.web.app import create_app
    from facade_digitizer.web.workbench import Workbench

    data = tmp_path / "data"
    (data / "примеры").mkdir(parents=True)
    shutil.copy(near.path, data / "примеры" / "фасад.png")
    wb = Workbench(data, runner=ws.test_runner())
    app = create_app(wb, token=TOKEN)
    with TestClient(app, base_url=BASE, headers={"X-Facade-Token": TOKEN}) as c:
        c.workbench = wb
        yield c


def _act(client, state, name, **args):
    r = client.post("/api/action", json={"desk": state["desk"], "rev": state["rev"],
                                         "name": name, "args": args})
    assert r.status_code == 200, r.text
    return r.json()["state"]


def test_foreign_host_is_421(client):
    r = client.get("/api/ping", headers={"Host": "evil.example:8765"})
    assert r.status_code == 421
    # Порт не проверяется: контейнер могут опубликовать на другом порту хоста, а от
    # DNS rebinding защищает имя (у чужой страницы в Host — её домен).
    assert client.get("/api/ping", headers={"Host": "127.0.0.1:9999"}).status_code == 200
    assert client.get("/api/ping", headers={"Host": "[::1]:8765"}).status_code == 200
    assert client.get("/api/ping", headers={"Host": "localhost.evil.example"}).status_code == 421
    assert client.get("/api/ping", headers={"Host": f"localhost:{PORT}"}).status_code == 200


def test_post_without_token_is_403(client):
    r = client.post("/api/open", json={"path": "примеры/фасад.png"},
                    headers={"X-Facade-Token": ""})
    assert r.status_code == 403
    assert client.workbench.desk is None


def test_index_carries_the_token(client):
    r = client.get("/")
    assert r.status_code == 200 and TOKEN in r.text
    assert "no-store" in r.headers["cache-control"]


def test_upload_open_click_export_download_roundtrip(client, near):
    r = client.post("/api/upload", files={"file": ("мой снимок.png", b"\x89PNG not really")})
    assert r.status_code == 200 and r.json()["path"] == "снимки/мой снимок.png"
    library = client.get("/api/library").json()
    paths = [i["path"] for f in library["folders"] for i in f["images"]]
    assert "примеры/фасад.png" in paths
    state = client.post("/api/open", json={"path": "примеры/фасад.png"}).json()["state"]
    assert state["frame"]["ready"]
    state = _act(client, state, "set_step", step="scale")
    for x, y in near.base_px:
        state = _act(client, state, "click", x=float(x), y=float(y), view_scale=1.0)
    state = _act(client, state, "set_base", span_mm=near.span_mm, span_sigma_mm=None,
                 scale_source="operator_reference", origin_is_facade_corner=False)
    state = _act(client, state, "new_mark", class_="window")
    for x, y in near.corners:
        state = _act(client, state, "click", x=float(x), y=float(y), view_scale=4.0)
    assert state["result"]["rows"]
    state = _act(client, state, "export")
    names = [f["name"] for f in state["result"]["export"]["files"]]
    assert "фасад.json" in names and "фасад.dxf" in names
    r = client.get("/api/export/фасад.json")
    assert r.status_code == 200 and r.json()["schema_version"] == "1.2"
    r = client.get("/api/session/download")
    assert r.status_code == 200 and r.json()["format"] == "facade-digitizer-operator-session"
    assert "attachment" in r.headers["content-disposition"]


def test_frame_png_matches_frame(client):
    state = client.post("/api/open", json={"path": "примеры/фасад.png"}).json()["state"]
    r = client.get(f"/api/frame.png?v={state['frame']['version']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "immutable" in r.headers["cache-control"]
    image = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(image, client.workbench.desk.session.frame.frame.color)
    assert client.get("/api/usable.png").status_code == 200
    assert client.get("/api/thumb", params={"path": "примеры/фасад.png"}).status_code == 200


def test_stale_revision_is_409_with_state(client):
    state = client.post("/api/open", json={"path": "примеры/фасад.png"}).json()["state"]
    _act(client, state, "set_step", step="scale")
    r = client.post("/api/action", json={"desk": state["desk"], "rev": state["rev"],
                                         "name": "set_step", "args": {"step": "frame"}})
    assert r.status_code == 409
    assert r.json()["state"]["rev"] > state["rev"]
    assert r.json()["state"]["step"] == "scale"


def test_busy_desk_is_409(client):
    state = client.post("/api/open", json={"path": "примеры/фасад.png"}).json()["state"]
    client.workbench.desk.busy = "обрабатывается"
    r = client.post("/api/action", json={"desk": state["desk"], "rev": state["rev"],
                                         "name": "click",
                                         "args": {"x": 1.0, "y": 1.0, "view_scale": 1.0}})
    assert r.status_code == 409 and "обрабатывается" in r.json()["error"]


def test_unknown_action_is_400(client):
    state = client.post("/api/open", json={"path": "примеры/фасад.png"}).json()["state"]
    r = client.post("/api/action", json={"desk": state["desk"], "rev": state["rev"],
                                         "name": "_frame_ready", "args": {}})
    assert r.status_code == 400


def test_path_traversal_is_refused(client):
    for path in ("../секрет.png", "C:/Windows/win.ini"):
        r = client.post("/api/open", json={"path": path})
        assert r.status_code in (400, 404)
    r = client.get("/api/thumb", params={"path": "../../x.png"})
    assert r.status_code == 400
    r = client.get("/api/export/..%2F..%2Fsecret.json")
    assert r.status_code in (400, 404)


def test_state_without_open_image(client):
    r = client.get("/api/state")
    assert r.status_code == 200 and r.json()["state"]["desk"] is None


def test_restart_and_save_copy_and_close(client, near):
    state = client.post("/api/open", json={"path": "примеры/фасад.png"}).json()["state"]
    state = _act(client, state, "set_step", step="scale")
    _act(client, state, "click", x=float(near.base_px[0][0]), y=float(near.base_px[0][1]),
         view_scale=1.0)
    r = client.post("/api/session/copy", json={})
    assert r.status_code == 200 and r.json()["path"].endswith(".session.json")
    state = client.post("/api/restart").json()["state"]
    assert state["scale"]["ends"] == 0
    assert client.post("/api/close").json()["state"]["desk"] is None


def test_static_files_are_revalidated(client):
    """Модули страницы перепроверяются при каждом открытии: после обновления
    программы браузер не держит старый `app.js` рядом с новым сервером."""
    r = client.get("/static/app.js")
    assert r.status_code == 200 and "no-cache" in r.headers["cache-control"]


def test_unreadable_body_is_400_not_500(client):
    """Тело не в UTF-8 или не JSON (curl из консоли Windows отправляет кириллицу в
    ANSI) — ошибка запроса с объяснением, а не падение сервера."""
    for body in (b'{"path": "\xef\xf0\xe8"}', b"not json", b"[1, 2]"):
        r = client.post("/api/open", content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 400, body
        assert "JSON" in r.json()["error"]
