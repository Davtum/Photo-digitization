"""HTTP-слой веб-интерфейса. Задача 7.

Тонкий: маршруты переводят запросы в вызовы `Workbench`/`Desk` и обратно; логики
экрана здесь нет.

**Безопасность `localhost`.** Сервер слушает только loopback, но его видит любой
сайт, открытый в браузере оператора: форма чужой страницы может отправить `POST`
на 127.0.0.1, а через DNS rebinding чужая страница может и читать ответы. Поэтому:

1. заголовок `Host` — только `localhost`, `127.0.0.1` или `[::1]` с портом сервера,
   иначе `421` (закрывает rebinding: у чужого имени другой `Host`);
2. изменяющие запросы — только с заголовком `X-Facade-Token` (случайный на запуск,
   вписывается в страницу при её отдаче), иначе `403`: форма чужого сайта заголовок
   не поставит, а `fetch` с ним упрётся в CORS-preflight, которого сервер не
   разрешает.
"""
import secrets
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from facade_digitizer.web.desk import DeskBusy
from facade_digitizer.web.workbench import Stale, Workbench

STATIC = Path(__file__).with_name("static")
LOOPBACK = {"localhost", "127.0.0.1", "[::1]"}
NO_STORE = {"Cache-Control": "no-store"}
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def new_token() -> str:
    return secrets.token_urlsafe(24)


def _host_ok(host: str, port: int | None) -> bool:
    if not host:
        return False
    if host.startswith("["):
        name, _, rest = host.partition("]")
        name, port_text = name + "]", rest.lstrip(":")
    else:
        name, _, port_text = host.partition(":")
    if name.lower() not in LOOPBACK:
        return False
    return port is None or port_text == str(port)


def _error(status: int, text: str, state: dict | None = None) -> JSONResponse:
    body = {"error": text}
    if state is not None:
        body["state"] = state
    return JSONResponse(body, status_code=status, headers=NO_STORE)


def _image(data: bytes | None, media: str, request: Request) -> Response:
    if data is None:
        return _error(404, "изображения нет")
    headers = IMMUTABLE if "v" in request.query_params else NO_STORE
    return Response(data, media_type=media, headers=headers)


def create_app(workbench: Workbench, *, token: str, port: int | None) -> FastAPI:
    app = FastAPI(title="Оцифровка фасада", docs_url=None, redoc_url=None,
                  openapi_url=None)
    wb = workbench

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if not _host_ok(request.headers.get("host", ""), port):
            return _error(421, "сервер принимает запросы только с этого компьютера")
        if request.method not in ("GET", "HEAD") and \
                not secrets.compare_digest(request.headers.get("x-facade-token", ""), token):
            return _error(403, "нет токена страницы: обновите страницу")
        return await call_next(request)

    def state_response(extra: dict | None = None) -> JSONResponse:
        return JSONResponse({"state": wb.state(), **(extra or {})}, headers=NO_STORE)

    # --- страница -------------------------------------------------------------------

    @app.get("/")
    def index():
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        return Response(html.replace("{{TOKEN}}", token), media_type="text/html",
                        headers=NO_STORE)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/ping")
    def ping():
        return {"app": "facade-digitizer", "data_dir": str(wb.data_dir)}

    # --- библиотека -----------------------------------------------------------------

    @app.get("/api/library")
    def library():
        return JSONResponse(wb.library(), headers=NO_STORE)

    @app.get("/api/thumb")
    def thumb(path: str):
        try:
            return Response(wb.thumbnail(path), media_type="image/jpeg", headers=NO_STORE)
        except FileNotFoundError:
            return _error(404, f"снимок не найден: {path}")
        except ValueError as error:
            return _error(400, str(error))

    @app.post("/api/upload")
    def upload(file: Annotated[UploadFile, File()]):
        try:
            return {"path": wb.upload_image(file.filename or "снимок", file.file.read())}
        except ValueError as error:
            return _error(400, str(error))

    @app.post("/api/profile/upload")
    def upload_profile(file: Annotated[UploadFile, File()]):
        try:
            return {"path": wb.upload_profile(file.filename or "профиль.json",
                                              file.file.read())}
        except ValueError as error:
            return _error(400, str(error))

    # --- рабочий стол ---------------------------------------------------------------

    @app.get("/api/state")
    def state():
        return state_response()

    @app.post("/api/open")
    async def open_image(request: Request):
        body = await request.json()
        try:
            wb.open(str(body.get("path", "")))
        except FileNotFoundError as error:
            return _error(404, str(error))
        except ValueError as error:
            return _error(400, str(error))
        return state_response()

    @app.post("/api/operator")
    async def answer_operator(request: Request):
        body = await request.json()
        try:
            wb.answer_operator(str(body.get("choice", "")))
        except ValueError as error:
            return _error(400, str(error))
        return state_response()

    @app.post("/api/close")
    def close():
        wb.close()
        return state_response()

    @app.post("/api/restart")
    def restart():
        try:
            wb.restart()
        except ValueError as error:
            return _error(400, str(error))
        return state_response()

    @app.post("/api/session/copy")
    async def save_copy(request: Request):
        body = await request.json()
        try:
            path = wb.save_copy(body.get("folder") or None)
        except (ValueError, OSError) as error:
            return _error(400, str(error))
        return state_response({"path": path})

    @app.get("/api/session/download")
    def session_download():
        try:
            name, data = wb.session_bytes()
        except ValueError as error:
            return _error(404, str(error))
        return Response(data, media_type="application/json", headers={
            **NO_STORE, "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})

    @app.post("/api/action")
    async def action(request: Request):
        body = await request.json()
        try:
            wb.act(body.get("desk"), int(body.get("rev", -1)), str(body.get("name", "")),
                   body.get("args") or {})
        except Stale as error:
            return _error(409, str(error), wb.state())
        except DeskBusy as error:
            return _error(409, str(error), wb.state())
        except (KeyError, TypeError) as error:
            return _error(400, f"недопустимое действие: {error}", wb.state())
        return state_response()

    # --- изображения и файлы --------------------------------------------------------

    def desk_bytes(attr: str):
        with wb.lock:
            return getattr(wb.desk, attr, None) if wb.desk is not None else None

    @app.get("/api/frame.png")
    def frame_png(request: Request):
        return _image(desk_bytes("frame_png"), "image/png", request)

    @app.get("/api/usable.png")
    def usable_png(request: Request):
        return _image(desk_bytes("usable_png"), "image/png", request)

    @app.get("/api/rectified.jpg")
    def rectified_jpg(request: Request):
        return _image(desk_bytes("rect_jpg"), "image/jpeg", request)

    @app.get("/api/rectified_mask.png")
    def rectified_mask(request: Request):
        return _image(desk_bytes("rect_mask_png"), "image/png", request)

    @app.get("/api/export/{name}")
    def export_file(name: str):
        try:
            path = wb.export_file(name)
        except FileNotFoundError:
            return _error(404, f"файла нет: {name}")
        except ValueError as error:
            return _error(400, str(error))
        return FileResponse(path, filename=path.name, headers=NO_STORE)

    return app
