"""Точка входа: `facade-digitize-ui` либо `python -m facade_digitizer.web`.

Запускает сервер на этом компьютере и открывает страницу в браузере. Рабочая папка
(`--data-dir`, по умолчанию `./data`) — снимки, профили, сессии и выгрузка.

`--operator op-xxxxxxxx` — непрозрачная метка оператора, назначенная исследованием
(план 3, задача 22); без неё метка создаётся случайной на запуск. Имя сюда не
вводится: в выходном файле оно было бы персональными данными (п. 14).

**Один сервер на рабочую папку.** Два сервера над одной папкой автосохраняли бы
одни и те же файлы сессий, и выигрывал бы последний записавший. Поэтому в папке —
`.facade-ui.lock` с портом; если по нему отвечает интерфейс той же папки, второй
запуск просто открывает браузер на первом.

**Только этот компьютер.** Сервер слушает loopback; `--listen-all` — только внутри
контейнера, где порт наружу публикуется на 127.0.0.1 хоста (`docker-compose.yml`).
"""
import argparse
import json
import os
import socket
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

LOCK_NAME = ".facade-ui.lock"
DEFAULT_PORT = 8765
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _parse(argv):
    p = argparse.ArgumentParser(prog="facade-digitize-ui",
                                description="Интерфейс оператора оцифровки фасада (в браузере).")
    p.add_argument("--data-dir", default=os.environ.get("FACADE_DATA_DIR", "data"),
                   help="рабочая папка: снимки, профили, сессии, выгрузка (по умолчанию ./data)")
    p.add_argument("--operator", default=os.environ.get("FACADE_OPERATOR") or None,
                   help="метка оператора вида op-1a2b3c4d (или FACADE_OPERATOR)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"порт (по умолчанию {DEFAULT_PORT}; занят — следующий свободный)")
    p.add_argument("--host", default="127.0.0.1", help="адрес: только этот компьютер")
    p.add_argument("--listen-all", action="store_true",
                   help="слушать все адреса — только внутри контейнера Docker")
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    p.add_argument("--open", default=None, help="сразу открыть снимок (путь в рабочей папке)")
    return p.parse_args(argv)


def free_port(host: str, start: int, attempts: int = 20) -> int:
    for port in range(start, start + attempts) if start else [0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host if host != "0.0.0.0" else "", port))
            except OSError:
                continue
            return s.getsockname()[1] if port == 0 else port
    raise SystemExit(f"нет свободного порта в {start}–{start + attempts - 1}")


def running_url(data_dir: Path) -> str | None:
    """Адрес уже запущенного интерфейса этой рабочей папки, если он отвечает."""
    lock = data_dir / LOCK_NAME
    try:
        port = int(json.loads(lock.read_text(encoding="utf-8"))["port"])
        url = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url + "api/ping", timeout=1.0) as r:
            ping = json.loads(r.read())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if ping.get("app") == "facade-digitizer" and \
            Path(ping.get("data_dir", "")).resolve() == data_dir:
        return url
    return None


def _serve(app, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(argv=None, *, serve=None, open_browser=webbrowser.open) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    host = "0.0.0.0" if args.listen_all else args.host
    if not args.listen_all and args.host not in LOOPBACK:
        print(f"--host {args.host}: сервер принимает запросы только с этого компьютера "
              "(127.0.0.1); вход по паролю и работа по сети не предусмотрены.", file=sys.stderr)
        return 2
    try:
        from facade_digitizer.web.app import create_app, new_token
        from facade_digitizer.web.workbench import Workbench
    except ImportError as error:
        print("Интерфейс оператора требует FastAPI и uvicorn: pip install -e \".[ui]\". "
              f"Причина: {error}", file=sys.stderr)
        return 2
    from facade_digitizer.ui.labour import checked_label

    if args.operator is not None:
        try:
            checked_label(args.operator)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    url = running_url(data_dir)
    if url is not None:
        print(f"Интерфейс этой рабочей папки уже запущен: {url}")
        if not args.no_browser:
            open_browser(url)
        return 0

    port = free_port(host if host != "0.0.0.0" else "127.0.0.1", args.port)
    workbench = Workbench(data_dir, operator=args.operator)
    if args.open:
        workbench.open(args.open)
    app = create_app(workbench, token=new_token(), port=port)
    lock = data_dir / LOCK_NAME
    lock.write_text(json.dumps({"pid": os.getpid(), "port": port, "data_dir": str(data_dir)}),
                    encoding="utf-8")
    url = f"http://127.0.0.1:{port}/"
    print(f"Интерфейс оператора: {url}   (рабочая папка {data_dir}; остановить — Ctrl+C)",
          flush=True)
    if not args.no_browser:
        threading.Timer(1.0, open_browser, [url]).start()
    try:
        (serve or _serve)(app, host, port)
    finally:
        workbench.runner.shutdown()
        try:
            if json.loads(lock.read_text(encoding="utf-8")).get("pid") == os.getpid():
                lock.unlink()
        except (OSError, ValueError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
