"""Точка входа `facade-digitize-ui`: один сервер на рабочую папку, только loopback. Задача 9."""
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from facade_digitizer.web import __main__ as entry


def _ping_server(data_dir):
    """Сервер, отвечающий на /api/ping так, как отвечает уже запущенный интерфейс."""
    body = json.dumps({"app": "facade-digitizer", "data_dir": str(data_dir)}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/api/ping" else 404)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_lock_reuses_running_server(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    server = _ping_server(data.resolve())
    port = server.server_address[1]
    (data / entry.LOCK_NAME).write_text(json.dumps({"pid": 1, "port": port}), encoding="utf-8")
    served, opened = [], []
    code = entry.main(["--data-dir", str(data)], serve=lambda *a: served.append(a),
                      open_browser=opened.append)
    server.shutdown()
    assert code == 0 and served == []
    assert opened == [f"http://127.0.0.1:{port}/"]


def test_stale_lock_is_taken_and_removed_after(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / entry.LOCK_NAME).write_text(json.dumps({"pid": 1, "port": 1}), encoding="utf-8")
    seen = {}

    def serve(app, host, port):
        seen["lock"] = json.loads((data / entry.LOCK_NAME).read_text(encoding="utf-8"))
        seen["host"], seen["port"] = host, port

    code = entry.main(["--data-dir", str(data), "--no-browser", "--port", "0"], serve=serve)
    assert code == 0
    assert seen["host"] == "127.0.0.1"
    assert seen["lock"] == {"pid": os.getpid(), "port": seen["port"],
                            "data_dir": str(data.resolve())}
    assert not (data / entry.LOCK_NAME).exists()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
def test_non_loopback_host_is_refused(tmp_path, host, capsys):
    code = entry.main(["--data-dir", str(tmp_path), "--host", host], serve=lambda *a: None)
    assert code == 2 and "только с этого компьютера" in capsys.readouterr().err


def test_container_may_listen_on_all_interfaces(tmp_path):
    seen = {}
    code = entry.main(["--data-dir", str(tmp_path), "--no-browser", "--listen-all",
                       "--port", "0"], serve=lambda app, host, port: seen.update(host=host))
    assert code == 0 and seen["host"] == "0.0.0.0"


def test_busy_port_moves_to_the_next(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert entry.free_port("127.0.0.1", port) != port


def test_bad_operator_label_is_refused(tmp_path, capsys):
    code = entry.main(["--data-dir", str(tmp_path), "--operator", "Иван"],
                      serve=lambda *a: None)
    assert code == 2 and "непрозрачна" in capsys.readouterr().err
