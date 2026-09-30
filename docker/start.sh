#!/bin/sh
# Запуск в контейнере: примеры при первом запуске, затем сервер интерфейса.
set -e

if ! touch /data/.probe 2>/dev/null; then
    echo "Папка /data недоступна для записи (на Linux её создал Docker от root)." >&2
    echo "Выполните на хосте: sudo chown -R 1000:1000 data   — и перезапустите." >&2
    exit 1
fi
rm -f /data/.probe
# Lock-файл прежнего запуска контейнера указывает на сервер, которого уже нет.
rm -f /data/.facade-ui.lock

# Примеры с известными размерами — один раз, при первом запуске на пустой папке.
if [ ! -d /data/примеры ]; then
    echo "Готовлю примеры снимков в /data/примеры …"
    python /opt/facade/scripts/make_acceptance_inputs.py --out-dir /data/примеры >/dev/null
fi

OPEN=""
if [ "${DEMO:-0}" = "1" ]; then
    echo "Готовлю демонстрацию в /data/демо …"
    python /opt/facade/scripts/demo_ui.py --out-dir /data/демо >/dev/null
    OPEN="--open демо/facade_demo.png"
fi

echo "Интерфейс оператора: откройте в браузере http://localhost:8765"
# Внутри контейнера сервер слушает все адреса, наружу порт опубликован только на
# 127.0.0.1 хоста (docker-compose.yml).
exec facade-digitize-ui --data-dir /data --listen-all --no-browser --port 8765 \
    ${OPERATOR:+--operator "$OPERATOR"} $OPEN
