#!/bin/sh
# Запуск в контейнере: виртуальный экран → оконный менеджер → VNC → noVNC → приложение.
set -e

Xvfb "$DISPLAY" -screen 0 "$RESOLUTION" -nolisten tcp >/tmp/xvfb.log 2>&1 &
for _ in $(seq 1 50); do [ -e "/tmp/.X11-unix/X${DISPLAY#:}" ] && break; sleep 0.1; done
openbox >/tmp/openbox.log 2>&1 &
x11vnc -display "$DISPLAY" -forever -shared -nopw -quiet -rfbport 5900 \
       -localhost >/tmp/x11vnc.log 2>&1 &
websockify --web /usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &

if ! touch /data/.probe 2>/dev/null; then
    echo "Папка /data недоступна для записи (на Linux её создал Docker от root)." >&2
    echo "Выполните на хосте: sudo chown -R 1000:1000 data   — и перезапустите." >&2
    exit 1
fi
rm -f /data/.probe

# Примеры с известными размерами — один раз, при первом запуске на пустой папке.
if [ ! -d /data/примеры ]; then
    echo "Готовлю примеры снимков в /data/примеры …"
    python3 /opt/facade/scripts/make_acceptance_inputs.py --out-dir /data/примеры >/dev/null
fi

echo "Интерфейс оператора: откройте в браузере http://localhost:6080"

if [ "${DEMO:-0}" = "1" ]; then
    python3 /opt/facade/scripts/demo_ui.py --show --out-dir /data/демо || true
fi

# Окно закрыли — открываем заново: контейнер живёт, пока его не остановят.
while true; do
    facade-digitize-ui --maximized ${OPERATOR:+--operator "$OPERATOR"} || true
    sleep 1
done
