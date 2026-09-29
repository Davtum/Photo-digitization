# Интерфейс оператора в контейнере с доступом через браузер (noVNC).
#
#   docker compose up --build        →  открыть http://localhost:6080
#
# Окно Qt работает на виртуальном экране (Xvfb) внутри контейнера и показывается в
# браузере; папка ./data хоста смонтирована в /data — туда кладутся снимки и туда же
# пишутся экспорт JSON/DXF и файлы сессии. Подробности — docs/docker.md.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

# Виртуальный экран, VNC, noVNC, оконный менеджер и системные библиотеки Qt (xcb).
# Шрифты DejaVu — кириллица в интерфейсе.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates \
        xvfb x11vnc novnc websockify openbox \
        libegl1 libgl1 libglib2.0-0 libdbus-1-3 libfontconfig1 libfreetype6 \
        libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
        libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
        libxcb-xinerama0 libxcb-xfixes0 libxcb-xkb1 libx11-xcb1 libxrender1 libxi6 \
        libsm6 libice6 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/facade
COPY pyproject.toml ./
COPY facade_digitizer ./facade_digitizer
RUN python3 -m venv /opt/venv && pip install --upgrade pip && pip install ".[ui]"
COPY scripts ./scripts
COPY docs ./docs
COPY docker/start.sh /usr/local/bin/facade-start
# CRLF снимается на случай клона на Windows с autocrlf: иначе /bin/sh^M не найдётся.
RUN sed -i 's/\r$//' /usr/local/bin/facade-start && chmod +x /usr/local/bin/facade-start \
    && printf '<meta http-equiv="refresh" content="0; url=vnc.html?autoconnect=1&resize=scale">\n' \
       > /usr/share/novnc/index.html \
    && (userdel -r ubuntu 2>/dev/null || true) \
    && useradd --create-home --uid 1000 facade \
    && mkdir -p /data && chown facade /data

USER facade
ENV DISPLAY=:1 \
    QT_QPA_PLATFORM=xcb \
    RESOLUTION=1600x950x24
WORKDIR /data
EXPOSE 6080
ENTRYPOINT ["facade-start"]
