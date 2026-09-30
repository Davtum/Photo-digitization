# Интерфейс оператора в контейнере: сервер на порту 8765, страница — в браузере хоста.
#
#   docker compose up --build        →  открыть http://localhost:8765
#
# Папка ./data хоста смонтирована в /data — это рабочая папка: снимки, профили,
# сессии и выгрузка JSON/DXF. Подробности — docs/docker.md.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenCV (пакет opencv-python) требует libGL и libglib даже без окна.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/facade
COPY pyproject.toml ./
COPY facade_digitizer ./facade_digitizer
RUN pip install ".[ui]"
COPY scripts ./scripts
COPY docs ./docs
COPY docker/start.sh /usr/local/bin/facade-start
# CRLF снимается на случай клона на Windows с autocrlf: иначе /bin/sh^M не найдётся.
RUN sed -i 's/\r$//' /usr/local/bin/facade-start && chmod +x /usr/local/bin/facade-start \
    && useradd --create-home --uid 1000 facade \
    && mkdir -p /data && chown facade /data

USER facade
WORKDIR /data
EXPOSE 8765
ENTRYPOINT ["facade-start"]
