# Образ для одноплатного компьютера (arm64) и обычного ПК (amd64)
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# libgl1/libglib2.0-0 нужны OpenCV; build-essential - для сборки lap на ARM, если нет готового пакета
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# пакеты, которые нужны только на Windows/разработке, в образ не ставим
RUN grep -v -i -E '^(colorama|nvidia-ml-py|pytest|iniconfig|pluggy)==' requirements.txt > requirements-docker.txt \
    && pip install -r requirements-docker.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY launcher.py .

# по умолчанию: детектор без окна; источник из переменной SOURCE
ENV HEADLESS=1 SOURCE=/dev/video0
CMD ["python", "launcher.py"]
