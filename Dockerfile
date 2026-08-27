FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MODEL_PATH=/app/best_boxing_model.pth \
    POSE_MODEL_PATH=/app/yolov8n-pose.pt

WORKDIR /app

# OpenCV runtime libraries; no compiler toolchain is needed for the pinned wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY best_boxing_model.pth ./best_boxing_model.pth
COPY yolov8n-pose.pt ./yolov8n-pose.pt
COPY labels.json ./labels.json

RUN test -s /app/best_boxing_model.pth \
    && test -s /app/yolov8n-pose.pt

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
