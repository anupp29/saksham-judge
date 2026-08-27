FROM python:3.10-slim

RUN useradd -m -u 1000 appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

COPY --chown=appuser:appuser . .

RUN python warmstart.py

USER appuser

ENV PORT=10000
ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 10000

CMD ["python", "app.py"]    