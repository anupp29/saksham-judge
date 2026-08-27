FROM python:3.10-slim

RUN useradd -m -u 1000 appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Bake YOLO weights into image at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

COPY --chown=appuser:appuser . .

# TorchScript warm-start cache
RUN python -c "
import torch
from model import load_model
from feature_extractor import SEQUENCE_LENGTH, FEATURE_DIM
model = load_model('best_boxing_model.pth', torch.device('cpu'))
model.eval()
traced = torch.jit.trace(model, (torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM),))
torch.jit.save(traced, 'boxing_traced.pt')
print('Warm-start cache written.')
"

USER appuser

# Render uses port 10000; Gradio will read this env var
ENV PORT=10000
ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 10000

CMD ["python", "app.py"]