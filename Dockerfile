# ─────────────────────────────────────────────────────────────────────────────
# Boxing Judge — Hugging Face Spaces Dockerfile
# CPU-only · free tier · warm-start · keep-alive endpoint
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# HF Spaces runs as uid 1000 — match it so file perms never bite you
RUN useradd -m -u 1000 appuser

# ── System deps (libGL for OpenCV) ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer — only rebuilds when requirements change) ───
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Pre-download YOLOv8-nano-pose weights at BUILD time (warm start) ──────
# This bakes the YOLO weights into the image so the container never
# has to download them at runtime — zero cold-start latency.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# ── Copy application code + model weights ────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Warm-start: load & JIT-trace the LSTM at build time ──────────────────
# The traced model is saved as boxing_traced.pt — inference.py loads it
# instantly with torch.jit.load() instead of re-initialising every boot.
RUN python -c "
import torch
from model import load_model
from feature_extractor import SEQUENCE_LENGTH, FEATURE_DIM

device = torch.device('cpu')
model = load_model('best_boxing_model.pth', device)
model.eval()

dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)
traced = torch.jit.trace(model, (dummy,))
torch.jit.save(traced, 'boxing_traced.pt')
print('TorchScript warm-start cache written → boxing_traced.pt')
"

# ── Switch to non-root user (HF requirement) ─────────────────────────────
USER appuser

# ── Expose Gradio's default port ─────────────────────────────────────────
EXPOSE 7860

# ── Health / keep-alive endpoint lives in app.py (see /ping route) ───────
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

CMD ["python", "app.py"]