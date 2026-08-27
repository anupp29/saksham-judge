"""
Boxing Judge — FastAPI backend
No Gradio. No HuggingFace. No dependency hell.

Endpoints:
  GET  /           → serves index.html (MediaPipe webcam UI)
  GET  /ping       → keep-alive
  POST /predict    → {keypoints: [[x,y,conf]×17], bbox: [x1,y1,x2,y2]} → {label, confidence, probabilities}
"""

from __future__ import annotations
import os, time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from feature_extractor import (
    CLASS_NAMES, SEQUENCE_LENGTH,
    FeatureBuffer, normalise_keypoints, normalise_bbox
)
from inference import BoxingInferenceEngine

# ── warm up at import time ─────────────────────────────────────────────────
print("[app] Warming engine…")
_engine = BoxingInferenceEngine.get()
print("[app] Engine ready ✓")
_START  = time.time()

app = FastAPI()

# ── request schema ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    keypoints: list[list[float]]   # 17 × [x, y, conf]  — pixel coords
    bbox:      list[float]         # [x1, y1, x2, y2]   — pixel coords
    img_w:     int
    img_h:     int

# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"status": "ok", "uptime_s": round(time.time() - _START, 1)}

@app.post("/predict")
def predict(req: PredictRequest):
    kp_raw   = np.array(req.keypoints, dtype=np.float32)   # (17, 3)
    bbox_raw = np.array(req.bbox,      dtype=np.float32)   # (4,)

    kp_norm   = normalise_keypoints(kp_raw,  req.img_w, req.img_h)
    bbox_norm = normalise_bbox(bbox_raw, req.img_w, req.img_h)

    seq = _engine.buf.push(kp_norm, bbox_norm)

    if seq is None:
        needed = SEQUENCE_LENGTH - len(_engine.buf._buf)
        return {"label": f"collecting ({needed} more)", "confidence": 0.0,
                "probabilities": {c: 0.0 for c in CLASS_NAMES}}

    x = torch.tensor(seq, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = _engine.lstm(x)
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    best  = int(np.argmax(probs))

    return {
        "label":         CLASS_NAMES[best],
        "confidence":    float(probs[best]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }

@app.post("/reset")
def reset():
    _engine.reset()
    return {"status": "reset"}

# serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)