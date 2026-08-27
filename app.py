"""
Boxing Judge — FastAPI backend
No Gradio. No HuggingFace. No dependency hell.

Endpoints:
  GET  /           → serves index.html (MediaPipe webcam UI)
  GET  /ping       → keep-alive
  POST /predict    → {keypoints: [[x,y,conf]×17], bbox: [x1,y1,x2,y2], img_w, img_h}
                     → {label, confidence, probabilities}

FIXES (2026-08-27):
  - /predict no longer pre-normalises kp/bbox before passing to buf.push().
    FeatureBuffer.push() now expects RAW PIXEL coords and handles normalisation
    plus pixel-scale motion metric computation internally.
  - Added "Idle" label pass-through to frontend.
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
    CLASS_NAMES, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD,
    FeatureBuffer,
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
    keypoints: list[list[float]]   # 17 × [x_px, y_px, conf]  — PIXEL coords
    bbox:      list[float]         # [x1_px, y1_px, x2_px, y2_px] — PIXEL coords
    img_w:     int
    img_h:     int

# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"status": "ok", "uptime_s": round(time.time() - _START, 1)}

@app.post("/predict")
def predict(req: PredictRequest):
    kp_raw_px   = np.array(req.keypoints, dtype=np.float32)   # (17, 3) PIXEL
    bbox_raw_px = np.array(req.bbox,      dtype=np.float32)   # (4,)  PIXEL

    # Pass RAW PIXEL arrays — FeatureBuffer handles normalisation internally
    # so motion metrics are computed in pixel space (matching training scale).
    seq = _engine.buf.push(kp_raw_px, bbox_raw_px, req.img_w, req.img_h)

    if seq is None:
        needed = SEQUENCE_LENGTH - len(_engine.buf._buf)
        label  = f"collecting ({needed} more)" if needed > 0 else "idle"
        return {"label": label, "confidence": 0.0,
                "probabilities": {c: 0.0 for c in CLASS_NAMES}}

    x = torch.tensor(seq, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = _engine.lstm(x)
    probs    = torch.softmax(logits[0], dim=0).cpu().numpy()
    raw_best = int(np.argmax(probs))

    # Confidence gate — suppress random-class predictions when uncertain
    if float(probs[raw_best]) < CONFIDENCE_THRESHOLD:
        return {"label": "idle", "confidence": float(probs[raw_best]),
                "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}}

    # Temporal vote smoothing
    smoothed = _engine.buf.apply_vote(raw_best)

    return {
        "label":         CLASS_NAMES[smoothed],
        "confidence":    float(probs[smoothed]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
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