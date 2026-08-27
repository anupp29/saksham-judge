"""
Boxing Judge — FastAPI backend
Image-first pipeline: browser sends JPEG frames → YOLO pose → BiLSTM → label.
No MediaPipe. No JSON keypoints. Server owns the full inference stack.

Endpoints:
  GET  /           → index.html
  GET  /ping       → keep-alive
  POST /predict    → multipart: frame=<JPEG bytes>  → {label, confidence, probabilities, keypoints, bbox}
  POST /reset      → clear sequence buffer
"""

from __future__ import annotations
import os, time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import BoxingInferenceEngine

print("[app] Warming engine…")
_engine = BoxingInferenceEngine.get()
print("[app] Engine ready ✓")
_START = time.time()

app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "ok", "uptime_s": round(time.time() - _START, 1)}


@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    """
    Accepts a single JPEG/PNG frame as multipart file upload.
    Runs YOLO pose detection + BiLSTM classification server-side.
    Returns label, confidence, per-class probabilities, and keypoints for
    the frontend to draw the skeleton overlay.
    """
    raw = await frame.read()
    arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = _engine.predict(img_rgb)

    # Serialise keypoints for frontend skeleton drawing (None-safe)
    kp_out   = result["keypoints"].tolist()   if result["keypoints"]  is not None else None
    bbox_out = result["bbox"].tolist()        if result["bbox"]       is not None else None

    return {
        "label":         result["label"],
        "confidence":    result["confidence"],
        "probabilities": result["probabilities"],
        "fps":           result["fps"],
        "keypoints":     kp_out,
        "bbox":          bbox_out,
    }


@app.post("/reset")
def reset():
    _engine.reset()
    return {"status": "reset"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)