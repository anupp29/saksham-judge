"""
Inference engine — warm-start edition.

Boot order:
  1. Load YOLO (weights baked into Docker image at build time)
  2. Load BoxingLSTM from TorchScript cache (boxing_traced.pt)
     Falls back to loading from .pth if cache not found.

FIXES (2026-08-27):
  - predict() now passes raw PIXEL kp+bbox to FeatureBuffer.push()
    (previously passed pre-normalised arrays → motion metric skew)
  - Applies confidence gate: predictions below CONFIDENCE_THRESHOLD
    return "idle" instead of a random class
  - Applies temporal vote smoothing via FeatureBuffer.apply_vote()
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from feature_extractor import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    FeatureBuffer,
    normalise_bbox,
    normalise_keypoints,
)
from model import load_model

ROOT          = Path(__file__).parent
WEIGHTS_PATH  = ROOT / "best_boxing_model.pth"
TRACED_PATH   = ROOT / "boxing_traced.pt"       # TorchScript warm-start cache
YOLO_MODEL    = "yolov8n-pose.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_lstm() -> torch.nn.Module:
    """Load TorchScript cache if available (fast), else fall back to .pth."""
    if TRACED_PATH.exists():
        print("[engine] Loading TorchScript cache (warm start)…")
        model = torch.jit.load(str(TRACED_PATH), map_location=DEVICE)
        model.eval()
        return model
    print("[engine] TorchScript cache not found — loading from .pth…")
    return load_model(str(WEIGHTS_PATH), DEVICE)


class BoxingInferenceEngine:
    """Thread-safe, stateful inference engine (singleton)."""

    _instance: Optional["BoxingInferenceEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        print(f"[engine] Initialising on {DEVICE}…")
        t0 = time.perf_counter()

        self.yolo = YOLO(YOLO_MODEL)
        self.yolo.to(DEVICE)

        self.lstm = _load_lstm()

        # Dummy forward pass → kernels compiled, memory allocated, truly warm
        with torch.no_grad():
            dummy = torch.zeros(1, SEQUENCE_LENGTH, FEATURE_DIM, device=DEVICE)
            self.lstm(dummy)

        self.buf = FeatureBuffer(seq_len=SEQUENCE_LENGTH)
        print(f"[engine] Ready in {time.perf_counter() - t0:.2f}s")

    @classmethod
    def get(cls) -> "BoxingInferenceEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def predict(self, frame_rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = frame_rgb.shape[:2]

        with torch.no_grad():
            results = self.yolo(
                frame_rgb,
                verbose=False,
                device=DEVICE,
                half=(DEVICE.type == "cuda"),
            )

        kp_raw_px = bbox_raw_px = None

        if results and results[0].keypoints is not None:
            kps_all   = results[0].keypoints.data   # (N, 17, 3)
            boxes_all = results[0].boxes.xyxy        # (N, 4)
            if len(kps_all) > 0:
                areas     = (boxes_all[:, 2] - boxes_all[:, 0]) * (boxes_all[:, 3] - boxes_all[:, 1])
                best      = int(areas.argmax())
                kp_raw_px   = kps_all[best].cpu().numpy()   # PIXEL coords (x, y, conf)
                bbox_raw_px = boxes_all[best].cpu().numpy()  # PIXEL coords [x1,y1,x2,y2]

        result = {
            "label":         "No person detected",
            "confidence":    0.0,
            "probabilities": {c: 0.0 for c in CLASS_NAMES},
            "fps":           0.0,
            "keypoints":     None,
            "bbox":          None,
        }

        if kp_raw_px is not None:
            result["keypoints"] = kp_raw_px
            result["bbox"]      = bbox_raw_px

            # Pass RAW PIXEL arrays — FeatureBuffer handles normalisation internally
            # and computes motion metrics in pixel space (matching training scale).
            seq = self.buf.push(kp_raw_px, bbox_raw_px, w, h)

            if seq is None:
                needed = SEQUENCE_LENGTH - len(self.buf._buf)
                if needed > 0:
                    result["label"] = f"Collecting frames… ({needed} more)"
                else:
                    result["label"] = "Idle"
            else:
                x = torch.tensor(seq, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    logits, _ = self.lstm(x)
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                raw_best = int(np.argmax(probs))

                # ── Confidence gate ────────────────────────────────────────
                # If model is uncertain (probs near-uniform ≈ 1/8 = 0.125),
                # suppress prediction → "Idle". Avoids random-class spam.
                if float(probs[raw_best]) < CONFIDENCE_THRESHOLD:
                    result["label"]      = "Idle"
                    result["confidence"] = float(probs[raw_best])
                    result["probabilities"] = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
                else:
                    # ── Temporal vote smoothing ────────────────────────────
                    smoothed = self.buf.apply_vote(raw_best)
                    result["label"]         = CLASS_NAMES[smoothed]
                    result["confidence"]    = float(probs[smoothed])
                    result["probabilities"] = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        result["fps"] = 1.0 / (time.perf_counter() - t0 + 1e-9)
        return result

    def reset(self):
        self.buf.reset()