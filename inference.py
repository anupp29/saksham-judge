"""
Inference engine — warm-start edition.

Boot order:
  1. Load YOLO (weights baked into Docker image at build time)
  2. Load BoxingLSTM from TorchScript cache (boxing_traced.pt)
     Falls back to loading from .pth if cache not found.
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

        kp_raw = bbox_raw = None

        if results and results[0].keypoints is not None:
            kps_all   = results[0].keypoints.data   # (N, 17, 3)
            boxes_all = results[0].boxes.xyxy        # (N, 4)
            if len(kps_all) > 0:
                areas   = (boxes_all[:, 2] - boxes_all[:, 0]) * (boxes_all[:, 3] - boxes_all[:, 1])
                best    = int(areas.argmax())
                kp_raw   = kps_all[best].cpu().numpy()
                bbox_raw = boxes_all[best].cpu().numpy()

        result = {
            "label":         "No person detected",
            "confidence":    0.0,
            "probabilities": {c: 0.0 for c in CLASS_NAMES},
            "fps":           0.0,
            "keypoints":     None,
            "bbox":          None,
        }

        if kp_raw is not None:
            result["keypoints"] = kp_raw
            result["bbox"]      = bbox_raw

            seq = self.buf.push(
                normalise_keypoints(kp_raw, w, h),
                normalise_bbox(bbox_raw, w, h),
            )

            if seq is None:
                needed = SEQUENCE_LENGTH - len(self.buf._buf)
                result["label"] = f"Collecting frames… ({needed} more)"
            else:
                x = torch.tensor(seq, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    logits, _ = self.lstm(x)
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                best  = int(np.argmax(probs))

                result["label"]         = CLASS_NAMES[best]
                result["confidence"]    = float(probs[best])
                result["probabilities"] = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        result["fps"] = 1.0 / (time.perf_counter() - t0 + 1e-9)
        return result

    def reset(self):
        self.buf.reset()