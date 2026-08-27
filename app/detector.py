"""Lightweight YOLO pose adapter for raw webcam frames."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# Keep Ultralytics settings inside the service filesystem. This avoids relying
# on a writable home directory in restricted containers and on Render.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / ".ultralytics"))
from ultralytics import YOLO


DETECTION_DIM = 55


@dataclass(frozen=True)
class DetectionResult:
    detection: np.ndarray | None
    keypoints: np.ndarray | None
    bbox: np.ndarray | None
    width: int
    height: int
    detect_ms: float


class PoseDetector:
    """YOLOv8-nano-pose adapter selecting the largest detected person."""

    def __init__(self, model_path: str | Path, confidence: float = 0.35) -> None:
        requested_path = Path(model_path).expanduser()
        # Preserve a public model name such as ``yolov8n-pose.pt`` so
        # Ultralytics can resolve/download it during local setup. Render uses a
        # concrete file path produced by the build step.
        self.model_path = str(requested_path.resolve()) if requested_path.is_file() else str(model_path)
        self.confidence = confidence
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # If the file is not present, Ultralytics may download the named public
        # model. Render's build step downloads it explicitly for deterministic
        # startup; local development can use the same behavior.
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Warm detector kernels before readiness. The image is deliberately
        # small; real requests still use their native resolution.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(blank, conf=self.confidence, verbose=False, device=self.device)

    def detect(self, image_bytes: bytes, max_bytes: int = 5_000_000) -> DetectionResult:
        if not image_bytes or len(image_bytes) > max_bytes:
            raise ValueError(f"image must be non-empty and no larger than {max_bytes} bytes")
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode image; send JPEG, PNG, or WebP bytes")
        height, width = image.shape[:2]
        started = time.perf_counter()
        results = self.model(image, conf=self.confidence, verbose=False, device=self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return DetectionResult(None, None, None, width, height, elapsed_ms)

        keypoints = results[0].keypoints.data.detach().cpu().numpy()
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        if len(keypoints) == 0 or len(boxes) == 0:
            return DetectionResult(None, None, None, width, height, elapsed_ms)

        count = min(len(keypoints), len(boxes))
        areas = (boxes[:count, 2] - boxes[:count, 0]) * (boxes[:count, 3] - boxes[:count, 1])
        selected = int(np.argmax(areas))
        selected_keypoints = np.asarray(keypoints[selected], dtype=np.float32)
        selected_bbox = np.asarray(boxes[selected], dtype=np.float32)
        if selected_keypoints.shape != (17, 3) or selected_bbox.shape != (4,):
            raise RuntimeError("pose detector did not return COCO-17 keypoints and xyxy bbox")
        detection = np.concatenate((selected_bbox, selected_keypoints.reshape(-1))).astype(np.float32)
        return DetectionResult(detection, selected_keypoints, selected_bbox, width, height, elapsed_ms)
