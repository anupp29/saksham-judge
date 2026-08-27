"""
Feature extraction pipeline — mirrors the training notebook exactly.

YOLO keypoint index map (17 points, each with x, y, conf → 51 values):
  0 Nose   1 LeftEye  2 RightEye  3 LeftEar   4 RightEar
  5 LShoulder 6 RShoulder 7 LElbow 8 RElbow
  9 LWrist  10 RWrist  11 LHip  12 RHip  13 LKnee  14 RKnee
  15 LAnkle  16 RAnkle

Per-frame feature vector (63 dims):
  keypoints.flatten()   → 51  (17 × 3)
  bounding_box          →  4
  motion_metrics        →  8  (speed, distance, l_ext, r_ext, power, guard, align, stance)
"""

from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np

SEQUENCE_LENGTH = 16          # must match training
FEATURE_DIM = 63             # 51 + 4 + 8
KP_CONF_THRESHOLD = 0.3

# Boxing action class names — order must match label_encoder.classes_ used at training.
# Edit this list only if you know the exact class order from your training run.
CLASS_NAMES = [
    "body_hook_left",
    "body_hook_right",
    "hook_left",
    "hook_right",
    "jab_left",
    "jab_right",
    "uppercut_left",
    "uppercut_right",
]


def _valid(kp: np.ndarray, idx: int) -> bool:
    """True when the keypoint at index `idx` has confidence above threshold."""
    return kp[idx, 2] > KP_CONF_THRESHOLD


def _calculate_motion_metrics(
    current_kp: Optional[np.ndarray],  # shape (17, 3)
    prev_kp: Optional[np.ndarray],
) -> list[float]:
    """
    Returns [speed, distance, l_arm_ext, r_arm_ext, power_index,
             guard_height, body_alignment, stance_width]
    """
    metrics = [0.0] * 8

    if current_kp is None or prev_kp is None:
        return metrics

    try:
        # -- movement
        if _valid(current_kp, 0) and _valid(prev_kp, 0):
            disp = current_kp[0, :2] - prev_kp[0, :2]
            dist = float(np.linalg.norm(disp))
            metrics[0] = dist * 50.0   # speed  (assuming ~50 fps)
            metrics[1] = dist          # distance

        # -- left arm extension (shoulder 5 → wrist 9)
        if _valid(current_kp, 5) and _valid(current_kp, 9):
            metrics[2] = float(np.linalg.norm(current_kp[9, :2] - current_kp[5, :2]))

        # -- right arm extension (shoulder 6 → wrist 10)
        if _valid(current_kp, 6) and _valid(current_kp, 10):
            metrics[3] = float(np.linalg.norm(current_kp[10, :2] - current_kp[6, :2]))

        # -- power index
        metrics[4] = max(metrics[2], metrics[3]) * metrics[0]

        # -- guard height (wrist avg height relative to nose)
        if _valid(current_kp, 0) and _valid(current_kp, 9) and _valid(current_kp, 10):
            wrist_y = (current_kp[9, 1] + current_kp[10, 1]) / 2.0
            metrics[5] = float(wrist_y - current_kp[0, 1])

        # -- body alignment (dot product of shoulder & hip vectors, normalised)
        if all(_valid(current_kp, i) for i in [5, 6, 11, 12]):
            sv = current_kp[6, :2] - current_kp[5, :2]
            hv = current_kp[12, :2] - current_kp[11, :2]
            norms = np.linalg.norm(sv) * np.linalg.norm(hv)
            if norms > 1e-8:
                metrics[6] = float(abs(np.dot(sv, hv) / norms))

        # -- stance width (ankle distance)
        if _valid(current_kp, 15) and _valid(current_kp, 16):
            metrics[7] = float(np.linalg.norm(current_kp[16, :2] - current_kp[15, :2]))

    except Exception:
        pass

    return metrics


class FeatureBuffer:
    """
    Maintains a sliding window of feature vectors and builds
    (1, SEQUENCE_LENGTH, FEATURE_DIM) tensors ready for the LSTM.
    """

    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len = seq_len
        self._buf: deque[list[float]] = deque(maxlen=seq_len)
        self._prev_kp: Optional[np.ndarray] = None

    def reset(self):
        self._buf.clear()
        self._prev_kp = None

    def push(self, kp_flat: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        kp_flat : (51,)  — raw YOLO keypoints flattened (x,y,conf × 17)
        bbox    : (4,)   — [x1, y1, x2, y2] in pixel coords (normalise if needed)
        Returns (1, seq_len, 63) float32 array when the buffer is full, else None.
        """
        # Reshape for metric calculation
        kp = kp_flat.reshape(17, 3)

        motion = _calculate_motion_metrics(kp, self._prev_kp)
        self._prev_kp = kp.copy()

        feature = np.concatenate([kp_flat, bbox, motion], dtype=np.float32)
        assert feature.shape == (FEATURE_DIM,), f"Bad feature dim {feature.shape}"

        self._buf.append(feature.tolist())

        if len(self._buf) == self.seq_len:
            seq = np.array(list(self._buf), dtype=np.float32)  # (seq, feat)
            return seq[np.newaxis]   # (1, seq, feat)

        return None


def normalise_keypoints(kp_xyc: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Divide x coords by img_w and y coords by img_h so keypoints are in [0,1].
    kp_xyc: (17, 3) array with x, y in pixel space.
    Returns (51,) flattened normalised array.
    """
    kp = kp_xyc.copy().astype(np.float32)
    kp[:, 0] /= max(img_w, 1)
    kp[:, 1] /= max(img_h, 1)
    return kp.flatten()


def normalise_bbox(bbox_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Normalise [x1, y1, x2, y2] to [0,1] range.
    """
    b = bbox_xyxy.copy().astype(np.float32)
    b[0] /= max(img_w, 1)
    b[1] /= max(img_h, 1)
    b[2] /= max(img_w, 1)
    b[3] /= max(img_h, 1)
    return b