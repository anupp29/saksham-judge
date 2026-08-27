"""
Feature extraction — mirrors training notebook exactly.

COCO-17 keypoint order (what YOLO was trained on):
  0 Nose  1 LEye  2 REye  3 LEar  4 REar
  5 LShoulder  6 RShoulder  7 LElbow  8 RElbow
  9 LWrist  10 RWrist  11 LHip  12 RHip
  13 LKnee  14 RKnee  15 LAnkle  16 RAnkle

MediaPipe Pose returns 33 landmarks. Correct mapping to COCO-17:
  MP index → COCO index
  0→0, 2→1, 5→2, 7→3, 8→4,
  11→5, 12→6, 13→7, 14→8,
  15→9, 16→10, 23→11, 24→12,
  25→13, 26→14, 27→15, 28→16
"""

from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np

SEQUENCE_LENGTH   = 16
FEATURE_DIM       = 63        # 51 kp + 4 bbox + 8 motion
KP_CONF_THRESHOLD = 0.4

# Minimum wrist speed (normalised units/frame) to count as a punch motion.
# Below this → buffer keeps filling but prediction is suppressed → no gibberish.
MOTION_GATE = 0.008

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

# MediaPipe 33-point → COCO 17-point index map
MP_TO_COCO_17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def mp33_to_coco17(mp_landmarks: list, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert MediaPipe 33-landmark list to COCO-17 (17,3) array in pixel coords.
    mp_landmarks: list of objects with .x .y .visibility (normalised 0-1)
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    for coco_idx, mp_idx in enumerate(MP_TO_COCO_17):
        lm = mp_landmarks[mp_idx]
        kp[coco_idx] = [lm.x * img_w, lm.y * img_h, lm.visibility]
    return kp


def _valid(kp: np.ndarray, idx: int) -> bool:
    return bool(kp[idx, 2] > KP_CONF_THRESHOLD)


def _motion_metrics(cur: np.ndarray, prev: Optional[np.ndarray]) -> list[float]:
    m = [0.0] * 8
    if prev is None:
        return m
    try:
        # speed / distance (nose)
        if _valid(cur, 0) and _valid(prev, 0):
            d = float(np.linalg.norm(cur[0, :2] - prev[0, :2]))
            m[0] = d * 50.0
            m[1] = d
        # arm extensions
        if _valid(cur, 5) and _valid(cur, 9):
            m[2] = float(np.linalg.norm(cur[9, :2] - cur[5, :2]))
        if _valid(cur, 6) and _valid(cur, 10):
            m[3] = float(np.linalg.norm(cur[10, :2] - cur[6, :2]))
        m[4] = max(m[2], m[3]) * m[0]
        # guard height
        if _valid(cur, 0) and _valid(cur, 9) and _valid(cur, 10):
            m[5] = float((cur[9,1] + cur[10,1]) / 2.0 - cur[0,1])
        # body alignment
        if all(_valid(cur, i) for i in [5,6,11,12]):
            sv = cur[6,:2] - cur[5,:2]
            hv = cur[12,:2] - cur[11,:2]
            n = np.linalg.norm(sv) * np.linalg.norm(hv)
            if n > 1e-8:
                m[6] = float(abs(np.dot(sv, hv) / n))
        # stance width
        if _valid(cur, 15) and _valid(cur, 16):
            m[7] = float(np.linalg.norm(cur[16,:2] - cur[15,:2]))
    except Exception:
        pass
    return m


class FeatureBuffer:
    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len   = seq_len
        self._buf      : deque[list[float]] = deque(maxlen=seq_len)
        self._prev_kp  : Optional[np.ndarray] = None
        self._wrist_spd: float = 0.0   # tracked for motion gate

    def reset(self):
        self._buf.clear()
        self._prev_kp  = None
        self._wrist_spd = 0.0

    def push(self, kp_flat: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        kp_flat : (51,) normalised
        bbox    : (4,)  normalised
        Returns (1, seq_len, 63) only when buffer full AND motion detected.
        """
        kp = kp_flat.reshape(17, 3)
        motion = _motion_metrics(kp, self._prev_kp)

        # track wrist speed for gate (wrist coords are normalised → small values)
        if self._prev_kp is not None:
            lw = float(np.linalg.norm(kp[9,:2]  - self._prev_kp[9,:2]))
            rw = float(np.linalg.norm(kp[10,:2] - self._prev_kp[10,:2]))
            self._wrist_spd = max(lw, rw)
        else:
            self._wrist_spd = 0.0

        self._prev_kp = kp.copy()

        feat = np.concatenate([kp_flat, bbox, motion], dtype=np.float32)
        self._buf.append(feat.tolist())

        if len(self._buf) < self.seq_len:
            return None

        # ── MOTION GATE ───────────────────────────────────────────────────
        # If wrists haven't moved enough → person is still → suppress output
        if self._wrist_spd < MOTION_GATE:
            return None

        return np.array(list(self._buf), dtype=np.float32)[np.newaxis]


def normalise_keypoints(kp_xyc: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    kp = kp_xyc.copy().astype(np.float32)
    kp[:, 0] /= max(img_w, 1)
    kp[:, 1] /= max(img_h, 1)
    return kp.flatten()


def normalise_bbox(bbox_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    b = bbox_xyxy.copy().astype(np.float32)
    b[0] /= max(img_w, 1); b[1] /= max(img_h, 1)
    b[2] /= max(img_w, 1); b[3] /= max(img_h, 1)
    return b