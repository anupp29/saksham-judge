"""
Feature extraction — mirrors training notebook exactly.

ROOT-CAUSE FIX (2026-08-27)
============================
The training pipeline (process_video_data → calculate_enhanced_motion_metrics)
works entirely in RAW PIXEL coordinates: YOLO .npy files store bbox [x1,y1,x2,y2]
and 17 keypoints [x_px, y_px, conf] in pixel space.  All 8 motion metrics
(arm extension, guard height, stance width, etc.) are therefore pixel-scale
values — typically in the 50–500 range for a 640×480 frame.

The original inference code called normalise_keypoints() → push(), so motion
metrics were computed on normalised (0–1) coordinates: ~640× smaller than
training.  This caused catastrophic train/serve feature skew — the model's
LSTM weights, tuned for pixel-scale motion magnitudes, received input an
order of magnitude outside its training distribution.  Result: logits are
near-uniform → softmax over 8 classes → random class (jab_right wins due
to minor weight bias) even when the subject is completely still.

FIX: FeatureBuffer.push() now receives raw PIXEL keypoints and bbox, computes
motion metrics in pixel space (matching training), then stores the normalised
kp+bbox together with the pixel-scale motion vector.  callers pass raw arrays.

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
FEATURE_DIM       = 63        # 51 kp (normalised) + 4 bbox (normalised) + 8 motion (pixel-scale)
KP_CONF_THRESHOLD = 0.4

# ── Motion gate ────────────────────────────────────────────────────────────
# Minimum wrist speed in PIXEL units/frame to trigger a prediction.
# Was 0.008 (normalised) = ~5px — correct concept but gating on a single frame
# was too noisy (MediaPipe jitter can spike 1 frame even when still).
# NEW: gate uses a rolling mean of the last GATE_WINDOW frames.
# 4.0 px/frame is well above still-jitter (1–3 px) and below a real punch start.
MOTION_GATE_PX   = 4.0   # pixels — gate threshold (pixel units, matching training scale)
GATE_WINDOW      = 6      # rolling-average window for wrist speed

# ── Prediction confidence guard ────────────────────────────────────────────
# Suppress output when the model is uncertain (uniform probs → ~0.125 each).
# A real punch should produce confident single-class output.
CONFIDENCE_THRESHOLD = 0.55

# ── Temporal smoothing ─────────────────────────────────────────────────────
# Majority-vote over recent predictions eliminates single-frame noise.
VOTE_WINDOW = 5   # frames

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
    Convert MediaPipe 33-landmark list to COCO-17 (17,3) array in PIXEL coords.
    mp_landmarks: list of objects with .x .y .visibility (normalised 0-1)
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    for coco_idx, mp_idx in enumerate(MP_TO_COCO_17):
        lm = mp_landmarks[mp_idx]
        kp[coco_idx] = [lm.x * img_w, lm.y * img_h, lm.visibility]
    return kp


def _valid(kp: np.ndarray, idx: int) -> bool:
    return bool(kp[idx, 2] > KP_CONF_THRESHOLD)


def _motion_metrics_px(cur_px: np.ndarray, prev_px: Optional[np.ndarray]) -> list[float]:
    """
    Compute the 8 boxing motion metrics in PIXEL coordinates.
    This matches the training notebook's calculate_enhanced_motion_metrics()
    which operates on raw YOLO pixel-coord keypoints.

    cur_px / prev_px : (17, 3) in pixel coords [x_px, y_px, conf]
    Returns list of 8 floats in pixel units (same scale as training).
    """
    m = [0.0] * 8
    if prev_px is None:
        return m
    try:
        # speed / distance — nose keypoint
        if _valid(cur_px, 0) and _valid(prev_px, 0):
            d = float(np.linalg.norm(cur_px[0, :2] - prev_px[0, :2]))
            m[0] = d * 50.0   # speed (distance × assumed 50 fps — matches notebook)
            m[1] = d          # raw distance
        # left arm extension: left shoulder (5) → left wrist (9)
        if _valid(cur_px, 5) and _valid(cur_px, 9):
            m[2] = float(np.linalg.norm(cur_px[9, :2] - cur_px[5, :2]))
        # right arm extension: right shoulder (6) → right wrist (10)
        if _valid(cur_px, 6) and _valid(cur_px, 10):
            m[3] = float(np.linalg.norm(cur_px[10, :2] - cur_px[6, :2]))
        # power index = max_extension × speed
        m[4] = max(m[2], m[3]) * m[0]
        # guard height: avg wrist y − nose y (positive = wrists above nose)
        if _valid(cur_px, 0) and _valid(cur_px, 9) and _valid(cur_px, 10):
            m[5] = float((cur_px[9, 1] + cur_px[10, 1]) / 2.0 - cur_px[0, 1])
        # body alignment: cos-similarity of shoulder vector and hip vector
        if all(_valid(cur_px, i) for i in [5, 6, 11, 12]):
            sv = cur_px[6, :2] - cur_px[5, :2]
            hv = cur_px[12, :2] - cur_px[11, :2]
            n = np.linalg.norm(sv) * np.linalg.norm(hv)
            if n > 1e-8:
                m[6] = float(abs(np.dot(sv, hv) / n))
        # stance width: left ankle (15) → right ankle (16)
        if _valid(cur_px, 15) and _valid(cur_px, 16):
            m[7] = float(np.linalg.norm(cur_px[16, :2] - cur_px[15, :2]))
    except Exception:
        pass
    return m


def _wrist_speed_px(cur_px: np.ndarray, prev_px: Optional[np.ndarray]) -> float:
    """Return max(left_wrist_speed, right_wrist_speed) in pixel units/frame."""
    if prev_px is None:
        return 0.0
    lw = float(np.linalg.norm(cur_px[9, :2] - prev_px[9, :2]))
    rw = float(np.linalg.norm(cur_px[10, :2] - prev_px[10, :2]))
    return max(lw, rw)


class FeatureBuffer:
    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len      = seq_len
        self._buf         : deque[list[float]] = deque(maxlen=seq_len)
        self._prev_kp_px  : Optional[np.ndarray] = None   # pixel-coord kp history
        # Rolling wrist-speed buffer for stable gate
        self._spd_win     : deque[float] = deque(maxlen=GATE_WINDOW)
        # Temporal vote buffer for prediction smoothing
        self._vote_buf    : deque[int] = deque(maxlen=VOTE_WINDOW)

    def reset(self):
        self._buf.clear()
        self._prev_kp_px  = None
        self._spd_win.clear()
        self._vote_buf.clear()

    def push(self, kp_raw_px: np.ndarray, bbox_raw_px: np.ndarray,
             img_w: int, img_h: int) -> Optional[np.ndarray]:
        """
        kp_raw_px  : (17, 3) in PIXEL coords   [x_px, y_px, conf]
        bbox_raw_px: (4,)    in PIXEL coords   [x1_px, y1_px, x2_px, y2_px]
        img_w, img_h: frame dimensions for normalisation

        Computes motion metrics in PIXEL space (matching training),
        then normalises kp and bbox for storage in the sequence buffer.

        Returns (1, seq_len, 63) only when buffer is full AND motion gate passes.
        """
        # ── 1. Compute pixel-scale motion metrics FIRST (matches training) ──
        motion = _motion_metrics_px(kp_raw_px, self._prev_kp_px)

        # ── 2. Update rolling wrist-speed for gate (pixel units) ──────────
        spd = _wrist_speed_px(kp_raw_px, self._prev_kp_px)
        self._spd_win.append(spd)
        rolling_spd = float(np.mean(self._spd_win))

        # ── 3. Store raw kp for next frame's motion diff ───────────────────
        self._prev_kp_px = kp_raw_px.copy()

        # ── 4. Normalise kp and bbox for storage ──────────────────────────
        kp_norm   = normalise_keypoints(kp_raw_px, img_w, img_h)
        bbox_norm = normalise_bbox(bbox_raw_px, img_w, img_h)

        # ── 5. Build 63-dim feature vector ────────────────────────────────
        feat = np.concatenate([kp_norm, bbox_norm, motion], dtype=np.float32)
        self._buf.append(feat.tolist())

        if len(self._buf) < self.seq_len:
            return None

        # ── 6. MOTION GATE (rolling average, pixel units) ─────────────────
        if rolling_spd < MOTION_GATE_PX:
            return None

        return np.array(list(self._buf), dtype=np.float32)[np.newaxis]

    def apply_vote(self, raw_pred: int) -> int:
        """
        Temporal majority-vote smoothing.
        Accumulates raw per-frame predictions and returns the class
        that appeared most often in the last VOTE_WINDOW frames.
        Ties broken by class index (stable).
        """
        self._vote_buf.append(raw_pred)
        counts = np.bincount(list(self._vote_buf), minlength=len(CLASS_NAMES))
        return int(np.argmax(counts))


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