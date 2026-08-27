"""Train-compatible feature construction and strict request validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


SEQUENCE_LENGTH = 16
FEATURE_DIM = 63
DETECTION_DIM = 55
MOTION_DIM = 8
KP_CONFIDENCE_THRESHOLD = 0.3


class InputContractError(ValueError):
    """An input cannot be represented in the model's training contract."""


def _array(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InputContractError(f"{name} must contain numeric values") from exc
    if result.shape != (expected,):
        raise InputContractError(f"{name} must contain exactly {expected} values")
    if not np.isfinite(result).all():
        raise InputContractError(f"{name} must contain only finite values")
    return result


def _valid(keypoints: np.ndarray, index: int) -> bool:
    return bool(keypoints[index, 2] > KP_CONFIDENCE_THRESHOLD)


def motion_metrics(current: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Mirror calculate_enhanced_motion_metrics from the training notebook."""
    result = np.zeros(MOTION_DIM, dtype=np.float32)
    if previous is None:
        return result
    cur = current[4:].reshape(17, 3)
    prev = previous[4:].reshape(17, 3)
    if _valid(cur, 0) and _valid(prev, 0):
        displacement = cur[0, :2] - prev[0, :2]
        distance = float(np.linalg.norm(displacement))
        result[0] = distance * 50.0
        result[1] = distance
    if _valid(cur, 5) and _valid(cur, 9):
        result[2] = np.linalg.norm(cur[9, :2] - cur[5, :2])
    if _valid(cur, 6) and _valid(cur, 10):
        result[3] = np.linalg.norm(cur[10, :2] - cur[6, :2])
    result[4] = max(float(result[2]), float(result[3])) * float(result[0])
    if _valid(cur, 0) and _valid(cur, 9) and _valid(cur, 10):
        result[5] = (cur[9, 1] + cur[10, 1]) / 2.0 - cur[0, 1]
    if all(_valid(cur, i) for i in (5, 6, 11, 12)):
        shoulder = cur[6, :2] - cur[5, :2]
        hip = cur[12, :2] - cur[11, :2]
        norms = np.linalg.norm(shoulder) * np.linalg.norm(hip)
        if not (np.all(shoulder == 0) or np.all(hip == 0)):
            result[6] = abs(float(np.dot(shoulder, hip) / (norms + 1e-8)))
    if _valid(cur, 15) and _valid(cur, 16):
        result[7] = np.linalg.norm(cur[16, :2] - cur[15, :2])
    return result


def _motion_from_payload(value: Mapping[str, Any]) -> np.ndarray:
    try:
        values = [
            value["movement"]["speed"], value["movement"]["distance"],
            value["boxing_specific"]["left_arm_extension"], value["boxing_specific"]["right_arm_extension"],
            value["boxing_specific"]["power_index"], value["boxing_specific"]["guard_height"],
            value["pose"]["body_alignment"], value["pose"]["stance_width"],
        ]
        return _array(values, MOTION_DIM, "motion_metrics")
    except (KeyError, TypeError) as exc:
        raise InputContractError("motion_metrics must include movement, boxing_specific, and pose fields") from exc


def _frame_detection(frame: Mapping[str, Any]) -> np.ndarray | None:
    if "detection" in frame:
        return _array(frame["detection"], DETECTION_DIM, "detection")
    if "keypoints" in frame or "bbox" in frame:
        if "keypoints" not in frame or "bbox" not in frame:
            raise InputContractError("keypoints and bbox must be supplied together")
        return np.concatenate((_array(frame["bbox"], 4, "bbox"), _array(frame["keypoints"], 51, "keypoints"))).astype(np.float32)
    return None


def frame_to_features(frame: Mapping[str, Any], previous_detection: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if "features" in frame:
        return _array(frame["features"], FEATURE_DIM, "features"), None
    detection = _frame_detection(frame)
    if detection is None:
        raise InputContractError("each frame needs either features (63), detection (55), or keypoints+bbox")
    metrics = _motion_from_payload(frame["motion_metrics"]) if frame.get("motion_metrics") is not None else motion_metrics(detection, previous_detection)
    # The notebook uses raw detector coordinates for keypoints and bbox.
    return np.concatenate((detection[4:], detection[:4], metrics)).astype(np.float32), detection


def build_feature_matrix(frames: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if len(frames) != SEQUENCE_LENGTH:
        raise InputContractError(f"exactly {SEQUENCE_LENGTH} frames are required; received {len(frames)}")
    output: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise InputContractError(f"frame {index} must be an object")
        features, detection = frame_to_features(frame, previous)
        output.append(features)
        previous = detection
    return np.stack(output).astype(np.float32)


def build_from_features(features: Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
        raise InputContractError(f"features must have shape [{SEQUENCE_LENGTH}, {FEATURE_DIM}]")
    if not np.isfinite(result).all():
        raise InputContractError("features must contain only finite values")
    return result

