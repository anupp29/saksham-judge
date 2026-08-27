import numpy as np
import pytest

from app.preprocessing import (
    DETECTION_DIM,
    FEATURE_DIM,
    InputContractError,
    SEQUENCE_LENGTH,
    build_feature_matrix,
    build_from_features,
    motion_metrics,
)


def detection(x_shift: float = 0.0) -> list[float]:
    values = [100.0, 80.0, 540.0, 460.0]
    for index in range(17):
        values.extend([320.0 + x_shift + index, 120.0 + index, 0.9])
    return values


def test_raw_detection_sequence_matches_training_shape() -> None:
    matrix = build_feature_matrix([{"detection": detection(i)} for i in range(SEQUENCE_LENGTH)])
    assert matrix.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert matrix.dtype == np.float32
    assert matrix[0, :4].tolist() == pytest.approx([320.0, 120.0, 0.9, 321.0])


def test_motion_metric_order_and_pixel_scale() -> None:
    first = np.asarray(detection(), dtype=np.float32)
    second = np.asarray(detection(10.0), dtype=np.float32)
    result = motion_metrics(second, first)
    assert result.shape == (8,)
    assert result[0] == pytest.approx(500.0)
    assert result[1] == pytest.approx(10.0)


def test_precomputed_features_are_strict() -> None:
    values = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    assert build_from_features(values.tolist()).shape == values.shape
    with pytest.raises(InputContractError):
        build_from_features(np.zeros((SEQUENCE_LENGTH - 1, FEATURE_DIM)).tolist())


def test_bad_detection_is_rejected() -> None:
    with pytest.raises(InputContractError, match="55"):
        build_feature_matrix([{"detection": [0.0] * (DETECTION_DIM - 1)}] * SEQUENCE_LENGTH)
