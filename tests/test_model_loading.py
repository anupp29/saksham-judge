import numpy as np

from app.model import CLASS_NAMES, FEATURE_DIM, SEQUENCE_LENGTH, load_model_bundle


def test_real_checkpoint_loads_and_warms() -> None:
    bundle = load_model_bundle("best_boxing_model.pth")
    logits, attention, inference_ms = bundle.predict(
        np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    )
    assert tuple(logits.shape) == (1, len(CLASS_NAMES))
    assert tuple(attention.shape) == (1, SEQUENCE_LENGTH, 1)
    assert bundle.warmup_ms >= 0
    assert inference_ms >= 0
