"""
export_onnx.py
--------------
Exports best_boxing_model.pth → best_boxing_model.onnx
ONNX inference is typically 2–3× faster than PyTorch on CPU.

Usage:
    pip install onnx onnxruntime
    python export_onnx.py

After exporting, replace the torch inference block in inference.py
with the OnnxEngine class below.
"""

from pathlib import Path

import numpy as np
import torch

from feature_extractor import FEATURE_DIM, SEQUENCE_LENGTH
from model import load_model

ROOT = Path(__file__).parent
PT_PATH = ROOT / "best_boxing_model.pth"
ONNX_PATH = ROOT / "best_boxing_model.onnx"


def export():
    device = torch.device("cpu")
    model = load_model(str(PT_PATH), device)
    model.eval()

    dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)

    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_PATH),
        opset_version=17,
        input_names=["sequence"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "sequence": {0: "batch"},
            "logits":   {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"Exported → {ONNX_PATH}  ({ONNX_PATH.stat().st_size // 1024} KB)")


# ── Drop-in ONNX inference engine ─────────────────────────────────────────
class OnnxBoxingEngine:
    """
    Replace BoxingInferenceEngine.lstm with this for ~2-3× faster CPU inference.

    from export_onnx import OnnxBoxingEngine
    ...
    logits = self.onnx_engine.run(seq)   # (1, 8) numpy array
    """

    def __init__(self, onnx_path: str = str(ONNX_PATH)):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2          # match HF free-tier vCPUs
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(self, seq: np.ndarray) -> np.ndarray:
        """seq: (1, seq_len, feat_dim) float32 → logits (1, 8) float32"""
        outputs = self.session.run(None, {self.input_name: seq})
        return outputs[0]


if __name__ == "__main__":
    export()

    # Quick sanity check
    engine = OnnxBoxingEngine()
    dummy = np.random.rand(1, SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)
    logits = engine.run(dummy)
    print("ONNX output shape:", logits.shape)
    print("Logits:", logits)        