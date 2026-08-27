"""Exact PyTorch architecture and safe checkpoint loading for the notebook model."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


SEQUENCE_LENGTH = 16
FEATURE_DIM = 63
HIDDEN_SIZE = 128
NUM_LAYERS = 2  # Kept as metadata for compatibility with the training notebook.

# LabelEncoder sorts these strings lexicographically. This ordering is recovered
# from the notebook's classification report and must not be changed.
CLASS_NAMES = (
    "Blok lewą ręką",
    "Blok prawą ręką",
    "Chybienie lewą ręką",
    "Chybienie prawą ręką",
    "Głowa lewą ręką",
    "Głowa prawą ręką",
    "Korpus lewą ręką",
    "Korpus prawą ręką",
)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended, attention_weights


class BoxingLSTM(nn.Module):
    """The architecture used by cells 10 and 13 of the training notebook."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = len(CLASS_NAMES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # num_layers is retained in the constructor for checkpoint metadata; the
        # notebook implements two explicit one-layer BiLSTMs.
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_bn = nn.BatchNorm1d(input_size)
        self.input_dropout = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        x = self.input_dropout(x)
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        attended, attention_weights = self.attention(lstm2_out)
        out = self.fc1(attended)
        out = self.layer_norm1(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout1(out)
        return self.fc2(out), attention_weights


@dataclass(frozen=True)
class ModelBundle:
    model: BoxingLSTM
    device: torch.device
    checkpoint_sha256: str
    warmup_ms: float

    def predict(self, features: Any) -> tuple[torch.Tensor, torch.Tensor, float]:
        started = time.perf_counter()
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits, attention = self.model(tensor)
        return logits, attention, (time.perf_counter() - started) * 1000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_bundle(model_path: str | Path) -> ModelBundle:
    """Load and warm the state-dict checkpoint once per process."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoxingLSTM().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a PyTorch state_dict mapping")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch; missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        model(torch.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), device=device))
    return ModelBundle(model, device, _sha256(path), (time.perf_counter() - started) * 1000.0)

