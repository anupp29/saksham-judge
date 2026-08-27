"""
BoxingLSTM — exact architecture used during training.
input_size  = 63  (51 keypoint coords + 4 bbox + 8 motion metrics)
hidden_size = 128
num_classes = 8
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states):
        weights = self.attention(hidden_states)          # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        attended = torch.sum(weights * hidden_states, dim=1)  # (B, H)
        return attended, weights


class BoxingLSTM(nn.Module):
    """
    Bidirectional 2-layer LSTM with attention for boxing action classification.
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalisation
        self.input_bn = nn.BatchNorm1d(input_size)
        self.input_dropout = nn.Dropout(0.3)

        # Bi-LSTM layer 1  (input → hidden*2 because bidirectional)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Bi-LSTM layer 2
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention over time-steps
        self.attention = AttentionLayer(hidden_size * 2)

        # Classification head
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x : (batch, seq_len, input_size)
        returns logits (batch, num_classes) and attention weights
        """
        # BatchNorm expects (batch, features, seq) → transpose, norm, transpose back
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        x = self.input_dropout(x)

        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)

        attended, attn_weights = self.attention(lstm2_out)

        out = self.fc1(attended)
        out = self.layer_norm1(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout1(out)
        out = self.fc2(out)

        return out, attn_weights


def load_model(weights_path: str, device: torch.device) -> BoxingLSTM:
    model = BoxingLSTM(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model