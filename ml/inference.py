from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, h: torch.Tensor):
        weights = torch.softmax(self.score(h).squeeze(-1), dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return context, weights


class TemporalClassifier(nn.Module):
    """Architecture used by the exported VIGIL arrhythmia checkpoint."""

    def __init__(self, beat_len: int, kind: str, hidden: int = 48, layers: int = 1,
                 dropout: float = 0.25, n_classes: int = 5):
        super().__init__()
        self.kind = kind
        self.encoder = nn.Sequential(
            nn.Linear(beat_len, 96), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(96, 48), nn.ReLU()
        )
        bidirectional = kind in ["BiLSTM", "BiLSTM_Attention"]
        base = "LSTM" if bidirectional else kind
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[base]
        self.rnn = cls(
            48, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=bidirectional,
        )
        dim = hidden * (2 if bidirectional else 1)
        self.attn = TemporalAttention(dim) if kind == "BiLSTM_Attention" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        h, _ = self.rnn(z)
        if self.attn is not None:
            context, weights = self.attn(h)
            return self.head(context), weights
        return self.head(h[:, -1, :]), None


def _normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32).reshape(-1).copy()
    med = np.median(beat)
    scale = np.median(np.abs(beat - med)) * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(beat) + 1e-6
    return np.clip((beat - med) / scale, -8, 8).astype(np.float32)


def load_model(weights_path: str | Path, bundle_path: str | Path | None = None,
               device: str | None = None):
    """Load the checkpoint and metadata exported by the VIGIL notebook."""
    weights_path = Path(weights_path)
    checkpoint = torch.load(weights_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", ["N", "S", "V", "F", "Q"])
    input_shape = checkpoint.get("input_shape", [8, 180])
    model_type = checkpoint.get("model_type", "RNN")
    model = TemporalClassifier(
        beat_len=int(input_shape[-1]), kind=model_type,
        hidden=int(cfg.get("hidden_size", 48)),
        layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.25)),
        n_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    bundle = {}
    if bundle_path is not None and Path(bundle_path).exists():
        bundle = json.loads(Path(bundle_path).read_text())
    return model, bundle, target_device


def predict_ecg(signal: Sequence[float] | np.ndarray, weights_path: str | Path,
                bundle_path: str | Path | None = None, device: str | None = None) -> dict[str, Any]:
    """Predict one centered beat from a raw ECG signal.

    The UI should pass a one-dimensional ECG segment containing at least 180 samples.
    The function centers the window on the midpoint of the supplied segment, applies the
    same robust normalization used during training, and returns JSON-serializable output.
    """
    model, bundle, target_device = load_model(weights_path, bundle_path, device)
    pre = int(bundle.get("beat_pre_samples", 90))
    post = int(bundle.get("beat_post_samples", 90))
    seq_len = int(bundle.get("sequence_length", 8))
    classes = bundle.get("class_names", ["N", "S", "V", "F", "Q"])
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size < pre + post:
        raise ValueError(f"signal must contain at least {pre + post} samples")
    center = arr.size // 2
    if center - pre < 0 or center + post > arr.size:
        center = pre
    beat = _normalize_beat(arr[center - pre:center + post])
    sequence = np.zeros((seq_len, pre + post), dtype=np.float32)
    sequence[-1] = beat
    x = torch.tensor(sequence, dtype=torch.float32, device=target_device).unsqueeze(0)
    with torch.no_grad():
        logits, attention = model(x)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probabilities))
    response = {
        "class_name": classes[idx],
        "class_index": idx,
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(classes)},
        "model_version": bundle.get("model_version", "vigil-arrhythmia-0.1"),
        "sample_rate_hz": bundle.get("sample_rate_hz", 360),
    }
    if attention is not None:
        response["attention"] = [
            {"beat_position": int(i), "weight": float(w)}
            for i, w in enumerate(attention.cpu().numpy()[0])
        ]
    return response
