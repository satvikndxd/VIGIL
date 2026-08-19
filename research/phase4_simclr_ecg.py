"""VIGIL Phase 4: SimCLR-style self-supervised contrastive learning for 1D ECG.

Purpose
-------
Phase 1--3 showed that the compact RNN + RR + handcrafted-morphology baseline
was the strongest measured model on VIGIL's locked record-level split, but that
its N->V failures are dominated by cross-record / original-symbol shift.

This module proposes, but does NOT claim results for, Phase 4. The encoder is
first trained without class labels on waveforms drawn ONLY from training
records. It learns to map two physiologically plausible views of the same beat
near each other, then serves as the initialization for a supervised five-class
classifier. Validation and the locked test set must remain untouched during
self-supervised pretraining.

Expected input layout
---------------------
* A batch is a float Tensor shaped [B, L] or [B, C, L].
* VIGIL's beat-centered waveform window is normally L=180 (90 pre / 90 post).
* The code is agnostic to L because the encoder ends in adaptive pooling.
* Each waveform should use the SAME robust normalization contract as the
  existing VIGIL pipeline. Do not normalize using validation or test records.

Important scientific guardrails
-------------------------------
1. Contrastive pretraining uses train-record waveforms only. It must never
   consume validation/test waveforms, even without labels.
2. Augmentations model nuisance variation; they must not systematically erase
   diagnostic morphology. Masking protects the center of the beat by default.
3. Phase 4 is a hypothesis. It becomes a model-selection candidate only after
   the existing VIGIL replacement gate, seed study, bootstrap, calibration,
   robustness, and per-record analyses are repeated.

References
----------
Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations," ICML 2020: https://proceedings.mlr.press/v119/chen20j.html
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 1. Reproducibility helpers
# ---------------------------------------------------------------------------


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch seeds for a controlled rerun.

    Deterministic CuDNN can lower throughput, but is appropriate while Phase 4
    is still evaluated as a scientific intervention. A thesis result is more
    valuable when another student can reproduce the same candidate ranking.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Warn rather than fail if an operation has no deterministic kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# 2. ECG-specific SimCLR augmentations
# ---------------------------------------------------------------------------


class ECGSimCLRAugment(nn.Module):
    """Generate a physiologically conservative contrastive view of ECG beats.

    The goal is NOT generic image-style corruption. A positive view must retain
    the beat's underlying identity while varying nuisance factors that differ
    across recordings: slow baseline drift, gain / lead amplitude variation,
    and incomplete local observations.

    Augmentations included by design:
      * Baseline wander injection: low-frequency sinusoidal drift.
      * Random amplitude scaling: mimics gain and contact differences.
      * Temporal masking: replaces a short non-central interval with its local
        channel mean. The default protected center avoids always deleting the
        QRS complex at the beat-centered window midpoint.

    Parameters are intentionally modest. Aggressive waveform warping or
    central-QRS deletion can convert a positive pair into a different
    morphology class and would teach the wrong invariance.
    """

    def __init__(
        self,
        wander_probability: float = 0.80,
        wander_amplitude_fraction: Tuple[float, float] = (0.02, 0.10),
        wander_cycles_per_window: Tuple[float, float] = (0.05, 0.50),
        amplitude_probability: float = 0.80,
        amplitude_scale: Tuple[float, float] = (0.85, 1.15),
        masking_probability: float = 0.50,
        mask_fraction: Tuple[float, float] = (0.04, 0.12),
        protect_center_fraction: float = 0.30,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.wander_probability = wander_probability
        self.wander_amplitude_fraction = wander_amplitude_fraction
        self.wander_cycles_per_window = wander_cycles_per_window
        self.amplitude_probability = amplitude_probability
        self.amplitude_scale = amplitude_scale
        self.masking_probability = masking_probability
        self.mask_fraction = mask_fraction
        self.protect_center_fraction = protect_center_fraction
        self.eps = eps

        if not 0.0 <= protect_center_fraction < 1.0:
            raise ValueError("protect_center_fraction must be in [0, 1).")

    @staticmethod
    def _as_bcl(x: Tensor) -> Tuple[Tensor, bool]:
        """Return [B, C, L] and whether a channel dimension was added."""
        if x.ndim == 2:  # [B, L] -> one ECG channel
            return x.unsqueeze(1), True
        if x.ndim == 3:
            return x, False
        raise ValueError(f"Expected [B, L] or [B, C, L], got {tuple(x.shape)}")

    @staticmethod
    def _uniform(
        low: float, high: float, shape: Sequence[int], device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return torch.empty(*shape, device=device, dtype=dtype).uniform_(low, high)

    def _inject_baseline_wander(self, x: Tensor) -> Tensor:
        """Add a low-frequency sinusoidal drift scaled to each waveform.

        For a beat-centered short window, low-frequency wander appears mainly
        as a slowly changing offset. We parameterize frequency as cycles per
        window so the transform works even if future experiments change the
        sample rate or window length.
        """
        batch, _, length = x.shape
        device, dtype = x.device, x.dtype

        apply = (torch.rand(batch, 1, 1, device=device) < self.wander_probability).to(dtype)
        # A robust amplitude proxy avoids a few sharp QRS samples dominating
        # the injected wander magnitude.
        per_beat_scale = x.detach().flatten(1).std(dim=1, keepdim=True).clamp_min(self.eps)
        per_beat_scale = per_beat_scale.view(batch, 1, 1)

        amplitude = self._uniform(
            self.wander_amplitude_fraction[0],
            self.wander_amplitude_fraction[1],
            (batch, 1, 1),
            device,
            dtype,
        ) * per_beat_scale
        cycles = self._uniform(
            self.wander_cycles_per_window[0],
            self.wander_cycles_per_window[1],
            (batch, 1, 1),
            device,
            dtype,
        )
        phase = self._uniform(0.0, 2.0 * math.pi, (batch, 1, 1), device, dtype)
        time = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype).view(1, 1, length)
        wander = amplitude * torch.sin((2.0 * math.pi * cycles * time) + phase)
        return x + apply * wander

    def _apply_amplitude_scaling(self, x: Tensor) -> Tensor:
        """Apply per-beat gain variation without changing the temporal axis."""
        batch = x.shape[0]
        apply = (torch.rand(batch, 1, 1, device=x.device) < self.amplitude_probability).to(x.dtype)
        scale = self._uniform(
            self.amplitude_scale[0],
            self.amplitude_scale[1],
            (batch, 1, 1),
            x.device,
            x.dtype,
        )
        # If an augmentation is not selected, use exact identity scale 1.0.
        scale = apply * scale + (1.0 - apply)
        return x * scale

    def _sample_mask_start(self, length: int, width: int, device: torch.device) -> int:
        """Sample a start index, preferring regions outside the central QRS zone."""
        center_start = int(length * (1.0 - self.protect_center_fraction) / 2.0)
        center_end = length - center_start
        candidates = []
        for start in range(0, max(1, length - width + 1)):
            end = start + width
            # Preserve the beat-centered central region when a non-central
            # interval exists. Fall back to any interval for very long masks.
            if end <= center_start or start >= center_end:
                candidates.append(start)
        if not candidates:
            return int(torch.randint(0, max(1, length - width + 1), (1,), device=device).item())
        choice = int(torch.randint(0, len(candidates), (1,), device=device).item())
        return candidates[choice]

    def _temporal_mask(self, x: Tensor) -> Tensor:
        """Mask short non-central spans, using a per-channel mean fill value.

        Mean filling represents local absence / dropout without injecting a
        sharp artificial high-amplitude edge. The operation is executed per
        beat because mask positions must differ across contrastive views.
        """
        out = x.clone()
        batch, channels, length = out.shape
        fill = out.mean(dim=-1, keepdim=True)
        min_width = max(1, int(round(self.mask_fraction[0] * length)))
        max_width = max(min_width, int(round(self.mask_fraction[1] * length)))

        for b in range(batch):
            if torch.rand((), device=out.device).item() >= self.masking_probability:
                continue
            width = int(torch.randint(min_width, max_width + 1, (1,), device=out.device).item())
            start = self._sample_mask_start(length, width, out.device)
            out[b, :, start : start + width] = fill[b].expand(channels, width)
        return out

    def forward(self, x: Tensor) -> Tensor:
        """Return one independently augmented ECG view with input shape preserved."""
        x_bcl, squeezed = self._as_bcl(x)
        # Clone first: callers may reuse the original view for a second branch.
        view = x_bcl.clone()
        view = self._inject_baseline_wander(view)
        view = self._apply_amplitude_scaling(view)
        view = self._temporal_mask(view)
        return view.squeeze(1) if squeezed else view


class ECGPairDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset that emits two independent contrastive views of one ECG beat.

    Only pass train-record waveforms here. The `record_ids` argument is optional
    and stored for auditability, not used by SimCLR itself. Keeping record IDs
    alongside the dataset makes it easy to assert that no validation/test record
    is accidentally included in pretraining.
    """

    def __init__(
        self,
        waveforms: Union[Tensor, np.ndarray],
        transform: Optional[nn.Module] = None,
        record_ids: Optional[Sequence[Union[str, int]]] = None,
    ) -> None:
        super().__init__()
        self.waveforms = torch.as_tensor(waveforms, dtype=torch.float32).contiguous()
        if self.waveforms.ndim not in (2, 3):
            raise ValueError("waveforms must have shape [N, L] or [N, C, L].")
        if record_ids is not None and len(record_ids) != len(self.waveforms):
            raise ValueError("record_ids and waveforms must have the same length.")
        self.record_ids = tuple(map(str, record_ids)) if record_ids is not None else None
        self.transform = transform if transform is not None else ECGSimCLRAugment()

    def __len__(self) -> int:
        return self.waveforms.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.waveforms[index]
        # Two stochastic views of the *same* waveform form the positive pair.
        return self.transform(x.unsqueeze(0)).squeeze(0), self.transform(x.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# 3. One-dimensional encoder and SimCLR projection head
# ---------------------------------------------------------------------------


class ECGResidualBlock(nn.Module):
    """A lightweight dilated residual block for multi-scale QRS morphology.

    Dilation increases the receptive field without aggressively downsampling
    the 180-sample waveform. The residual path helps preserve fine local QRS
    structure while later layers combine longer waveform context.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dilation: int = 1, stride: int = 1
    ) -> None:
        super().__init__()
        kernel_size = 5
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=0.05)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class ECGEncoder1D(nn.Module):
    """Map a beat-centered waveform to a morphology embedding h in R^D.

    The encoder is intentionally compact so it is a controlled Phase 4
    intervention rather than an uncontrolled parameter-count increase. The
    projection head used during pretraining is discarded before fine-tuning;
    supervised classification consumes `h`, not the projection `z`.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ECGResidualBlock(base_channels, base_channels, dilation=1),
            ECGResidualBlock(base_channels, base_channels * 2, dilation=2, stride=2),
            ECGResidualBlock(base_channels * 2, base_channels * 4, dilation=4, stride=2),
            ECGResidualBlock(base_channels * 4, embedding_dim, dilation=8, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    @staticmethod
    def _as_bcl(x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3:
            return x
        raise ValueError(f"Expected [B, L] or [B, C, L], got {tuple(x.shape)}")

    def forward(self, x: Tensor) -> Tensor:
        x = self._as_bcl(x)
        x = self.stem(x)
        x = self.features(x)
        return self.pool(x).squeeze(-1)  # [B, embedding_dim]


class ProjectionHead(nn.Module):
    """Nonlinear SimCLR projection MLP.

    Separating h (useful downstream representation) from z (contrastive-loss
    representation) lets the contrastive objective reshape z without forcing
    the classifier to consume that exact geometry.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class ECGSimCLR(nn.Module):
    """Encoder + projection head used only during self-supervised pretraining."""

    def __init__(
        self,
        encoder: Optional[ECGEncoder1D] = None,
        in_channels: int = 1,
        embedding_dim: int = 128,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else ECGEncoder1D(in_channels, embedding_dim)
        self.projector = ProjectionHead(self.encoder.embedding_dim, hidden_dim=2 * embedding_dim, output_dim=projection_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(self.encoder(x))

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Return the pre-projection morphology representation h for downstream use."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 4. NT-Xent / InfoNCE objective
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    r"""Normalized temperature-scaled cross-entropy loss used by SimCLR.

    For two augmented views of the same ECG beat, i and j, the positive-pair
    objective is

        l_{i,j} = -log( exp(sim(z_i, z_j) / tau)
                       / sum_{k != i} exp(sim(z_i, z_k) / tau) ).

    Here `sim` is cosine similarity because all z vectors are L2-normalized.
    The denominator includes the true positive and every other example in the
    2B-view minibatch. The final loss averages both directions of every pair.

    Interpretation for ECG: two views of the same beat should remain close
    despite nuisance drift, gain, or short missing spans; beats from different
    source examples act as negatives. A future extension may add domain-aware
    negative handling if semantically identical repeated morphology should not
    be repelled, but the baseline below is faithful SimCLR-style NT-Xent.
    """

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        if z_i.shape != z_j.shape:
            raise ValueError("Both views must produce identically shaped embeddings.")
        if z_i.ndim != 2:
            raise ValueError("Embeddings must have shape [B, D].")
        batch_size = z_i.shape[0]
        if batch_size < 2:
            raise ValueError("NT-Xent needs at least two source ECG examples per batch.")

        # Normalize so dot product is cosine similarity: sim(a, b) = a^T b.
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
        # [2B, 2B] logits, divided by temperature tau to control concentration.
        logits = (z @ z.T) / self.temperature

        # An embedding is never allowed to choose itself as a positive.
        diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, float("-inf"))

        # Positive index: i <-> i+B for the concatenation [z_i; z_j].
        targets = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# 5. Pretraining loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimCLRConfig:
    """Default Phase 4 settings; tune only through validation experiments."""

    temperature: float = 0.10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 128
    seed: int = 42


def _move_batch(batch: Union[Tensor, Sequence[Tensor]], device: Union[str, torch.device]) -> Tensor:
    """Accept loaders that yield tensors or (x, y, ...) tuples and return x."""
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return x.to(device, non_blocking=True)


def pretrain_one_epoch(
    model: ECGSimCLR,
    loader: Iterable[Tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: NTXentLoss,
    device: Union[str, torch.device],
) -> float:
    """Run one label-free SimCLR epoch and return mean NT-Xent loss."""
    model.train()
    total_loss, batches = 0.0, 0
    for view_1, view_2 in loader:
        view_1 = view_1.to(device, non_blocking=True)
        view_2 = view_2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        z_1 = model(view_1)
        z_2 = model(view_2)
        loss = loss_fn(z_1, z_2)
        loss.backward()
        # A conservative clip prevents an unstable contrastive update from
        # destroying the encoder before downstream fine-tuning starts.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("Pretraining loader produced zero batches. Check drop_last and batch size.")
    return total_loss / batches


def pretrain_simclr(
    model: ECGSimCLR,
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    config: SimCLRConfig,
    device: Union[str, torch.device],
) -> list[float]:
    """Full Phase 4 pretraining procedure using only train-record waveforms.

    AdamW is a *proposal* for Phase 4, not a rewrite of reported Phase 1--3
    artifacts. Store its exact configuration beside every candidate result.
    """
    set_global_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = NTXentLoss(config.temperature)
    history: list[float] = []
    for epoch in range(1, config.epochs + 1):
        epoch_loss = pretrain_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history.append(epoch_loss)
        print(f"[SimCLR pretrain] epoch={epoch:03d}/{config.epochs} ntxent={epoch_loss:.5f}")
    return history


def save_pretrained_encoder(model: ECGSimCLR, output_path: Union[str, Path], config: SimCLRConfig) -> None:
    """Persist encoder weights and exact pretraining hyperparameters together."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "projector_state_dict": model.projector.state_dict(),
            "simclr_config": config.__dict__,
            "architecture": "ECGEncoder1D + ProjectionHead",
        },
        output_path,
    )


# ---------------------------------------------------------------------------
# 6. Supervised fine-tuning on the existing N/S/V/F/Q task
# ---------------------------------------------------------------------------


class ECGClassifier(nn.Module):
    """Five-class head attached to a pretrained ECG encoder.

    The projection head is deliberately absent. Fine-tuning uses h because h is
    the representation intended to transfer to the actual classification task.
    """

    def __init__(self, encoder: ECGEncoder1D, num_classes: int = 5, dropout: float = 0.20) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder.embedding_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Optional linear-probe stage before full fine-tuning."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable


def fine_tune_one_epoch(
    model: ECGClassifier,
    loader: Iterable[Union[Tuple[Tensor, Tensor], Sequence[Tensor]]],
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    class_weights: Optional[Tensor] = None,
) -> float:
    """Train the supervised N/S/V/F/Q head for one epoch.

    `class_weights` should be computed from the training split only. Do not use
    validation/test counts, and do not claim class-weighting fixes OOD shift
    without rerunning VIGIL's original replacement gate.
    """
    model.train()
    total_loss, batches = 0.0, 0
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Fine-tuning loader must yield (waveform, mapped_class, ...).")
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True).long()
        weight = class_weights.to(device) if class_weights is not None else None

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("Fine-tuning loader produced zero batches.")
    return total_loss / batches


# ---------------------------------------------------------------------------
# 7. Minimal integration sketch (not executed on import)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # This block intentionally does not download data or run a result-producing
    # experiment. Connect it to VIGIL's existing TRAIN-record waveform arrays.
    #
    # Expected leakage-safe usage:
    #   train_waveforms = X_train_waveforms  # records in split_lock.train only
    #   simclr_ds = ECGPairDataset(train_waveforms, ECGSimCLRAugment(), train_record_ids)
    #   loader = DataLoader(simclr_ds, batch_size=128, shuffle=True, drop_last=True)
    #   ssl = ECGSimCLR(in_channels=1, embedding_dim=128, projection_dim=128)
    #   history = pretrain_simclr(ssl, loader, SimCLRConfig(), device="cuda")
    #   save_pretrained_encoder(ssl, "research_results/phase4/simclr_encoder.pt", SimCLRConfig())
    #   classifier = ECGClassifier(ssl.encoder, num_classes=5)
    #
    # Then fine-tune only with labeled train records, select on the locked
    # validation records, and evaluate once on the unchanged locked test split.
    print("Phase 4 SimCLR module loaded. Connect train-record waveforms to begin a controlled experiment.")
