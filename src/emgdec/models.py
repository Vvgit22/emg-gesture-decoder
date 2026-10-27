"""The neural network: a small 1-D CNN that reads raw EMG windows.

What goes in
    A batch of raw windows, float32, shape [B, C, W] — exactly the layout
    the memmap stores (C channels, W time samples), *un-normalised*.
    LDA ate hand-made features; this model eats the signal itself.

What comes out
    Logits, shape [B, 17] — one score per gesture; the biggest wins.

How it works
    The model's first step is `(x - mean) / std`, where mean and std are
    *registered buffers*: they travel inside the model and are saved and
    exported with it. That is deliberate — the browser demo must be able
    to feed raw electrode signal straight in, so the recipe for centring
    it lives inside the graph, not in a sidecar file. The stats come from
    the TRAIN windows of each split only (see `set_normalisation`).

    The body is four strided convolutions with batch-norm and ReLU. A
    convolution slides a small detector along the trace: one output
    channel might learn "a burst of energy around 80 Hz", another "a slow
    rise". Each layer halves the time axis, so the fourth layer sees a
    summary of the whole 250 ms. `AdaptiveAvgPool1d(1)` then averages over
    what is left of time — this is why the model accepts ANY window
    length >= 16, which the window-length ablation relies on. Dropout +
    a linear layer produce the 17 scores. Total: ~57k parameters, tiny
    enough that every split trains in seconds on a laptop GPU.

Channel configs (which electrode rings feed the model):
    forearm = channels 0-15, wrist = 16-27, both = all 28.
"""
from __future__ import annotations

import torch
from torch import nn

CHANNELS = {
    "forearm": range(0, 16),
    "wrist": range(16, 28),
    "both": range(0, 28),
}


class TinyCNN(nn.Module):
    """Small 1-D CNN: normalise -> conv stack -> global pool -> linear."""

    def __init__(self, n_channels: int, n_classes: int = 17,
                 mean: torch.Tensor | None = None,
                 std: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_channels, 1)
                             if mean is None else mean.reshape(n_channels, 1))
        self.register_buffer("std", torch.ones(n_channels, 1)
                             if std is None else std.reshape(n_channels, 1))
        self.body = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(96, n_classes)

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store the per-channel stats as buffers (std floored at 1e-6)."""
        self.mean.copy_(mean.reshape(self.mean.shape))
        self.std.copy_(std.reshape(self.std.shape).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std  # normalisation inside the graph
        x = self.body(x).squeeze(-1)
        return self.head(x)


def n_params(model: nn.Module) -> int:
    """Total parameter count — ~57k for C=28 (conv kernels + BN + linear)."""
    return sum(p.numel() for p in model.parameters())
