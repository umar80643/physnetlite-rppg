"""
Stage 3: Deep learning model.

A compact 3D-CNN inspired by PhysNet (Yu, Li & Zhao, 2019, "Remote
Photoplethysmograph Signal Measurement from Facial Videos Using
Spatio-Temporal Networks"). Takes a (B, C, T, H, W) clip of face-ROI
frames and outputs a reconstructed 1D pulse waveform of length T, from
which BPM is derived via FFT (src.utils.estimate_hr_fft) -- the same way
ground truth and the classical baseline are scored, so all three are
compared on equal footing.

Kept intentionally small (a handful of conv blocks, <1M params) so it
trains on CPU or a single consumer GPU in reasonable time, per the
project's stated goal of a correct, well-evaluated pipeline rather than
a SOTA architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Conv3d -> BatchNorm3d -> ReLU, with optional spatial-only or
    spatiotemporal max pooling."""

    def __init__(self, in_ch: int, out_ch: int, pool: str | None = "spatial"):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        if pool == "spatial":
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        elif pool == "spatiotemporal":
            self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        elif pool is None:
            self.pool = nn.Identity()
        else:
            raise ValueError(f"Unknown pool type: {pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x)


class PhysNetLite(nn.Module):
    """Compact PhysNet-style architecture.

    Input:  (B, 3, T, H, W) RGB face-ROI clip, values in [0, 1].
    Output: (B, T) reconstructed pulse waveform (z-scored per-sample).

    Architecture: 4 spatial-downsampling conv blocks bring (H, W) to a
    small spatial extent while preserving the temporal dimension T
    (critical -- the whole point is to keep per-frame temporal
    resolution for pulse extraction), followed by adaptive spatial
    pooling to 1x1 and a temporal 1D conv head that maps directly to a
    per-frame waveform value.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.block1 = ConvBlock3D(in_channels, c, pool="spatial")
        self.block2 = ConvBlock3D(c, c * 2, pool="spatial")
        self.block3 = ConvBlock3D(c * 2, c * 4, pool="spatial")
        self.block4 = ConvBlock3D(c * 4, c * 4, pool="spatial")

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep T, collapse H,W

        # Temporal head: 1D conv over the T dimension per channel, then
        # project channels -> 1 (the waveform value at each timestep).
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(c * 4, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(c * 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T, H, W) float tensor in [0, 1].

        Returns:
            (B, T) predicted pulse waveform, z-scored per sample so it's
            directly comparable (up to sign/scale) to a z-scored ground
            truth waveform under the negative-Pearson loss.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)          # (B, C, T, H', W')
        x = self.spatial_pool(x)    # (B, C, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, C, T)

        waveform = self.temporal_conv(x).squeeze(1)  # (B, T)
        # Per-sample z-score normalization (standard for the negative
        # Pearson loss, which is scale/shift invariant anyway, but this
        # keeps outputs numerically well-behaved).
        mean = waveform.mean(dim=1, keepdim=True)
        std = waveform.std(dim=1, keepdim=True) + 1e-6
        return (waveform - mean) / std

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NegativePearsonLoss(nn.Module):
    """Negative Pearson correlation loss, the standard training loss in
    the rPPG literature (used by PhysNet, DeepPhys, and follow-ups).
    Minimizing this maximizes correlation between predicted and
    ground-truth waveforms, which is invariant to their absolute scale
    and DC offset -- appropriate since we don't know the true units of
    the model's internal "waveform" output relative to a pulse
    oximeter's.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, T) predicted waveform.
            target: (B, T) ground-truth waveform, same length.

        Returns:
            Scalar loss = mean over batch of (1 - Pearson correlation).
        """
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)

        numerator = (pred_c * target_c).sum(dim=1)
        denominator = torch.sqrt((pred_c**2).sum(dim=1) * (target_c**2).sum(dim=1) + 1e-8)
        corr = numerator / denominator
        return (1 - corr).mean()
