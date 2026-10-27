"""Channel / spatial attention blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


def build_attention(kind: str, channels: int) -> nn.Module:
    if kind == "cbam":
        return CBAM(channels)
    if kind == "se":
        return SEBlock(channels)
    if kind in ("none", None):
        return nn.Identity()
    raise ValueError(f"Unknown attention '{kind}'")


def build_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
        "silu": nn.SiLU(inplace=True),
        "gelu": nn.GELU(),
        "mish": nn.Mish(inplace=True),
    }[name]
