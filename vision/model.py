"""Lightweight PyTorch change-detection model for before/after image pairs."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChangeDetector(nn.Module):
    """Tiny U-Net style encoder-decoder for binary change-mask prediction.

    Inputs:
        before: ``(B, 3, H, W)`` image tensor, normalized.
        after:  ``(B, 3, H, W)`` image tensor, normalized.

    Output:
        ``(B, 1, H, W)`` raw logits — apply sigmoid for change probabilities.

    Designed to run on CPU for demo purposes.  H and W must be divisible by 8.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (input = 6 channels: before||after concatenated)
        self.enc1 = _ConvBlock(6, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)

        # Bottleneck
        self.bottleneck = _ConvBlock(128, 256)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(64, 32)

        # 1×1 head — outputs single-channel logit map
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        x = torch.cat([before, after], dim=1)  # (B, 6, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
