"""Multi-head spatial attention for the ChangeDetector bottleneck."""
from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAttention(nn.Module):
    """Multi-head self-attention over spatial tokens at the U-Net bottleneck.

    The feature map ``(B, C, H, W)`` is reshaped into a sequence of spatial
    tokens ``(B, H*W, C)``, processed with ``nn.MultiheadAttention``, and
    projected back. A residual connection preserves the original activations.

    Args:
        channels: Number of feature channels (must be divisible by ``num_heads``).
        num_heads: Number of parallel attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed)
        out = attended.permute(0, 2, 1).view(B, C, H, W)
        return x + out