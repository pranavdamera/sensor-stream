"""Deterministic PIL-to-tensor preprocessing helpers for change detection.

torchvision is used when available.  If it is not installed, a pure
PIL + NumPy fallback is used — the math is identical.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

try:
    from torchvision import transforms as _T
    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover
    _HAS_TORCHVISION = False

# ImageNet statistics — keep in sync with training normalization.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pil_to_tensor_numpy(image: Image.Image, image_size: int) -> torch.Tensor:
    """Pure PIL+NumPy path used when torchvision is absent."""
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0   # (H, W, 3)
    arr = (arr - _MEAN) / _STD
    chw = arr.transpose(2, 0, 1).copy()               # (3, H, W), contiguous
    return torch.tensor(chw, dtype=torch.float32)    # explicit dtype needed for NumPy 2.x compat


def pil_to_tensor(image: Image.Image, image_size: int = 256) -> torch.Tensor:
    """Convert a PIL RGB image to a normalized float tensor ``(3, H, W)``."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if _HAS_TORCHVISION:
        tf = _T.Compose([
            _T.Resize((image_size, image_size)),
            _T.ToTensor(),
            _T.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
        ])
        return tf(image)
    return _pil_to_tensor_numpy(image, image_size)


def mask_pil_to_tensor(mask: Image.Image, image_size: int = 256) -> torch.Tensor:
    """Convert a binary mask PNG to a float tensor ``(1, H, W)`` in [0, 1]."""
    if mask.mode != "L":
        mask = mask.convert("L")
    if _HAS_TORCHVISION:
        resize = _T.Resize((image_size, image_size), interpolation=_T.InterpolationMode.NEAREST)
        mask = resize(mask)
        arr = np.array(mask, dtype=np.float32) / 255.0
    else:
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        arr = np.array(mask, dtype=np.float32) / 255.0
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
    return (t > 0.5).float()
