"""Inference helpers for the PyTorch change-detection model.

cv2 is used for the overlay when available; falls back to pure PIL+NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vision.model import ChangeDetector
from vision.transforms import pil_to_tensor


def load_change_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> ChangeDetector:
    """Load a ChangeDetector from a saved ``.pt`` checkpoint."""
    model = ChangeDetector()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image_pair(
    before_path: str | Path,
    after_path: str | Path,
    image_size: int = 256,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess a before/after image pair into batched tensors.

    Returns ``(before, after)`` each shaped ``(1, 3, H, W)`` on ``device``.
    """
    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")
    before = pil_to_tensor(before_img, image_size).unsqueeze(0).to(device)
    after = pil_to_tensor(after_img, image_size).unsqueeze(0).to(device)
    return before, after


def predict_change_mask(
    model: ChangeDetector,
    before: torch.Tensor,
    after: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return probability map and binary mask.

    Returns ``(prob_map, binary_mask)`` as float32 numpy arrays ``(H, W)``.
    """
    with torch.no_grad():
        logits = model(before, after)
        # Avoid tensor.numpy() — broken when NumPy ≥ 2.x is paired with older torch builds.
        probs_list = torch.sigmoid(logits).squeeze().cpu().tolist()
    probs = np.array(probs_list, dtype=np.float32)
    binary = (probs >= threshold).astype(np.uint8)
    return probs, binary


def _resize_mask_numpy(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize using PIL (no cv2 required)."""
    pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (size, size), Image.NEAREST
    )
    return (np.array(pil) > 127).astype(np.uint8)


def save_overlay(
    before_path: str | Path,
    after_path: str | Path,
    mask: np.ndarray,
    output_path: str | Path,
    image_size: int = 256,
) -> None:
    """Save a red-tinted overlay of detected changes on the after image."""
    after_img = Image.open(after_path).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    after_arr = np.array(after_img, dtype=np.uint8)

    if mask.shape != (image_size, image_size):
        try:
            import cv2
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (image_size, image_size),
                interpolation=cv2.INTER_NEAREST,
            )
        except ImportError:
            mask_resized = _resize_mask_numpy(mask, image_size)
    else:
        mask_resized = mask.astype(np.uint8)

    overlay = after_arr.copy()
    changed = mask_resized.astype(bool)
    overlay[changed, 0] = np.clip(after_arr[changed, 0].astype(int) + 120, 0, 255)
    overlay[changed, 1] = np.clip(after_arr[changed, 1].astype(int) - 60,  0, 255)
    overlay[changed, 2] = np.clip(after_arr[changed, 2].astype(int) - 60,  0, 255)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
