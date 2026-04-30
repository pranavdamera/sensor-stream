"""Vision model loading and inference helpers for the FastAPI app."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_CHECKPOINT_NAME = "change_detector.pt"


def resolve_checkpoint(models_dir: Path) -> Path:
    return models_dir / _CHECKPOINT_NAME


def load_vision_model_if_present(models_dir: Path):
    """Load the change detector if a checkpoint exists; return None otherwise."""
    ckpt = resolve_checkpoint(models_dir)
    if not ckpt.is_file():
        logger.info("No vision checkpoint found at %s — /vision/change-detect will report missing.", ckpt)
        return None
    try:
        from vision.inference import load_change_model
        model = load_change_model(ckpt, device="cpu")
        logger.info("Loaded vision change detector from %s", ckpt)
        return model
    except Exception as exc:
        logger.error("Failed to load vision model: %s", exc)
        return None


def run_change_detection(
    model,
    before_bytes: bytes,
    after_bytes: bytes,
    image_size: int = 256,
    threshold: float = 0.5,
) -> dict:
    """Run inference on raw image bytes; return result dict."""
    from io import BytesIO

    from vision.transforms import pil_to_tensor

    before_img = Image.open(BytesIO(before_bytes)).convert("RGB")
    after_img = Image.open(BytesIO(after_bytes)).convert("RGB")

    before_t = pil_to_tensor(before_img, image_size).unsqueeze(0)
    after_t = pil_to_tensor(after_img, image_size).unsqueeze(0)

    with torch.no_grad():
        logits = model(before_t, after_t)
        probs_list = torch.sigmoid(logits).squeeze().cpu().tolist()

    probs = np.array(probs_list, dtype=np.float32)
    binary = (probs >= threshold).astype(np.uint8)
    changed_ratio = float(binary.mean())
    change_detected = changed_ratio > 0.01  # >1 % of pixels changed

    return {
        "change_detected": change_detected,
        "changed_pixel_ratio": changed_ratio,
        "threshold": threshold,
        "model_loaded": True,
    }
