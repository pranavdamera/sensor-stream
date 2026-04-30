"""Run before/after change-detection inference on a single image pair."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Change-detection inference on one image pair.")
    p.add_argument("--before", required=True, help="Path to before image.")
    p.add_argument("--after", required=True, help="Path to after image.")
    p.add_argument("--checkpoint", default="saved_models/change_detector.pt")
    p.add_argument("--output", default="results/demo_change")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--untrained-demo",
        action="store_true",
        help="Run with randomly initialized weights if no checkpoint exists.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = _parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    before_path = Path(args.before)
    after_path = Path(args.after)
    ckpt_path = (project_root / args.checkpoint).resolve()
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    from vision.inference import (
        load_change_model,
        predict_change_mask,
        preprocess_image_pair,
        save_overlay,
    )
    from vision.model import ChangeDetector

    checkpoint_missing = not ckpt_path.is_file()
    if checkpoint_missing:
        if args.untrained_demo:
            logger.warning(
                "Checkpoint not found — running with randomly initialized weights. "
                "Output is for demonstration only, NOT a trained prediction."
            )
            import torch
            model = ChangeDetector()
            model.eval()
        else:
            logger.error(
                "Checkpoint not found: %s\n"
                "Run scripts/train_change_detector.py first, or pass --untrained-demo.",
                ckpt_path,
            )
            return 1
    else:
        model = load_change_model(ckpt_path)

    before_t, after_t = preprocess_image_pair(before_path, after_path, args.image_size)
    prob_map, binary_mask = predict_change_mask(model, before_t, after_t, args.threshold)

    # Save binary mask
    mask_path = Path(str(output_base) + "_mask.png")
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_img.save(mask_path)
    logger.info("Saved binary mask → %s", mask_path)

    # Save red overlay
    overlay_path = Path(str(output_base) + "_overlay.png")
    save_overlay(before_path, after_path, binary_mask, overlay_path, args.image_size)
    logger.info("Saved overlay     → %s", overlay_path)

    changed_ratio = float(binary_mask.mean())
    logger.info(
        "Changed pixel ratio: %.4f%s",
        changed_ratio,
        " [UNTRAINED DEMO — not a real prediction]" if checkpoint_missing else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
