"""Evaluate the trained change-detection model on the validation split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate change detector on val split.")
    p.add_argument("--data-root", default="data/vision/demo_or_dataset")
    p.add_argument("--checkpoint", default="saved_models/change_detector.pt")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = _parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = (project_root / args.checkpoint).resolve()
    data_root = (project_root / args.data_root).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        logger.error("Checkpoint not found: %s — run train_change_detector.py first.", ckpt_path)
        return 1

    from vision.dataset import ChangeDetectionDataset
    from vision.inference import load_change_model
    from vision.metrics import compute_all_metrics

    try:
        val_ds = ChangeDetectionDataset(data_root, split="val", image_size=args.image_size)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset error: %s", exc)
        return 1

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model = load_change_model(ckpt_path, device=args.device)

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for before, after, mask in val_loader:
            before = before.to(args.device)
            after = after.to(args.device)
            logits = model(before, after)
            probs = np.array(torch.sigmoid(logits).squeeze().cpu().tolist(), dtype=np.float32)
            binary = (probs >= args.threshold).astype(np.uint8)
            all_preds.append(binary)
            all_targets.append(np.array(mask.squeeze().tolist(), dtype=np.uint8))

    y_pred = np.concatenate([a.ravel() for a in all_preds])
    y_true = np.concatenate([a.ravel() for a in all_targets])

    metrics = compute_all_metrics(y_true, y_pred)
    metrics["num_images"] = len(val_ds)
    metrics["threshold"] = args.threshold
    metrics["checkpoint"] = str(ckpt_path)

    out_path = results_dir / "change_detection_metrics.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
    logger.info("Saved → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
