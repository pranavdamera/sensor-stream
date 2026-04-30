"""Train the lightweight PyTorch change-detection model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train satellite image change detector.")
    p.add_argument("--data-root", default="data/vision/demo_or_dataset", help="Dataset root.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", default="saved_models")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for before, after, mask in loader:
            before = before.to(device)
            after = after.to(device)
            mask = mask.to(device)
            logits = model(before, after)
            loss = criterion(logits, mask)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * before.size(0)
    return total_loss / max(len(loader.dataset), 1)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = _parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    data_root = (project_root / args.data_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from vision.dataset import ChangeDetectionDataset
    from vision.model import ChangeDetector

    try:
        train_ds = ChangeDetectionDataset(data_root, split="train", image_size=args.image_size)
        val_ds = ChangeDetectionDataset(data_root, split="val", image_size=args.image_size)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset error: %s", exc)
        logger.error(
            "Prepare paired images under %s/train/{before,after,mask} and val/…", data_root
        )
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ChangeDetector().to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss = _run_epoch(model, val_loader, criterion, None, args.device)
        logger.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f", epoch, args.epochs, train_loss, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    ckpt_path = output_dir / "change_detector.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    meta = {
        "image_size": args.image_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "device": args.device,
        "data_root": str(data_root),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "history": history,
    }
    meta_path = output_dir / "change_detector_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved checkpoint → %s", ckpt_path)
    logger.info("Saved metadata  → %s", meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
