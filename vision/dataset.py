"""Paired before/after/mask dataset for satellite change detection."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from vision.transforms import mask_pil_to_tensor, pil_to_tensor


class ChangeDetectionDataset(Dataset):
    """Dataset for paired satellite change-detection images.

    Expected directory layout::

        root/
          train/
            before/  *.png (or .jpg)
            after/   *.png (or .jpg)
            mask/    *.png  (binary: 255 = changed, 0 = no change)
          val/
            before/
            after/
            mask/

    Files are matched by stem (filename without extension).  Only stems present
    in all three subdirectories are included.
    """

    _IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
    ) -> None:
        self.root = Path(root) / split
        self.image_size = image_size

        before_dir = self.root / "before"
        after_dir = self.root / "after"
        mask_dir = self.root / "mask"

        for d in (before_dir, after_dir, mask_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Expected directory not found: {d}")

        def _stems(directory: Path) -> set[str]:
            return {
                p.stem
                for p in directory.iterdir()
                if p.suffix.lower() in self._IMG_SUFFIXES
            }

        valid_stems = _stems(before_dir) & _stems(after_dir) & _stems(mask_dir)
        if not valid_stems:
            raise ValueError(f"No matching image triples found under {self.root}")

        self.stems = sorted(valid_stems)
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.mask_dir = mask_dir

    def _find_file(self, directory: Path, stem: str) -> Path:
        for suffix in self._IMG_SUFFIXES:
            p = directory / f"{stem}{suffix}"
            if p.is_file():
                return p
        raise FileNotFoundError(f"No image found for stem {stem!r} in {directory}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        before_img = Image.open(self._find_file(self.before_dir, stem)).convert("RGB")
        after_img = Image.open(self._find_file(self.after_dir, stem)).convert("RGB")
        mask_img = Image.open(self._find_file(self.mask_dir, stem))

        before = pil_to_tensor(before_img, self.image_size)
        after = pil_to_tensor(after_img, self.image_size)
        mask = mask_pil_to_tensor(mask_img, self.image_size)
        return before, after, mask
