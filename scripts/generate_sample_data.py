"""Generate synthetic before/after/mask image triples for pipeline demos.

Creates small (256×256) satellite-like RGB PNGs using only PIL + NumPy.
No external datasets or internet access required.

Scenes:
  train/scene_001 — building footprint appears on open ground
  train/scene_002 — road construction cuts through vegetation
  train/scene_003 — water body recedes (drought / seasonal)
  val/scene_001   — urban infill: empty lot → structure

Usage:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VISION_ROOT = ROOT / "data" / "vision" / "demo_or_dataset"

RNG = np.random.default_rng(42)
SIZE = 256


# ---------------------------------------------------------------------------
# Low-level image helpers
# ---------------------------------------------------------------------------

def _base_terrain(hue: str = "green") -> np.ndarray:
    """Return a noisy terrain-like RGB base image (H, W, 3) uint8."""
    if hue == "green":
        base = np.array([60, 100, 40], dtype=np.float32)
    elif hue == "tan":
        base = np.array([150, 130, 90], dtype=np.float32)
    elif hue == "blue":
        base = np.array([60, 100, 160], dtype=np.float32)
    else:
        base = np.array([120, 120, 120], dtype=np.float32)

    noise = RNG.integers(-25, 25, (SIZE, SIZE, 3)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color) -> np.ndarray:
    out = img.copy()
    noise = RNG.integers(-10, 10, (y1 - y0, x1 - x0, 3)).astype(np.int32)
    block = np.clip(np.array(color, dtype=np.int32) + noise, 0, 255).astype(np.uint8)
    out[y0:y1, x0:x1] = block
    return out


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int, color) -> np.ndarray:
    """Draw a thick axis-aligned line segment (horizontal or vertical only)."""
    out = img.copy()
    if abs(x1 - x0) > abs(y1 - y0):
        # horizontal
        xa, xb = sorted([x0, x1])
        ya = max(0, y0 - thickness // 2)
        yb = min(SIZE, y0 + thickness // 2)
        noise = RNG.integers(-8, 8, (yb - ya, xb - xa, 3)).astype(np.int32)
        block = np.clip(np.array(color, dtype=np.int32) + noise, 0, 255).astype(np.uint8)
        out[ya:yb, xa:xb] = block
    else:
        # vertical
        ya, yb = sorted([y0, y1])
        xa = max(0, x0 - thickness // 2)
        xb = min(SIZE, x0 + thickness // 2)
        noise = RNG.integers(-8, 8, (yb - ya, xb - xa, 3)).astype(np.int32)
        block = np.clip(np.array(color, dtype=np.int32) + noise, 0, 255).astype(np.uint8)
        out[ya:yb, xa:xb] = block
    return out


def _mask_from_rect(x0, y0, x1, y1) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _mask_from_line(x0, y0, x1, y1, thickness) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if abs(x1 - x0) > abs(y1 - y0):
        xa, xb = sorted([x0, x1])
        ya = max(0, y0 - thickness // 2)
        yb = min(SIZE, y0 + thickness // 2)
        mask[ya:yb, xa:xb] = 255
    else:
        ya, yb = sorted([y0, y1])
        xa = max(0, x0 - thickness // 2)
        xb = min(SIZE, x0 + thickness // 2)
        mask[ya:yb, xa:xb] = 255
    return mask


def _save(split: str, stem: str, before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> None:
    for sub, arr in [("before", before), ("after", after)]:
        p = VISION_ROOT / split / sub
        p.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(p / f"{stem}.png")
    p = VISION_ROOT / split / "mask"
    p.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(p / f"{stem}.png")
    print(f"  Saved {split}/{stem}")


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

def scene_building(split: str, stem: str) -> None:
    """Empty lot → building footprint."""
    before = _base_terrain("tan")
    # building: dark grey rectangle
    after = _draw_rect(before, 80, 70, 180, 160, [90, 85, 80])
    # roof detail: slightly lighter centre
    after = _draw_rect(after, 100, 90, 160, 140, [160, 155, 145])
    mask = _mask_from_rect(80, 70, 180, 160)
    _save(split, stem, before, after, mask)


def scene_road(split: str, stem: str) -> None:
    """Vegetation → road construction (horizontal cut)."""
    before = _base_terrain("green")
    after = _draw_line(before, 20, 128, 236, 128, thickness=22, color=[180, 165, 130])
    mask = _mask_from_line(20, 128, 236, 128, thickness=22)
    _save(split, stem, before, after, mask)


def scene_water(split: str, stem: str) -> None:
    """Water body shrinks (right half dries)."""
    before = _base_terrain("green")
    before = _draw_rect(before, 0, 0, SIZE, SIZE // 2, [60, 100, 160])  # water top half
    after = _base_terrain("green")
    after = _draw_rect(after, 0, 0, SIZE // 2, SIZE // 2, [60, 100, 160])  # water shrinks
    # changed = right half of water region now shows dry land
    mask = _mask_from_rect(SIZE // 2, 0, SIZE, SIZE // 2)
    _save(split, stem, before, after, mask)


def scene_urban_infill(split: str, stem: str) -> None:
    """Small urban infill: empty parcel → low structure."""
    before = _base_terrain("tan")
    after = _draw_rect(before, 60, 60, 130, 130, [100, 95, 88])
    after = _draw_rect(after, 75, 75, 115, 115, [200, 195, 185])
    mask = _mask_from_rect(60, 60, 130, 130)
    _save(split, stem, before, after, mask)


def main() -> None:
    print(f"Writing sample data to {VISION_ROOT}")
    scene_building("train", "scene_001")
    scene_road("train",    "scene_002")
    scene_water("train",   "scene_003")
    scene_urban_infill("val", "scene_001")
    print("Done.")


if __name__ == "__main__":
    main()
