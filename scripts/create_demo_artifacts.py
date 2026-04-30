"""Generate all demo artifacts from the untrained prototype model.

This script:
  1. Generates synthetic sample images (runs generate_sample_data.py).
  2. Saves an untrained-prototype checkpoint to saved_models/change_detector.pt.
  3. Runs inference on the val scene → outputs/sample_overlay.png.
  4. Measures vision-only inference latency → results/latency.json.
  5. Evaluates the untrained model on val data → results/change_detection_metrics.json
     (clearly labelled as untrained baseline, not a trained result).

Run from the project root:
    python scripts/create_demo_artifacts.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAVED_MODELS = ROOT / "saved_models"
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
VISION_DATA = ROOT / "data" / "vision" / "demo_or_dataset"
VAL_BEFORE = VISION_DATA / "val" / "before" / "scene_001.png"
VAL_AFTER  = VISION_DATA / "val" / "after"  / "scene_001.png"
VAL_MASK   = VISION_DATA / "val" / "mask"   / "scene_001.png"
CKPT = SAVED_MODELS / "change_detector.pt"

N_RUNS = 100


def step_generate_sample_data() -> None:
    print("\n[1/5] Generating synthetic sample images …")
    from scripts.generate_sample_data import main as gen
    gen()


def step_save_untrained_checkpoint() -> None:
    print("\n[2/5] Saving untrained-prototype checkpoint …")
    from vision.model import ChangeDetector
    SAVED_MODELS.mkdir(parents=True, exist_ok=True)
    model = ChangeDetector()
    torch.save({"model_state_dict": model.state_dict()}, CKPT)

    meta = {
        "status": "untrained_prototype",
        "note": (
            "This checkpoint contains randomly initialized weights. "
            "It is provided so the inference pipeline can be demonstrated "
            "end-to-end without a real training run. "
            "Do not treat outputs as meaningful change predictions."
        ),
        "image_size": 256,
        "epochs": 0,
        "trained_on": None,
    }
    with (SAVED_MODELS / "change_detector_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved checkpoint → {CKPT}")


def step_run_overlay() -> None:
    print("\n[3/5] Generating sample overlay …")
    from vision.inference import (
        load_change_model,
        predict_change_mask,
        preprocess_image_pair,
        save_overlay,
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    model = load_change_model(CKPT)
    before_t, after_t = preprocess_image_pair(VAL_BEFORE, VAL_AFTER)
    _, binary = predict_change_mask(model, before_t, after_t, threshold=0.5)

    overlay_path = OUTPUTS / "sample_overlay.png"
    save_overlay(VAL_BEFORE, VAL_AFTER, binary, overlay_path)
    changed_ratio = float(binary.mean())
    print(f"  Changed pixel ratio (untrained): {changed_ratio:.4f}")
    print(f"  Saved overlay → {overlay_path}")


def step_latency() -> None:
    print("\n[4/5] Benchmarking vision inference latency …")
    from vision.model import ChangeDetector

    RESULTS.mkdir(parents=True, exist_ok=True)
    model = ChangeDetector()
    model.eval()
    dummy_before = torch.randn(1, 3, 256, 256)
    dummy_after  = torch.randn(1, 3, 256, 256)

    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(dummy_before, dummy_after)

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(dummy_before, dummy_after)
            latencies.append((time.perf_counter() - t0) * 1000)

    s = sorted(latencies)
    result = {
        "vision": {
            "model": "change_detector (untrained prototype)",
            "image_size": 256,
            "p50_ms": round(s[int(N_RUNS * 0.50)], 2),
            "p95_ms": round(s[int(N_RUNS * 0.95)], 2),
            "mean_ms": round(float(np.mean(latencies)), 2),
            "num_runs": N_RUNS,
            "device": "cpu",
        },
        "telemetry": {"status": "missing_artifact — run train.py first"},
    }
    out = RESULTS / "latency.json"
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  p50={result['vision']['p50_ms']} ms  p95={result['vision']['p95_ms']} ms")
    print(f"  Saved → {out}")


def step_metrics() -> None:
    print("\n[5/5] Evaluating untrained prototype on val split …")
    from vision.inference import load_change_model, predict_change_mask, preprocess_image_pair
    from vision.metrics import compute_all_metrics

    RESULTS.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    model = load_change_model(CKPT)
    before_t, after_t = preprocess_image_pair(VAL_BEFORE, VAL_AFTER)
    _, binary = predict_change_mask(model, before_t, after_t, threshold=0.5)

    mask_gt = np.array(Image.open(VAL_MASK).convert("L").resize((256, 256), Image.NEAREST))
    mask_gt_bin = (mask_gt > 127).astype(np.uint8)

    m = compute_all_metrics(mask_gt_bin, binary)
    changed_ratio = float(binary.mean())

    result = {
        "status": "untrained_prototype",
        "warning": (
            "These metrics are from a randomly initialized model evaluated on one "
            "synthetic val sample. They reflect random baseline performance, not a "
            "trained result. Train the model on a real dataset before citing any numbers."
        ),
        "num_images": 1,
        "changed_pixel_ratio": changed_ratio,
        "threshold": 0.5,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
    }

    out = RESULTS / "change_detection_metrics.json"
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  Metrics (untrained): precision={result['precision']:.3f}  recall={result['recall']:.3f}  f1={result['f1']:.3f}")
    print(f"  Saved → {out}")


def main() -> None:
    print("=== create_demo_artifacts.py ===")
    step_generate_sample_data()
    step_save_untrained_checkpoint()
    step_run_overlay()
    step_latency()
    step_metrics()
    print("\nAll artifacts generated. See results/ and outputs/.")


if __name__ == "__main__":
    main()
