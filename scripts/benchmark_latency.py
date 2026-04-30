"""Benchmark inference latency for telemetry and vision models."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_RUNS = 50


def _percentile(values: list[float], p: float) -> float:
    arr = sorted(values)
    idx = max(0, int(len(arr) * p / 100) - 1)
    return arr[idx]


def _bench_telemetry(models_dir: Path, results: dict) -> None:
    keras_files = list(models_dir.glob("autoencoder_*.keras"))
    if not keras_files:
        logger.warning("No telemetry model found in %s — skipping.", models_dir)
        results["telemetry"] = {"status": "missing_artifact"}
        return

    from tensorflow import keras
    from model.threshold import load_threshold_json

    model_path = keras_files[0]
    channel = model_path.stem.removeprefix("autoencoder_")
    threshold_path = models_dir / f"threshold_{channel}.json"
    model = keras.models.load_model(model_path)
    meta = load_threshold_json(threshold_path)
    window_size = int(meta.get("window_size", 128))
    n_features = int(meta.get("n_features", 25))

    dummy = np.random.randn(1, window_size, n_features).astype(np.float32)
    # warm-up
    model.predict(dummy, verbose=0)

    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)

    results["telemetry"] = {
        "model": model_path.name,
        "channel": channel,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "mean_ms": float(np.mean(latencies)),
        "num_runs": N_RUNS,
        "device": "cpu",
    }
    logger.info("Telemetry p50=%.1f ms p95=%.1f ms", results["telemetry"]["p50_ms"], results["telemetry"]["p95_ms"])


def _bench_vision(models_dir: Path, results: dict) -> None:
    ckpt_path = models_dir / "change_detector.pt"
    if not ckpt_path.is_file():
        logger.warning("No change detector checkpoint found — skipping vision benchmark.")
        results["vision"] = {"status": "missing_artifact"}
        return

    import torch
    from vision.inference import load_change_model

    model = load_change_model(ckpt_path, device="cpu")
    dummy_before = torch.randn(1, 3, 256, 256)
    dummy_after = torch.randn(1, 3, 256, 256)

    # warm-up
    with torch.no_grad():
        model(dummy_before, dummy_after)

    latencies = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(dummy_before, dummy_after)
            latencies.append((time.perf_counter() - t0) * 1000)

    results["vision"] = {
        "model": "change_detector.pt",
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "mean_ms": float(np.mean(latencies)),
        "num_runs": N_RUNS,
        "device": "cpu",
    }
    logger.info("Vision p50=%.1f ms p95=%.1f ms", results["vision"]["p50_ms"], results["vision"]["p95_ms"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "saved_models"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    _bench_telemetry(models_dir, results)
    _bench_vision(models_dir, results)

    out_path = results_dir / "latency.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved latency report → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
