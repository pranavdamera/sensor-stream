"""Adaptive threshold calibration from validation reconstruction errors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_thresholds(
    val_errors: np.ndarray,
    percentile: float = 99.0,
) -> dict[str, float]:
    """Compute Gaussian and percentile thresholds from validation errors.

    Gaussian rule: ``mean(errors) + 3 * std(errors)``. Percentile rule uses
    the given percentile of the error distribution (default: 99th).

    Args:
        val_errors: One-dimensional array of per-window reconstruction errors.
        percentile: Percentile in ``(0, 100]`` for the alternative threshold.

    Returns:
        Dictionary with keys ``threshold_gaussian``, ``threshold_percentile``,
        ``mean``, ``std``, and ``percentile``.
    """
    errors = np.asarray(val_errors, dtype=np.float64).ravel()
    if errors.size == 0:
        raise ValueError("val_errors must be non-empty.")
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    gaussian = mean + 3.0 * std
    pct = float(np.percentile(errors, percentile))
    return {
        "threshold_gaussian": float(gaussian),
        "threshold_percentile": float(pct),
        "mean": mean,
        "std": std,
        "percentile": float(percentile),
    }


def save_threshold_json(
    output_path: Path,
    stats: dict[str, Any],
    *,
    primary_key: str = "threshold_gaussian",
) -> None:
    """Save threshold statistics next to the trained model.

    The file includes a top-level ``threshold`` field equal to
    ``stats[primary_key]`` for downstream inference.

    Args:
        output_path: JSON path (for example ``saved_models/threshold_P-1.json``).
        stats: Mapping produced by :func:`compute_thresholds`, optionally
            extended with extra metadata.
        primary_key: Which stat to expose as ``threshold`` for the API.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(stats)
    if primary_key not in payload:
        raise KeyError(f"primary_key {primary_key!r} not in stats.")
    payload["threshold"] = float(payload[primary_key])
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logger.info("Wrote threshold file %s (threshold=%s)", output_path, payload["threshold"])


def load_threshold_json(path: Path) -> dict[str, Any]:
    """Load threshold JSON written by :func:`save_threshold_json`.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary including at least ``threshold``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
