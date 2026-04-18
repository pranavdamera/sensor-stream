"""Model loading helpers and reconstruction-based scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

from model.threshold import load_threshold_json

logger = logging.getLogger(__name__)


def normalize_window(
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization using training statistics.

    Args:
        window: Array of shape ``(T, F)``.
        mean: Per-feature means ``(F,)``.
        std: Per-feature standard deviations ``(F,)``.

    Returns:
        Normalized window with the same shape as ``window``.
    """
    return ((window - mean) / std).astype(np.float32)


def reconstruction_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute scalar MSE between two matching windows.

    Args:
        y_true: Ground truth ``(1, T, F)`` or ``(T, F)``.
        y_pred: Predictions matching ``y_true`` shape.

    Returns:
        Mean squared error averaged over time and features.
    """
    if y_true.ndim == 3:
        y_true = y_true[0]
    if y_pred.ndim == 3:
        y_pred = y_pred[0]
    diff = (y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2
    return float(np.mean(diff))


def anomaly_confidence(error: float, threshold: float) -> float:
    """Map absolute gap between error and threshold to ``[0, 1]``.

    Uses ``abs(error - threshold) / max(threshold, eps)`` clipped to one.

    Args:
        error: Scalar reconstruction error.
        threshold: Decision threshold for the channel.

    Returns:
        Confidence score in ``[0, 1]``.
    """
    denom = max(float(threshold), 1e-8)
    return float(min(1.0, abs(float(error) - float(threshold)) / denom))


def load_channel_bundle(
    models_dir: Path,
    channel: str,
) -> tuple[keras.Model, dict[str, Any]]:
    """Load the Keras model and threshold JSON for a channel.

    Args:
        models_dir: Directory containing ``autoencoder_{channel}.keras`` and
            ``threshold_{channel}.json``.
        channel: Channel identifier.

    Returns:
        Tuple ``(model, threshold_dict)``.

    Raises:
        FileNotFoundError: If either artifact is missing.
    """
    model_path = models_dir / f"autoencoder_{channel}.keras"
    threshold_path = models_dir / f"threshold_{channel}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found for channel {channel}: {model_path}")
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"Threshold file not found for channel {channel}: {threshold_path}"
        )
    model = keras.models.load_model(model_path)
    meta = load_threshold_json(threshold_path)
    return model, meta


def predict_window_anomaly(
    model: keras.Model,
    meta: dict[str, Any],
    window: list[list[float]],
) -> tuple[bool, float, float, float]:
    """Score a single window: anomaly flag, error, threshold, confidence.

    Applies z-score normalization using ``mean_vector`` and ``std_vector`` from
    ``meta`` when present; otherwise assumes the window is already normalized.

    Args:
        model: Loaded Keras autoencoder.
        meta: Threshold JSON contents including ``threshold`` and optional
            ``mean_vector`` / ``std_vector``.
        window: Nested list shaped ``(T, F)``.

    Returns:
        Tuple ``(anomaly, reconstruction_error, threshold, confidence)``.
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"window must be 2-D, got shape {arr.shape}")
    if "mean_vector" in meta and "std_vector" in meta:
        mean = np.asarray(meta["mean_vector"], dtype=np.float32)
        std = np.asarray(meta["std_vector"], dtype=np.float32)
        arr = normalize_window(arr, mean, std)
    batch = np.expand_dims(arr, axis=0)
    pred = model.predict(batch, verbose=0)
    error = reconstruction_mse(batch, pred)
    threshold = float(meta["threshold"])
    is_anomaly = error > threshold
    confidence = anomaly_confidence(error, threshold)
    return is_anomaly, error, threshold, confidence
