"""Evaluate LSTM autoencoder vs Isolation Forest on held-out test windows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow import keras

from data.preprocess import (
    build_labeled_windows,
    load_channel_arrays,
    load_labeled_anomalies,
)
from model.baseline import fit_isolation_forest, predict_isolation_forest
from model.threshold import load_threshold_json

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved autoencoder and Isolation Forest baselines."
    )
    parser.add_argument("--channel", type=str, default="P-1", help="Channel id.")
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="Sliding window size (must match training).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Directory with train/, test/, labeled_anomalies.csv.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="saved_models",
        help="Directory with autoencoder and threshold artifacts.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for metrics JSON output.",
    )
    return parser.parse_args(argv)


def _mse_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute per-window MSE for batches ``(N, T, F)``.

    Args:
        y_true: Ground truth windows.
        y_pred: Reconstructed windows.

    Returns:
        One-dimensional array of length ``N``.
    """
    diff = (y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2
    return np.mean(diff, axis=(1, 2))


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for binary labels.

    Args:
        y_true: Integer ground-truth labels ``{0,1}``.
        y_pred: Integer predictions ``{0,1}``.

    Returns:
        Dictionary with ``precision``, ``recall``, ``f1``.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _log_metrics_table(
    lstm_metrics: dict[str, float],
    if_metrics: dict[str, float],
) -> None:
    """Emit a comparison table via logging (stdout)."""
    logger.info("%-18s %10s %10s %10s", "Model", "Precision", "Recall", "F1")
    logger.info(
        "%-18s %10.4f %10.4f %10.4f",
        "LSTM Autoencoder",
        lstm_metrics["precision"],
        lstm_metrics["recall"],
        lstm_metrics["f1"],
    )
    logger.info(
        "%-18s %10.4f %10.4f %10.4f",
        "Isolation Forest",
        if_metrics["precision"],
        if_metrics["recall"],
        if_metrics["f1"],
    )


def main(argv: list[str] | None = None) -> int:
    """Load artifacts, score test windows, and persist metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = (project_root / args.data_root).resolve()
    models_dir = (project_root / args.models_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    train_arr, test_arr = load_channel_arrays(raw_dir, args.channel)
    intervals = load_labeled_anomalies(raw_dir / "labeled_anomalies.csv", args.channel)
    train_windows, test_windows, y_true, _, _, _ = build_labeled_windows(
        train_arr,
        test_arr,
        window_size=args.window,
        train_stride=1,
        test_stride=64,
        anomaly_intervals=intervals,
    )

    model_path = models_dir / f"autoencoder_{args.channel}.keras"
    threshold_path = models_dir / f"threshold_{args.channel}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    model: keras.Model = keras.models.load_model(model_path)
    meta: dict[str, Any] = load_threshold_json(threshold_path)
    threshold = float(meta["threshold"])

    preds = model.predict(test_windows, verbose=0)
    errors = _mse_matrix(test_windows, preds)
    lstm_labels = (errors > threshold).astype(np.int32)

    estimator = fit_isolation_forest(train_windows)
    if_labels = predict_isolation_forest(estimator, test_windows)

    lstm_metrics = _classification_metrics(y_true, lstm_labels)
    if_metrics = _classification_metrics(y_true, if_labels)

    _log_metrics_table(lstm_metrics, if_metrics)

    payload: dict[str, Any] = {
        "channel": args.channel,
        "window_size": args.window,
        "threshold": threshold,
        "n_test_windows": int(test_windows.shape[0]),
        "lstm_autoencoder": lstm_metrics,
        "isolation_forest": if_metrics,
    }
    out_path = results_dir / f"metrics_{args.channel}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logger.info("Wrote metrics to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
