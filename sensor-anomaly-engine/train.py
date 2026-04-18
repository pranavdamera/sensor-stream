"""Train the LSTM autoencoder and persist model plus adaptive thresholds."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from data.preprocess import (
    build_labeled_windows,
    load_channel_arrays,
    load_labeled_anomalies,
)
from model.autoencoder import build_lstm_autoencoder
from model.threshold import compute_thresholds, save_threshold_json

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train LSTM autoencoder on SMAP sliding windows."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="P-1",
        help="Telemetry channel id (matches train/test npy basename).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for the autoencoder.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="Sliding window size in timesteps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Directory for ``.keras`` model and threshold JSON.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root folder with train/, test/, and labeled_anomalies.csv.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Trailing fraction of train windows used for threshold calibration.",
    )
    return parser.parse_args(argv)


def _mse_per_window(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute mean squared error for each window sample.

    Args:
        y_true: Ground truth windows ``(N, T, F)``.
        y_pred: Reconstructed windows, same shape as ``y_true``.

    Returns:
        One-dimensional array of length ``N`` with per-window MSE.
    """
    diff = (y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2
    return np.mean(diff, axis=(1, 2))


def _split_train_val(
    train_windows: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split train windows chronologically into train and validation sets.

    Args:
        train_windows: Array ``(N, T, F)`` in time order.
        val_fraction: Fraction of windows reserved for validation (``0, 1``).

    Returns:
        Tuple ``(train_part, val_part)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive).")
    n = train_windows.shape[0]
    if n < 2:
        raise ValueError("Need at least two train windows to split.")
    split = int(np.floor(n * (1.0 - val_fraction)))
    split = max(1, min(split, n - 1))
    return train_windows[:split], train_windows[split:]


def main(argv: list[str] | None = None) -> int:
    """Entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parent
    raw_dir = (project_root / args.data_root).resolve()
    labels_csv = raw_dir / "labeled_anomalies.csv"
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading channel %s from %s", args.channel, raw_dir)
    train_arr, test_arr = load_channel_arrays(raw_dir, args.channel)
    intervals = load_labeled_anomalies(labels_csv, args.channel)

    train_windows, test_windows, _, mean, std, _ = build_labeled_windows(
        train_arr,
        test_arr,
        window_size=args.window,
        train_stride=1,
        test_stride=64,
        anomaly_intervals=intervals,
    )

    train_part, val_part = _split_train_val(train_windows, args.val_fraction)
    logger.info(
        "Train/val split: train_windows=%s val_windows=%s",
        train_part.shape[0],
        val_part.shape[0],
    )

    n_features = int(train_part.shape[2])
    model = build_lstm_autoencoder(args.window, n_features)
    history = model.fit(
        train_part,
        train_part,
        epochs=args.epochs,
        batch_size=64,
        validation_data=(val_part, val_part),
        verbose=0,
    )
    final_loss = float(history.history["loss"][-1])
    logger.info("Final training loss (MSE): %.6f", final_loss)

    val_pred = model.predict(val_part, verbose=0)
    val_errors = _mse_per_window(val_part, val_pred)
    stats = compute_thresholds(val_errors, percentile=99.0)
    stats["channel"] = args.channel
    stats["window_size"] = float(args.window)
    stats["n_features"] = float(n_features)
    stats["mean_vector"] = mean.tolist()
    stats["std_vector"] = std.tolist()

    model_path = output_dir / f"autoencoder_{args.channel}.keras"
    threshold_path = output_dir / f"threshold_{args.channel}.json"
    model.save(model_path)
    save_threshold_json(threshold_path, stats, primary_key="threshold_gaussian")
    logger.info("Saved model to %s", model_path)
    logger.info("Saved thresholds to %s", threshold_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
