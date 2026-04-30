"""Run multi-channel telemetry evaluation and write summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate telemetry anomaly detection across channels.")
    p.add_argument("--channels", nargs="+", default=["P-1"], help="Channel ids to evaluate.")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--models-dir", default="saved_models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--window", type=int, default=128)
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training; only evaluate existing model artifacts.",
    )
    return p.parse_args(argv)


def _evaluate_channel(
    channel: str,
    raw_dir: Path,
    models_dir: Path,
    window: int,
) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support
    from tensorflow import keras

    from data.preprocess import build_labeled_windows, load_channel_arrays, load_labeled_anomalies
    from model.baseline import fit_isolation_forest, predict_isolation_forest
    from model.threshold import load_threshold_json

    model_path = models_dir / f"autoencoder_{channel}.keras"
    threshold_path = models_dir / f"threshold_{channel}.json"

    if not model_path.is_file():
        return {"channel": channel, "status": "missing_model"}
    if not threshold_path.is_file():
        return {"channel": channel, "status": "missing_threshold"}

    try:
        train_arr, test_arr = load_channel_arrays(raw_dir, channel)
        intervals = load_labeled_anomalies(raw_dir / "labeled_anomalies.csv", channel)
    except (FileNotFoundError, ValueError) as exc:
        return {"channel": channel, "status": f"data_error: {exc}"}

    train_windows, test_windows, y_true, _, _, _ = build_labeled_windows(
        train_arr, test_arr,
        window_size=window,
        train_stride=1,
        test_stride=64,
        anomaly_intervals=intervals,
    )

    model = keras.models.load_model(model_path)
    meta = load_threshold_json(threshold_path)
    threshold = float(meta["threshold"])

    preds = model.predict(test_windows, verbose=0)
    errors = np.mean((test_windows.astype(np.float64) - preds.astype(np.float64)) ** 2, axis=(1, 2))
    lstm_labels = (errors > threshold).astype(np.int32)

    estimator = fit_isolation_forest(train_windows)
    if_labels = predict_isolation_forest(estimator, test_windows)

    def _metrics(y_true, y_pred):
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        yt, yp = y_true.astype(bool), y_pred.astype(bool)
        tp = int(np.sum(yt & yp))
        fp = int(np.sum(~yt & yp))
        tn = int(np.sum(~yt & ~yp))
        fn = int(np.sum(yt & ~yp))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "false_positive_rate": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    return {
        "channel": channel,
        "status": "ok",
        "n_test_windows": int(test_windows.shape[0]),
        "n_anomaly_windows": int(y_true.sum()),
        "threshold": threshold,
        "lstm_autoencoder": _metrics(y_true, lstm_labels),
        "isolation_forest": _metrics(y_true, if_labels),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = _parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = (project_root / args.data_root).resolve()
    models_dir = (project_root / args.models_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for channel in args.channels:
        logger.info("Evaluating channel %s …", channel)
        result = _evaluate_channel(channel, raw_dir, models_dir, args.window)
        summary.append(result)
        logger.info("  %s: %s", channel, result.get("status", "?"))

    json_path = results_dir / "telemetry_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary → %s", json_path)

    # Write flat CSV for quick review
    csv_path = results_dir / "telemetry_metrics.csv"
    rows = []
    for entry in summary:
        if entry.get("status") != "ok":
            rows.append({"channel": entry["channel"], "status": entry.get("status", "?")})
            continue
        for model_key in ("lstm_autoencoder", "isolation_forest"):
            m = entry.get(model_key, {})
            rows.append({
                "channel": entry["channel"],
                "model": model_key,
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "false_positive_rate": m.get("false_positive_rate"),
                "tp": m.get("tp"),
                "fp": m.get("fp"),
                "tn": m.get("tn"),
                "fn": m.get("fn"),
            })

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved CSV → %s", csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
