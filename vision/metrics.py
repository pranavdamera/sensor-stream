"""Pixel-level binary change-detection metrics — no sklearn dependency."""

from __future__ import annotations

import numpy as np


def _confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) for flattened binary arrays."""
    yt = y_true.ravel().astype(bool)
    yp = y_pred.ravel().astype(bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    return tp, fp, tn, fn


def pixel_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, _, _ = _confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def pixel_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, _, _, fn = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def pixel_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = pixel_precision(y_true, y_pred)
    r = pixel_recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def pixel_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, _, fn = _confusion(y_true, y_pred)
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, F1, IoU, and confusion matrix counts."""
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "iou": iou,
        "false_positive_rate": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
