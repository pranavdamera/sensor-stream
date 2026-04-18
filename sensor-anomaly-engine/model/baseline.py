"""Isolation Forest baseline on flattened sliding windows."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    train_windows: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest on flattened training windows.

    Args:
        train_windows: Array of shape ``(N, window_size, n_features)``.
        contamination: Expected proportion of anomalies (sklearn argument).
        random_state: RNG seed for reproducibility.

    Returns:
        Fitted ``IsolationForest`` instance.
    """
    if train_windows.ndim != 3:
        raise ValueError(
            f"train_windows must be 3-D, got shape {train_windows.shape}"
        )
    n_samples = train_windows.shape[0]
    flat = train_windows.reshape(n_samples, -1)
    estimator = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    estimator.fit(flat)
    return estimator


def predict_isolation_forest(
    estimator: IsolationForest,
    windows: np.ndarray,
) -> np.ndarray:
    """Predict binary anomaly labels on window batches.

    Sklearn labels inliers as ``1`` and outliers as ``-1``. This function maps
    outliers to ``1`` (anomaly) and inliers to ``0`` (normal).

    Args:
        estimator: Fitted ``IsolationForest``.
        windows: Array of shape ``(N, window_size, n_features)``.

    Returns:
        Integer array of shape ``(N,)`` with values ``0`` (normal) or ``1``
        (anomaly).
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must be 3-D, got shape {windows.shape}")
    n_samples = windows.shape[0]
    flat = windows.reshape(n_samples, -1)
    raw = estimator.predict(flat)
    return (raw == -1).astype(np.int32)
