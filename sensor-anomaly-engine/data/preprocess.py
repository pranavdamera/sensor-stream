"""Sliding-window preprocessing and z-score normalization for SMAP telemetry."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_channel_arrays(
    raw_dir: Path,
    channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load train and test numpy arrays for a telemetry channel.

    Arrays may be one-dimensional ``(T,)`` or multivariate ``(T, F)``. Missing
    files raise ``FileNotFoundError`` with paths resolved via ``Path``.

    Args:
        raw_dir: Root directory containing ``train/`` and ``test/`` folders
            (for example ``data/raw`` after unzipping the telemanom archive).
        channel: Channel identifier such as ``P-1`` (matches ``{channel}.npy``).

    Returns:
        Tuple ``(train_array, test_array)`` as ``float32`` numpy arrays.

    Raises:
        FileNotFoundError: If either the train or test file is missing.
    """
    train_path = raw_dir / "train" / f"{channel}.npy"
    test_path = raw_dir / "test" / f"{channel}.npy"
    if not train_path.is_file():
        raise FileNotFoundError(f"Train array not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test array not found: {test_path}")
    train_arr = np.asarray(np.load(train_path), dtype=np.float32)
    test_arr = np.asarray(np.load(test_path), dtype=np.float32)
    if train_arr.ndim == 1:
        train_arr = train_arr.reshape(-1, 1)
    if test_arr.ndim == 1:
        test_arr = test_arr.reshape(-1, 1)
    return train_arr, test_arr


def zscore_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and standard deviation on training data.

    Args:
        train: Training matrix of shape ``(T, F)``.

    Returns:
        Tuple ``(mean, std)`` each of shape ``(F,)``. ``std`` uses ``1e-8``
        floor to avoid division by zero.
    """
    mean = np.mean(train, axis=0, dtype=np.float64)
    std = np.std(train, axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization using precomputed train statistics.

    Args:
        data: Array of shape ``(T, F)``.
        mean: Per-feature means of shape ``(F,)``.
        std: Per-feature standard deviations of shape ``(F,)``.

    Returns:
        Normalized array with the same shape as ``data``.
    """
    return ((data - mean) / std).astype(np.float32)


def _sliding_windows(
    series: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows and their start indices along the time axis.

    Args:
        series: Array of shape ``(T, F)``.
        window_size: Length of each window in timesteps.
        stride: Step between consecutive window starts.

    Returns:
        Tuple ``(windows, starts)`` where ``windows`` has shape
        ``(N, window_size, F)`` and ``starts`` has shape ``(N,)`` with the
        start index of each window in the original series.
    """
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive integers.")
    t_len, n_features = series.shape
    if t_len < window_size:
        raise ValueError(
            f"Series length {t_len} is shorter than window_size {window_size}."
        )
    starts = np.arange(0, t_len - window_size + 1, stride, dtype=np.int64)
    n_windows = starts.shape[0]
    if n_windows == 0:
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    windows = np.empty((n_windows, window_size, n_features), dtype=np.float32)
    for i, start in enumerate(starts):
        windows[i] = series[start : start + window_size]
    return windows, starts


def load_labeled_anomalies(
    csv_path: Path,
    channel: str,
    spacecraft: str = "SMAP",
) -> list[tuple[int, int]]:
    """Parse anomaly index ranges for a channel from telemanom CSV.

    Uses the first matching row when ``chan_id`` and ``spacecraft`` align.

    Args:
        csv_path: Path to ``labeled_anomalies.csv``.
        channel: Channel id (for example ``P-1``).
        spacecraft: Spacecraft label column filter (default ``SMAP``).

    Returns:
        List of ``(start, end)`` inclusive index pairs on the **test** series.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If no matching row is found or parsing fails.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labeled anomalies CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    mask = (frame["chan_id"].astype(str) == channel) & (
        frame["spacecraft"].astype(str) == spacecraft
    )
    rows = frame.loc[mask].drop_duplicates(
        subset=["chan_id", "spacecraft"],
        keep="first",
    )
    if rows.empty:
        raise ValueError(
            f"No labeled anomalies row for channel={channel!r}, "
            f"spacecraft={spacecraft!r} in {csv_path}."
        )
    sequences_raw = rows.iloc[0]["anomaly_sequences"]
    try:
        sequences: Any = ast.literal_eval(str(sequences_raw))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Could not parse anomaly_sequences for {channel}: {sequences_raw!r}"
        ) from exc
    if not isinstance(sequences, list):
        raise ValueError(f"anomaly_sequences must be a list, got {type(sequences)}")
    parsed: list[tuple[int, int]] = []
    for item in sequences:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(x, (int, np.integer)) for x in item)
        ):
            start_i = int(item[0])
            end_i = int(item[1])
            parsed.append((start_i, end_i))
        else:
            raise ValueError(f"Invalid anomaly interval entry: {item!r}")
    return parsed


def _window_overlaps_interval(
    start: int,
    window_size: int,
    interval_lo: int,
    interval_hi: int,
) -> bool:
    """Return whether a window overlaps an inclusive anomaly interval."""
    window_end = start + window_size - 1
    return not (window_end < interval_lo or start > interval_hi)


def build_labeled_windows(
    train: np.ndarray,
    test: np.ndarray,
    window_size: int,
    train_stride: int,
    test_stride: int,
    anomaly_intervals: list[tuple[int, int]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Normalize data, build sliding windows, and label test windows.

    Z-score parameters are fit on ``train`` only; the same scaling is applied
    to ``test``. Training uses ``train_stride``; evaluation windows on the test
    series use ``test_stride``. A test window is labeled anomalous (``1``) if
    it overlaps any inclusive interval in ``anomaly_intervals``.

    Args:
        train: Training matrix ``(T_train, F)``.
        test: Test matrix ``(T_test, F)``.
        window_size: Sliding window length.
        train_stride: Stride for training windows (for example ``1``).
        test_stride: Stride for test windows (for example ``64``).
        anomaly_intervals: Inclusive ``(start, end)`` indices on the **test**
            timeline (as provided in telemanom ``labeled_anomalies.csv``).

    Returns:
        Tuple
        ``(train_windows, test_windows, test_labels, mean, std, test_starts)``
        where ``test_labels`` and ``test_starts`` align row-wise with
        ``test_windows``.
    """
    mean, std = zscore_fit(train)
    train_z = zscore_apply(train, mean, std)
    test_z = zscore_apply(test, mean, std)

    train_windows, _ = _sliding_windows(train_z, window_size, train_stride)
    test_windows, test_starts = _sliding_windows(test_z, window_size, test_stride)

    test_labels = np.zeros(test_windows.shape[0], dtype=np.int32)
    for i, start in enumerate(test_starts):
        for lo, hi in anomaly_intervals:
            if _window_overlaps_interval(int(start), window_size, lo, hi):
                test_labels[i] = 1
                break

    logger.info(
        "Built windows: train=%s, test=%s, positive_test_windows=%s",
        train_windows.shape,
        test_windows.shape,
        int(np.sum(test_labels)),
    )
    return train_windows, test_windows, test_labels, mean, std, test_starts
