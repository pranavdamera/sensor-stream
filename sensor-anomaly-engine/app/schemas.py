"""Pydantic request and response models for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload for anomaly scoring on a single multivariate window."""

    channel: str = Field(..., description="Telemetry channel id (matches trained model).")
    window: list[list[float]] = Field(
        ...,
        description="Shape ``(window_size, n_features)`` — one row per timestep.",
    )


class PredictResponse(BaseModel):
    """Anomaly decision and reconstruction-error diagnostics."""

    anomaly: bool = Field(..., description="True if reconstruction error exceeds threshold.")
    reconstruction_error: float = Field(
        ...,
        description="Mean squared reconstruction error for the window.",
    )
    threshold: float = Field(..., description="Adaptive threshold used for the channel.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized distance between error and threshold in ``[0, 1]``.",
    )
