"""FastAPI application exposing health, telemetry anomaly, and vision endpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from tensorflow import keras

from app.inference import load_channel_bundle, predict_window_anomaly
from app.schemas import PredictRequest, PredictResponse
from app.vision_api import load_vision_model_if_present, run_change_detection

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_models_dir() -> Path:
    return _project_root() / "saved_models"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load all channel models and vision model once at startup."""
    env_dir = os.environ.get("SENSOR_MODELS_DIR")
    models_dir = Path(env_dir).resolve() if env_dir else _default_models_dir()
    app.state.models_dir = models_dir
    app.state.models: dict[str, keras.Model] = {}
    app.state.metas: dict[str, dict[str, Any]] = {}

    if not models_dir.is_dir():
        logger.warning("Models directory does not exist yet: %s", models_dir)
    else:
        for path in sorted(models_dir.glob("autoencoder_*.keras")):
            channel = path.stem.removeprefix("autoencoder_")
            threshold_path = models_dir / f"threshold_{channel}.json"
            if not threshold_path.is_file():
                logger.warning("Skipping %s — missing threshold file", path.name)
                continue
            try:
                model, meta = load_channel_bundle(models_dir, channel)
                app.state.models[channel] = model
                app.state.metas[channel] = meta
                logger.info("Loaded telemetry model for channel %s", channel)
            except OSError as exc:
                logger.error("Failed to load %s: %s", path, exc)

    app.state.vision_model = load_vision_model_if_present(models_dir)
    yield


app = FastAPI(
    title="Satellite Monitoring ML System",
    version="1.1.0",
    description=(
        "Telemetry anomaly detection (LSTM autoencoder) + "
        "satellite image change detection (PyTorch U-Net)."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "telemetry_channels": list(app.state.models.keys()),
        "vision_model_loaded": app.state.vision_model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Reconstruction-based anomaly detection for a telemetry window."""
    channel = payload.channel
    if channel not in app.state.models:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained model for channel {channel!r}. "
                f"Place autoencoder_{channel}.keras and threshold_{channel}.json "
                f"under {app.state.models_dir}."
            ),
        )
    try:
        anomaly, error, threshold, confidence = predict_window_anomaly(
            app.state.models[channel],
            app.state.metas[channel],
            payload.window,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(
        anomaly=bool(anomaly),
        reconstruction_error=float(error),
        threshold=float(threshold),
        confidence=float(confidence),
    )


@app.post("/vision/change-detect")
async def vision_change_detect(
    before: UploadFile = File(..., description="Before satellite image (PNG/JPG)."),
    after: UploadFile = File(..., description="After satellite image (PNG/JPG)."),
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict a change mask from a before/after satellite image pair.

    Returns ``change_detected``, ``changed_pixel_ratio``, ``threshold``,
    and ``model_loaded``.
    """
    if app.state.vision_model is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Vision change-detection model not loaded. "
                "Train and save a checkpoint to saved_models/change_detector.pt "
                "and restart the server."
            ),
        )
    before_bytes = await before.read()
    after_bytes = await after.read()
    try:
        result = run_change_detection(
            app.state.vision_model,
            before_bytes,
            after_bytes,
            threshold=threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}") from exc
    return result
