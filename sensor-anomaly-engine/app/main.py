"""FastAPI application exposing health and anomaly prediction endpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from tensorflow import keras

from app.inference import load_channel_bundle, predict_window_anomaly
from app.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return the repository root (parent of the ``app`` package)."""
    return Path(__file__).resolve().parent.parent


def _default_models_dir() -> Path:
    """Resolve the persisted model directory under the project root."""
    return _project_root() / "saved_models"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load all channel models and metadata once at startup."""
    env_dir = os.environ.get("SENSOR_MODELS_DIR")
    models_dir = Path(env_dir).resolve() if env_dir else _default_models_dir()
    app.state.models_dir = models_dir
    app.state.models: dict[str, keras.Model] = {}
    app.state.metas: dict[str, dict[str, Any]] = {}
    root = app.state.models_dir
    if not root.is_dir():
        logger.warning("Models directory does not exist yet: %s", root)
    else:
        for path in sorted(root.glob("autoencoder_*.keras")):
            stem = path.stem
            if not stem.startswith("autoencoder_"):
                continue
            channel = stem.removeprefix("autoencoder_")
            threshold_path = root / f"threshold_{channel}.json"
            if not threshold_path.is_file():
                logger.warning(
                    "Skipping %s — missing threshold file %s", path.name, threshold_path
                )
                continue
            try:
                model, meta = load_channel_bundle(root, channel)
                app.state.models[channel] = model
                app.state.metas[channel] = meta
                logger.info("Loaded model bundle for channel %s", channel)
            except OSError as exc:
                logger.error("Failed to load %s: %s", path, exc)
    yield


app = FastAPI(
    title="Sensor Anomaly Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for orchestrators and load balancers."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Run reconstruction-based anomaly detection for one window."""
    channel = payload.channel
    models: dict[str, keras.Model] = app.state.models
    metas: dict[str, dict[str, Any]] = app.state.metas
    if channel not in models or channel not in metas:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained model found for channel {channel!r}. "
                f"Place autoencoder_{channel}.keras and threshold_{channel}.json "
                f"under {app.state.models_dir}."
            ),
        )
    try:
        anomaly, error, threshold, confidence = predict_window_anomaly(
            models[channel],
            metas[channel],
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
