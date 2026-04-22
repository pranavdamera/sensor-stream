# Sensor-Stream Anomaly Detection Engine

> Multivariate anomaly detection for continuous sensor telemetry using LSTM autoencoders, benchmarked against classical baselines and deployed as a FastAPI inference service.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi) ![Docker](https://img.shields.io/badge/Docker-containerized-blue?logo=docker) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

This project builds an end-to-end anomaly detection pipeline for multivariate time-series sensor data. It was developed using the **NASA SMAP (Soil Moisture Active Passive)** public telemetry dataset — 427,000+ time-series samples across 55 telemetry channels — as a realistic stand-in for continuous sensor streams like those found in IoT, health monitoring, and industrial systems.

The core model is an **LSTM autoencoder** that learns normal sensor behavior during training and flags anomalies via reconstruction error at inference time. It is benchmarked against Isolation Forest and One-Class SVM baselines and deployed as a containerized REST API.

---

## Architecture

```
Raw Sensor Stream
       │
       ▼
┌─────────────────────┐
│  Preprocessing      │  Sliding window (128 steps), z-score normalization,
│  & Feature Eng.     │  adaptive threshold calibration
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LSTM Autoencoder   │  Encoder → bottleneck → Decoder
│  (TensorFlow)       │  Trained on normal sequences only
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Anomaly Scoring    │  Reconstruction error vs. adaptive threshold
│  & Thresholding     │  → binary anomaly label + confidence score
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FastAPI Service    │  POST /predict → JSON anomaly report
│  (Dockerized)       │  Sub-40ms per-window inference latency
└─────────────────────┘
```

---

## Results

| Model              | Precision | Recall | F1   |
|--------------------|-----------|--------|------|
| LSTM Autoencoder   | 0.83      | 0.79   | **0.81** |
| Isolation Forest   | 0.71      | 0.65   | 0.68 |
| One-Class SVM      | 0.67      | 0.61   | 0.64 |

> Evaluated via 5-fold cross-validation on held-out SMAP test channels. Adaptive thresholding reduced false positive rate by 23% vs. static baselines.

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Setup

```bash
git clone https://github.com/pranavdamera/sensor-stream
cd sensor-stream
pip install -r requirements.txt

# Download and preprocess SMAP dataset
python scripts/download_smap.py
python scripts/preprocess.py

# Train the LSTM autoencoder
python train.py --epochs 50 --window 128

# Run inference API locally
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t anomaly-engine .
docker run -p 8000:8000 anomaly-engine
```

### API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "P-1",
    "window": [0.12, -0.34, 0.89, ...]   # 128 timesteps
  }'
```

```json
{
  "anomaly": true,
  "reconstruction_error": 0.047,
  "threshold": 0.031,
  "confidence": 0.91
}
```

---

## Project Structure

```
sensor-anomaly-engine/
├── app/
│   ├── main.py          # FastAPI app
│   └── schemas.py       # Request/response models
├── model/
│   ├── autoencoder.py   # LSTM autoencoder architecture
│   └── threshold.py     # Adaptive threshold logic
├── scripts/
│   ├── download_smap.py
│   └── preprocess.py
├── train.py
├── evaluate.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Why This Matters

The same pipeline architecture powers real-world systems I've built — continuous ECG monitoring (Avhita Health) and NIR spectral classification (GoldenFlow Labs). This project generalizes that sensor-stream + inference API pattern to a public, reproducible benchmark.

---

## Dataset

NASA SMAP: [https://nsidc.org/data/smap](https://nsidc.org/data/smap) — publicly available under NASA's open data policy. No proprietary data is used.

---

## License

MIT
