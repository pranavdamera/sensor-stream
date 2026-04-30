# Satellite Monitoring ML System

> Telemetry anomaly detection + PyTorch satellite image change detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-containerized-blue?logo=docker)

---

## Overview

Two complementary satellite monitoring pipelines in one repo:

1. **Telemetry anomaly detection** — LSTM autoencoder trained on NASA SMAP spacecraft telemetry. Sliding-window preprocessing, z-score normalization, and adaptive per-channel thresholds; benchmarked against Isolation Forest.

2. **Satellite image change detection** — Lightweight PyTorch U-Net that takes a before/after RGB image pair and predicts a per-pixel binary change mask. ~2M parameters, runs on CPU. Currently a **prototype** — the model is randomly initialized until trained on a real dataset (LEVIR-CD, OSCD, xView2). Synthetic sample images and a demo overlay are included so the inference pipeline can be exercised without external data.

Both pipelines are exposed through a Dockerized FastAPI service and a two-tab Streamlit demo.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Satellite Monitoring ML System                │
│                                                                  │
│  ┌─────────────────────────┐   ┌────────────────────────────┐   │
│  │  Telemetry Pipeline     │   │  Vision Pipeline           │   │
│  │  (TensorFlow / NumPy)   │   │  (PyTorch)                 │   │
│  │                         │   │                            │   │
│  │  NASA SMAP .npy files   │   │  before/after image pair   │   │
│  │         │               │   │         │                  │   │
│  │  Sliding-window crop    │   │  Resize + normalize        │   │
│  │  Z-score normalization  │   │  (ImageNet stats)          │   │
│  │         │               │   │         │                  │   │
│  │  LSTM Autoencoder       │   │  Tiny U-Net                │   │
│  │  (encoder–bottleneck–   │   │  (6-ch input: before‖after)│   │
│  │   decoder)              │   │   skip connections         │   │
│  │         │               │   │         │                  │   │
│  │  Reconstruction error   │   │  Per-pixel change mask     │   │
│  │  vs. adaptive threshold │   │  (sigmoid probability)     │   │
│  └────────────────┬────────┘   └──────────────┬─────────────┘   │
│                   │                            │                  │
│            ┌──────▼────────────────────────────▼──────┐          │
│            │         FastAPI Service  (port 8000)      │          │
│            │   GET  /health                            │          │
│            │   POST /predict                           │          │
│            │   POST /vision/change-detect              │          │
│            └──────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
sensor-stream/
├── app/
│   ├── main.py            # FastAPI app — telemetry + vision endpoints
│   ├── inference.py       # Telemetry model loading + scoring helpers
│   ├── vision_api.py      # Vision model loading + image-bytes inference
│   └── schemas.py         # Pydantic request/response models
├── model/
│   ├── autoencoder.py     # LSTM autoencoder (TensorFlow)
│   ├── threshold.py       # Adaptive threshold calibration
│   └── baseline.py        # Isolation Forest baseline
├── vision/
│   ├── model.py           # ChangeDetector U-Net (PyTorch)
│   ├── dataset.py         # Paired before/after/mask Dataset
│   ├── transforms.py      # PIL-to-tensor (torchvision optional)
│   ├── inference.py       # load / predict / overlay helpers
│   └── metrics.py         # Pixel precision, recall, F1, IoU
├── data/
│   ├── preprocess.py      # Sliding-window + z-score
│   └── vision/
│       ├── README.md      # Dataset layout + download links
│       └── demo_or_dataset/   # Synthetic sample images (committed)
│           ├── train/{before,after,mask}/
│           └── val/{before,after,mask}/
├── demo/
│   └── streamlit_app.py   # Two-tab portfolio demo
├── outputs/
│   └── sample_overlay.png # Pre-generated change overlay (prototype)
├── results/
│   ├── latency.json       # Real CPU latency from benchmark_latency.py
│   └── change_detection_metrics.json  # Prototype baseline (untrained)
├── saved_models/
│   ├── change_detector.pt            # Untrained prototype checkpoint
│   └── change_detector_meta.json
├── scripts/
│   ├── download_smap.py
│   ├── generate_sample_data.py     # Regenerate synthetic PNGs
│   ├── create_demo_artifacts.py    # One-shot: images + checkpoint + artifacts
│   ├── train_change_detector.py
│   ├── evaluate_change_detector.py
│   ├── demo_change_inference.py
│   ├── run_telemetry_experiments.py
│   ├── benchmark_latency.py
│   └── smoke_check.py
├── train.py
├── setup_env.sh           # Clean venv setup
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### Clean virtual environment (recommended)

```bash
bash setup_env.sh          # creates .venv, installs deps, runs smoke check
source .venv/bin/activate
```

### Manual

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke check

```bash
python scripts/smoke_check.py
```

Expected output in a full environment:

```
=== Telemetry pipeline (requires tensorflow) ===
  [OK]   data.preprocess
  [OK]   model.autoencoder
  ...
=== Vision pipeline (requires torch; torchvision optional) ===
  [OK]   vision.model
  [OK]   vision.transforms
  [OK]   vision.inference
  [OK]   vision.dataset
  [OK]   app.vision_api
=== PyTorch forward pass ===
  [OK]   ChangeDetector forward (1,3,256,256)
=== Pixel metrics ===
  [OK]   compute_all_metrics on random masks
============================================
  Passed:            14/14
============================================
All checks passed.
```

`[SKIP]` lines mean a package is missing from your env — not a code bug.

---

## Quick Start — Vision Demo (no dataset required)

The repo ships with synthetic sample images and a pre-generated prototype overlay
so you can run the full pipeline immediately.

```bash
# Generate sample images + prototype checkpoint + all artifacts
python scripts/create_demo_artifacts.py
```

Expected output:

```
[1/5] Generating synthetic sample images …
  Saved train/scene_001  (building footprint)
  Saved train/scene_002  (road construction)
  Saved train/scene_003  (water recession)
  Saved val/scene_001    (urban infill)
[2/5] Saving untrained-prototype checkpoint …
  Saved checkpoint → saved_models/change_detector.pt
[3/5] Generating sample overlay …
  Saved overlay → outputs/sample_overlay.png
[4/5] Benchmarking vision inference latency …
  p50=~150 ms  p95=~185 ms  (CPU, 256×256)
[5/5] Evaluating untrained prototype …
  Saved → results/change_detection_metrics.json
```

> **Important:** `change_detector.pt` is an **untrained prototype** (random
> weights). The overlay and metrics in this mode have no predictive meaning —
> they demonstrate the pipeline, not a trained model.

```bash
# Run Streamlit demo (auto-loads sample images and overlay)
streamlit run demo/streamlit_app.py

# Or exercise the CLI directly
python scripts/demo_change_inference.py \
  --before data/vision/demo_or_dataset/val/before/scene_001.png \
  --after  data/vision/demo_or_dataset/val/after/scene_001.png  \
  --output results/demo_change
```

---

## Telemetry Pipeline

### Download data

```bash
python scripts/download_smap.py
# Places data in data/raw/train/, data/raw/test/, data/raw/labeled_anomalies.csv
```

### Train

```bash
python train.py --channel P-1 --epochs 50 --window 128
# Saves: saved_models/autoencoder_P-1.keras
#        saved_models/threshold_P-1.json
```

### Evaluate

```bash
# Single channel
python scripts/evaluate.py --channel P-1
# → results/metrics_P-1.json

# Multiple channels (must have trained models for each)
python scripts/run_telemetry_experiments.py --channels P-1 E-1 --evaluate-only
# → results/telemetry_summary.json
# → results/telemetry_metrics.csv
```

Metrics: precision, recall, F1, false positive rate, TP/FP/TN/FN for both
LSTM autoencoder and Isolation Forest baseline. Numbers are computed from real
model evaluation — nothing is pre-filled.

---

## Vision Pipeline — Training on a Real Dataset

The `ChangeDetector` model is ready to train. Prepare paired images
(see `data/vision/README.md` for LEVIR-CD, OSCD, xView2 instructions),
then:

```bash
python scripts/train_change_detector.py \
  --data-root data/vision/demo_or_dataset \
  --epochs 20 --batch-size 4 --image-size 256
# → saved_models/change_detector.pt  (replaces prototype)
# → saved_models/change_detector_meta.json

python scripts/evaluate_change_detector.py
# → results/change_detection_metrics.json  (real trained metrics)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Lists loaded channels and vision model status |
| `POST` | `/predict` | Telemetry anomaly score for one window |
| `POST` | `/vision/change-detect` | Before/after image change detection |

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"channel": "P-1", "window": [[0.1, -0.2, ...], ...]}'
```

```json
{
  "anomaly": true,
  "reconstruction_error": 0.047,
  "threshold": 0.031,
  "confidence": 0.91
}
```

### POST /vision/change-detect

```bash
curl -X POST http://localhost:8000/vision/change-detect \
  -F "before=@before.png" \
  -F "after=@after.png" \
  -F "threshold=0.5"
```

```json
{
  "change_detected": false,
  "changed_pixel_ratio": 0.0,
  "threshold": 0.5,
  "model_loaded": true
}
```

Returns `404` with a clear message if no checkpoint is present at startup.

---

## FastAPI server

```bash
uvicorn app.main:app --reload
# Docs: http://localhost:8000/docs
```

## Streamlit demo

```bash
streamlit run demo/streamlit_app.py
```

Tab 1 — Telemetry: paste a window JSON, hit Run, see anomaly score.
Tab 2 — Vision: use built-in sample images or upload your own; overlay shown inline.

## Docker

```bash
docker compose up --build
# API at http://localhost:8000
# saved_models/ is bind-mounted — drop trained checkpoints in without rebuilding
```

---

## Latency Benchmark

```bash
python scripts/benchmark_latency.py
# → results/latency.json
```

Measured on this machine (Apple Silicon, CPU):

```json
{
  "vision": {
    "model": "change_detector (untrained prototype)",
    "image_size": 256,
    "p50_ms": 148.02,
    "p95_ms": 185.60,
    "mean_ms": 151.58,
    "num_runs": 100,
    "device": "cpu"
  }
}
```

Telemetry latency will appear here after training and running the benchmark
with a saved `autoencoder_*.keras` checkpoint.

---

## Reproducibility

Full run from scratch:

```bash
# 1. Environment
bash setup_env.sh && source .venv/bin/activate

# 2. Sanity check
python scripts/smoke_check.py

# 3. Vision demo artifacts (no external data needed)
python scripts/create_demo_artifacts.py

# 4. Telemetry pipeline (requires SMAP download, ~500 MB)
python scripts/download_smap.py
python train.py --channel P-1 --epochs 50
python scripts/evaluate.py --channel P-1

# 5. Vision pipeline (requires real paired dataset)
# See data/vision/README.md for dataset setup
python scripts/train_change_detector.py --epochs 20
python scripts/evaluate_change_detector.py

# 6. Latency benchmark
python scripts/benchmark_latency.py
```

All metrics go to `results/`. Nothing in this README is pre-filled.

---

## Limitations and Future Work

- **Vision model is a prototype.** `change_detector.pt` ships as randomly
  initialized weights. Metrics and overlays from this checkpoint have no
  predictive value. Train on LEVIR-CD or OSCD for meaningful results.
- **Telemetry covers point anomalies only.** Contextual and collective anomaly
  patterns are not modeled.
- **RGB only.** Multispectral (NIR, SWIR) imagery would require `rasterio` and
  additional channel handling.
- **Streaming not implemented.** The API processes one window or one image pair
  at a time; Kafka / WebSocket integration is a natural next step.
- **Vision API returns ratio only.** The `/vision/change-detect` response does
  not include an encoded overlay image (future work).

---

## Datasets

- NASA SMAP: [nsidc.org/data/smap](https://nsidc.org/data/smap) — NASA open data
- LEVIR-CD: CC BY 4.0 — building change detection
- OSCD: CC BY-NC-SA — Sentinel-2 multi-city pairs
- xView2: CC BY-NC 4.0 — disaster damage assessment

See `data/vision/README.md` for layout and download instructions.

---

## License

MIT
