"""Streamlit demo for the Satellite Monitoring ML System.

Tab 1 — Telemetry Anomaly: paste a telemetry window, score it.
Tab 2 — Satellite Change Detection: upload or use sample images, see change overlay.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image as PILImage

API_BASE = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_BEFORE = PROJECT_ROOT / "data/vision/demo_or_dataset/val/before/scene_001.png"
SAMPLE_AFTER  = PROJECT_ROOT / "data/vision/demo_or_dataset/val/after/scene_001.png"
SAMPLE_OVERLAY = PROJECT_ROOT / "outputs/sample_overlay.png"
CKPT_PATH = PROJECT_ROOT / "saved_models/change_detector.pt"

st.set_page_config(page_title="Satellite Monitoring ML", layout="wide")
st.title("Satellite Monitoring ML System")
st.caption(
    "NASA SMAP telemetry anomaly detection · PyTorch satellite image change detection"
)

tab_telemetry, tab_vision = st.tabs(["Telemetry Anomaly", "Satellite Change Detection"])

# ---------------------------------------------------------------------------
# Tab 1 — Telemetry Anomaly
# ---------------------------------------------------------------------------
EXAMPLE_WINDOW = json.dumps(
    {"channel": "P-1", "window": [[float(i % 5) * 0.1] * 25 for i in range(128)]},
    indent=2,
)

with tab_telemetry:
    st.header("Telemetry Anomaly Detection")
    st.markdown(
        "Paste a JSON payload matching `{channel, window}` where `window` is a "
        "`(T, F)` array of multivariate sensor readings. "
        "The API reconstructs the window with an LSTM autoencoder and flags "
        "anomalies where reconstruction error exceeds an adaptive threshold."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        raw_json = st.text_area(
            "Request JSON",
            value=EXAMPLE_WINDOW,
            height=300,
            help="window shape must match the trained model (default T=128, F=25 for P-1)",
        )

    with col_right:
        if st.button("Run Anomaly Detection", key="tele_btn"):
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
                st.stop()

            try:
                import requests
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    anomaly = data["anomaly"]
                    color = "red" if anomaly else "green"
                    st.markdown(f"### Result: :{color}[{'ANOMALY' if anomaly else 'NORMAL'}]")
                    st.metric("Reconstruction Error", f"{data['reconstruction_error']:.6f}")
                    st.metric("Threshold", f"{data['threshold']:.6f}")
                    st.metric("Confidence", f"{data['confidence']:.4f}")
                elif resp.status_code == 404:
                    st.warning(resp.json().get("detail", "Model not found."))
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(
                    f"Could not reach API at {API_BASE}. "
                    f"Start the server with `uvicorn app.main:app --reload`. "
                    f"Error: {exc}"
                )

        st.markdown("---")
        st.markdown("**API endpoint:** `POST /predict`")
        st.code(
            "curl -X POST http://localhost:8000/predict \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"channel\": \"P-1\", \"window\": [[...]]}'",
            language="bash",
        )

# ---------------------------------------------------------------------------
# Tab 2 — Satellite Change Detection
# ---------------------------------------------------------------------------
with tab_vision:
    st.header("Satellite Image Change Detection")
    st.markdown(
        "Upload a **before** and **after** satellite image, or use the built-in "
        "synthetic sample.  A lightweight PyTorch U-Net predicts which pixels "
        "changed between acquisitions."
    )

    # --- Sample image shortcut ---
    use_sample = st.checkbox(
        "Use built-in synthetic sample images",
        value=SAMPLE_BEFORE.exists(),
        disabled=not SAMPLE_BEFORE.exists(),
        help="Requires running `python scripts/create_demo_artifacts.py` first.",
    )

    if use_sample and SAMPLE_BEFORE.exists():
        before_bytes = SAMPLE_BEFORE.read_bytes()
        after_bytes  = SAMPLE_AFTER.read_bytes()
        before_name, after_name = "sample_before.png", "sample_after.png"
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            before_file = st.file_uploader("Before image", type=["png", "jpg", "jpeg"])
        with col_b:
            after_file = st.file_uploader("After image", type=["png", "jpg", "jpeg"])
        if before_file and after_file:
            before_bytes = before_file.getvalue()
            after_bytes  = after_file.getvalue()
            before_name  = before_file.name
            after_name   = after_file.name
        else:
            before_bytes = after_bytes = None  # type: ignore[assignment]

    threshold = st.slider("Change threshold (sigmoid probability)", 0.1, 0.9, 0.5, 0.05)

    if st.button("Detect Changes", key="vision_btn"):
        if not before_bytes or not after_bytes:
            st.warning("Please upload or select sample images.")
            st.stop()

        result = None
        overlay_img = None
        api_ok = False

        # --- Try API ---
        try:
            import requests
            resp = requests.post(
                f"{API_BASE}/vision/change-detect",
                files={
                    "before": (before_name, before_bytes, "image/png"),
                    "after":  (after_name,  after_bytes,  "image/png"),
                },
                params={"threshold": threshold},
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                api_ok = True
            elif resp.status_code == 404:
                st.info("API vision model not loaded — running local inference instead.")
        except Exception:
            st.info("API not reachable — running local inference instead.")

        # --- Local inference fallback ---
        if not api_ok:
            try:
                from vision.inference import (
                    load_change_model,
                    predict_change_mask,
                    save_overlay,
                )
                from vision.model import ChangeDetector
                from vision.transforms import pil_to_tensor

                if CKPT_PATH.is_file():
                    model = load_change_model(CKPT_PATH, device="cpu")
                    is_untrained = False
                else:
                    st.warning(
                        "No checkpoint at `saved_models/change_detector.pt`. "
                        "Run `python scripts/create_demo_artifacts.py` to generate one. "
                        "Showing **untrained prototype** output."
                    )
                    model = ChangeDetector()
                    model.eval()
                    is_untrained = True

                before_img = PILImage.open(BytesIO(before_bytes)).convert("RGB")
                after_img  = PILImage.open(BytesIO(after_bytes)).convert("RGB")
                before_t = pil_to_tensor(before_img).unsqueeze(0)
                after_t  = pil_to_tensor(after_img).unsqueeze(0)
                _, binary = predict_change_mask(model, before_t, after_t, threshold)

                changed_ratio = float(binary.mean())
                result = {
                    "change_detected": changed_ratio > 0.01,
                    "changed_pixel_ratio": changed_ratio,
                    "threshold": threshold,
                    "model_loaded": not is_untrained,
                }

                # Build overlay using PIL only
                after_arr = np.array(after_img.resize((256, 256)), dtype=np.uint8)
                mask_r = np.array(
                    PILImage.fromarray(binary * 255).resize((256, 256), PILImage.NEAREST),
                    dtype=bool,
                )
                ov = after_arr.copy()
                ov[mask_r, 0] = np.clip(after_arr[mask_r, 0].astype(int) + 120, 0, 255)
                ov[mask_r, 1] = np.clip(after_arr[mask_r, 1].astype(int) - 60,  0, 255)
                ov[mask_r, 2] = np.clip(after_arr[mask_r, 2].astype(int) - 60,  0, 255)
                overlay_img = PILImage.fromarray(ov)

            except ImportError as exc:
                st.error(f"Local inference failed — missing dependency: {exc}")
                st.stop()

        # --- Display images ---
        before_display = PILImage.open(BytesIO(before_bytes)).resize((256, 256))
        after_display  = PILImage.open(BytesIO(after_bytes)).resize((256, 256))

        # Use pre-generated overlay if no live overlay (e.g., came from API)
        if overlay_img is None and SAMPLE_OVERLAY.exists() and use_sample:
            overlay_img = PILImage.open(SAMPLE_OVERLAY)

        cols = st.columns(3)
        cols[0].image(before_display, caption="Before", use_container_width=True)
        cols[1].image(after_display,  caption="After",  use_container_width=True)
        if overlay_img is not None:
            cols[2].image(overlay_img, caption="Change Overlay (red = changed)", use_container_width=True)
        else:
            cols[2].info("Overlay not available.")

        # --- Metrics ---
        if result:
            color = "red" if result["change_detected"] else "green"
            st.markdown(
                f"### :{color}[{'CHANGE DETECTED' if result['change_detected'] else 'NO SIGNIFICANT CHANGE'}]"
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Changed Pixel Ratio", f"{result['changed_pixel_ratio']:.4f}")
            m2.metric("Threshold", f"{result['threshold']:.2f}")
            m3.metric("Model Status", "Prototype (untrained)" if not result.get("model_loaded") else "Checkpoint loaded")

            if not result.get("model_loaded"):
                st.caption(
                    "This output is from a randomly initialized model. "
                    "Train on a real dataset (LEVIR-CD, OSCD) for meaningful predictions."
                )

    st.markdown("---")
    st.markdown("**API endpoint:** `POST /vision/change-detect`")
    st.code(
        "curl -X POST http://localhost:8000/vision/change-detect \\\n"
        "  -F 'before=@before.png' -F 'after=@after.png'",
        language="bash",
    )

    if SAMPLE_BEFORE.exists():
        with st.expander("About the sample images"):
            st.markdown(
                "The sample images are **synthetic** — generated programmatically with "
                "PIL to simulate satellite change-detection scenes (building footprint, "
                "road construction, water-body recession). "
                "They are not real satellite imagery. "
                "Run `python scripts/generate_sample_data.py` to regenerate them."
            )
