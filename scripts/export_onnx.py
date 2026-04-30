"""Export the ChangeDetector PyTorch model to ONNX for lightweight CPU serving.

Usage::

    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint saved_models/change_detector.pt \\
                                   --output saved_models/change_detector.onnx

Runtime inference with ONNX Runtime::

    import numpy as np, onnxruntime as ort
    sess = ort.InferenceSession("saved_models/change_detector.onnx")
    logits = sess.run(["change_logits"], {"before": before_np, "after": after_np})[0]
    probs  = 1 / (1 + np.exp(-logits))
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from vision.model import ChangeDetector

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ChangeDetector to ONNX.")
    parser.add_argument("--checkpoint", default="saved_models/change_detector.pt")
    parser.add_argument("--output", default="saved_models/change_detector.onnx")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)

    model = ChangeDetector()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning("No checkpoint at %s -- exporting untrained prototype.", ckpt_path)

    model.eval()
    dummy_before = torch.zeros(1, 3, args.image_size, args.image_size)
    dummy_after  = torch.zeros(1, 3, args.image_size, args.image_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        args=(dummy_before, dummy_after),
        f=str(output_path),
        input_names=["before", "after"],
        output_names=["change_logits"],
        dynamic_axes={
            "before":        {0: "batch_size"},
            "after":         {0: "batch_size"},
            "change_logits": {0: "batch_size"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    out_shape = list(model(dummy_before, dummy_after).shape)
    logger.info("Exported -> %s  |  output shape: %s  |  opset: %d",
                output_path, out_shape, args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())