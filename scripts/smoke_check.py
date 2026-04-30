"""Smoke check — verifies imports and model forward passes without real data.

Exit codes:
  0 — all checks passed (or only known-missing optional packages failed)
  1 — at least one code-logic check failed

Run from anywhere:
    python scripts/smoke_check.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of invocation directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_MISSING_PACKAGE_ERRORS = (ModuleNotFoundError, ImportError)


def _check(label: str, fn, *, optional_pkg: str | None = None) -> tuple[bool, bool]:
    """Run fn(); return (passed, is_env_gap).

    is_env_gap=True means the failure is a missing optional package, not a code bug.
    """
    try:
        fn()
        print(f"  [OK]   {label}")
        return True, False
    except _MISSING_PACKAGE_ERRORS as exc:
        if optional_pkg and optional_pkg in str(exc):
            print(f"  [SKIP] {label}  (missing: {optional_pkg})")
            return False, True
        print(f"  [FAIL] {label}: {exc}")
        return False, False
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        return False, False


def main() -> int:
    import numpy as np

    results: list[tuple[bool, bool]] = []  # (passed, is_env_gap)

    print("\n=== Telemetry pipeline (requires tensorflow) ===")
    results.append(_check("data.preprocess",  lambda: importlib.import_module("data.preprocess")))
    results.append(_check("model.autoencoder", lambda: importlib.import_module("model.autoencoder"), optional_pkg="tensorflow"))
    results.append(_check("model.threshold",   lambda: importlib.import_module("model.threshold"),   optional_pkg="tensorflow"))
    results.append(_check("model.baseline",    lambda: importlib.import_module("model.baseline"),    optional_pkg="tensorflow"))
    results.append(_check("app.schemas",       lambda: importlib.import_module("app.schemas")))
    results.append(_check("app.inference",     lambda: importlib.import_module("app.inference"),     optional_pkg="tensorflow"))

    print("\n=== Vision pipeline (requires torch; torchvision optional) ===")
    results.append(_check("vision.model",      lambda: importlib.import_module("vision.model")))
    results.append(_check("vision.transforms", lambda: importlib.import_module("vision.transforms")))
    results.append(_check("vision.metrics",    lambda: importlib.import_module("vision.metrics")))
    results.append(_check("vision.inference",  lambda: importlib.import_module("vision.inference")))
    results.append(_check("vision.dataset",    lambda: importlib.import_module("vision.dataset")))
    results.append(_check("app.vision_api",    lambda: importlib.import_module("app.vision_api")))

    print("\n=== PyTorch forward pass ===")

    def _forward():
        import torch
        from vision.model import ChangeDetector
        m = ChangeDetector()
        m.eval()
        before = torch.randn(1, 3, 256, 256)
        after  = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = m(before, after)
        assert out.shape == (1, 1, 256, 256), f"Bad output shape: {out.shape}"

    results.append(_check("ChangeDetector forward (1,3,256,256)", _forward))

    print("\n=== Pixel metrics ===")

    def _metrics():
        from vision.metrics import compute_all_metrics
        rng = np.random.default_rng(0)
        yt = rng.integers(0, 2, (256, 256), dtype=np.uint8)
        yp = rng.integers(0, 2, (256, 256), dtype=np.uint8)
        m = compute_all_metrics(yt, yp)
        assert {"f1", "iou", "precision", "recall"} <= m.keys()

    results.append(_check("compute_all_metrics on random masks", _metrics))

    # -------------------------------------------------------------------------
    n_ok  = sum(p for p, _ in results)
    n_gap = sum(g for _, g in results)
    n_fail = len(results) - n_ok - n_gap
    n_total = len(results)

    print(f"\n{'='*44}")
    print(f"  Passed:            {n_ok}/{n_total}")
    if n_gap:
        print(f"  Skipped (env gap): {n_gap}  ← install requirements.txt to fix")
    if n_fail:
        print(f"  Failed (code bug): {n_fail}  ← fix before committing")
    print(f"{'='*44}")

    if n_fail > 0:
        return 1
    if n_ok == n_total:
        print("All checks passed.")
    else:
        print("Vision pipeline OK. Install full requirements to enable telemetry checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
