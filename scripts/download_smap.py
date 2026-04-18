"""Download the telemanom SMAP archive and labeled anomalies CSV."""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELED_URL = (
    "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the download helper."""
    parser = argparse.ArgumentParser(
        description="Download SMAP telemetry arrays and anomaly labels into data/raw."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory where train/, test/, and CSV will be stored.",
    )
    parser.add_argument(
        "--skip-zip",
        action="store_true",
        help="Skip downloading data.zip if already present.",
    )
    return parser.parse_args(argv)


def _download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    """Stream a remote URL to ``destination`` with logging.

    Args:
        url: HTTP or HTTPS URL.
        destination: Output file path (parent directories created as needed).
        chunk_size: Read chunk size in bytes.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip archive into ``target_dir`` (flattening a single top dir).

    Args:
        zip_path: Path to ``data.zip``.
        target_dir: Root folder for ``train/`` and ``test/`` subfolders.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s into %s", zip_path, target_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def main(argv: list[str] | None = None) -> int:
    """Download artifacts required for training and evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = (project_root / args.output_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "data.zip"
    if args.skip_zip and zip_path.is_file():
        logger.info("Using existing archive %s", zip_path)
    else:
        _download_file(DATA_ZIP_URL, zip_path)
    _extract_zip(zip_path, raw_dir)

    labels_path = raw_dir / "labeled_anomalies.csv"
    _download_file(LABELED_URL, labels_path)

    if not (raw_dir / "train").is_dir() or not (raw_dir / "test").is_dir():
        logger.warning(
            "Expected %s/train and %s/test after unzip; contents: %s",
            raw_dir,
            raw_dir,
            [p.name for p in raw_dir.iterdir()],
        )

    logger.info("SMAP data ready under %s", raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
