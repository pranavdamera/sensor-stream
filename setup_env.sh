#!/usr/bin/env bash
# Create a clean Python 3.11 virtual environment and install all dependencies.
# Run from the repository root:
#   bash setup_env.sh

set -euo pipefail

VENV_DIR="${1:-.venv}"

echo "==> Creating virtual environment in ${VENV_DIR} ..."
python3.11 -m venv "${VENV_DIR}" || python3 -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip ..."
pip install --quiet --upgrade pip

echo "==> Installing dependencies ..."
pip install --quiet -r requirements.txt

echo ""
echo "==> Running smoke check ..."
python scripts/smoke_check.py

echo ""
echo "Done. Activate with:  source ${VENV_DIR}/bin/activate"
