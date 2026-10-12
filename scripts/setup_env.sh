#!/usr/bin/env bash
# Set up the fxagent virtual environment from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
echo "Using $PYTHON ($($PYTHON --version))"

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

echo "Installing CPU-only torch..."
pip install -q --index-url https://download.pytorch.org/whl/cpu torch

echo "Installing locked dependencies..."
pip install -q -r requirements.lock.txt
pip install -q -r requirements-dev.txt

echo "Installing fxagent in editable mode..."
pip install -q -e . 2>/dev/null || pip install -q -e . --no-build-isolation || true

mkdir -p data/raw data/processed models logs reports
echo "Setup complete. Activate with: source .venv/bin/activate"
