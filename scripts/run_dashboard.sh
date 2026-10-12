#!/usr/bin/env bash
# Launch the Streamlit dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
exec streamlit run src/fxagent/dashboard/app.py --server.port 8501 --server.address 127.0.0.1
