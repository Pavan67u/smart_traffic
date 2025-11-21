#!/usr/bin/env bash
# Helper to run the Flask web app using the project venv
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/.venv_mac/bin/activate"
python "$PROJECT_ROOT/web_app/app.py"
