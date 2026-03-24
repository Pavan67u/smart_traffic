#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_finetune.sh [--dry-run] [--device mps] [--epochs 10]
PYTHON=/Users/Pavan/projects/smart_traffic/.venv_mac/bin/python
export PYTHONPATH=/Users/Pavan/projects/smart_traffic

$PYTHON training/finetune.py "$@"
