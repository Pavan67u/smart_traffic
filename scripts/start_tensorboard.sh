#!/usr/bin/env bash
set -euo pipefail

# Start TensorBoard for runs/ directory on port 6006
LOGDIR=${1:-runs}
PORT=${2:-6006}

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "tensorboard not found in PATH. Install with: pip install tensorboard"
  exit 1
fi

echo "Starting TensorBoard for ${LOGDIR} on http://127.0.0.1:${PORT}"
exec tensorboard --logdir "${LOGDIR}" --host 127.0.0.1 --port "${PORT}"
