#!/usr/bin/env bash
set -euo pipefail

# Monitor finetune run logs and artifacts
# Usage: ./scripts/monitor_finetune.sh [run_name]
# If run_name omitted, the most recent runs/finetune/* is used.

RUNS_DIR="runs/finetune"

if [ ! -d "$RUNS_DIR" ]; then
  echo "No runs directory found at $RUNS_DIR"
  exit 1
fi

if [ $# -ge 1 ]; then
  RUN_NAME="$1"
else
  RUN_NAME=$(ls -1t "$RUNS_DIR" | head -n1)
fi

RUN_PATH="$RUNS_DIR/$RUN_NAME"

if [ ! -d "$RUN_PATH" ]; then
  echo "Run path not found: $RUN_PATH"
  exit 1
fi

LOG_FILE="$RUN_PATH/train.log"

echo "Monitoring run: $RUN_NAME"
echo "Log file: $LOG_FILE"

echo "Artifacts in $RUN_PATH:"
ls -lh "$RUN_PATH" || true

if [ -f "$LOG_FILE" ]; then
  echo "Tailing log (Ctrl-C to exit):"
  tail -n 200 -f "$LOG_FILE"
else
  echo "Log file not found yet. You can watch the folder for files:" 
  ls -lh "$RUN_PATH"
  echo "Waiting for log file to appear... (press Ctrl-C to exit)"
  while [ ! -f "$LOG_FILE" ]; do sleep 1; done
  tail -n 200 -f "$LOG_FILE"
fi
