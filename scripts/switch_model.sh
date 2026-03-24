#!/bin/bash

# Function to find best weights in a run directory
find_weights() {
    local run_pattern=$1
    # Sort by time, pick latest
    local run_dir=$(ls -td runs/finetune/$run_pattern* 2>/dev/null | head -1)
    if [ -z "$run_dir" ]; then
        echo ""
    else
        echo "$run_dir/weights/best.pt"
    fi
}

MODE=$1

if [ -z "$MODE" ]; then
    echo "Usage: $0 [vehicles|signs]"
    echo "  vehicles : Detects Cars, Buses, Trucks (for Red Light Violations)"
    echo "  signs    : Detects Traffic Signs (Speed Limit, Stop, etc.)"
    exit 1
fi

MODEL_PATH=""

if [ "$MODE" == "vehicles" ]; then
    MODEL_PATH=$(find_weights "vehicles_finetune")
    if [ -z "$MODEL_PATH" ]; then
        # Fallback to standard yolov8n if no finetune
        MODEL_PATH="yolov8n.pt"
    fi
elif [ "$MODE" == "signs" ]; then
    MODEL_PATH=$(find_weights "signs_finetune")
else
    echo "Invalid mode. Use 'vehicles' or 'signs'."
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find model weights for '$MODE'."
    exit 1
fi

echo "------------------------------------------------"
echo "Starting Smart Traffic App"
echo "Mode: $MODE"
echo "Model: $MODEL_PATH"
echo "------------------------------------------------"

export MODEL_PATH=$(pwd)/$MODEL_PATH
PYTHON_BIN=${PYTHON_BIN:-"$(pwd)/.venv_mac/bin/python"}
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=${PYTHON_BIN_FALLBACK:-python3}
fi

PYTHONPATH=$(pwd) "$PYTHON_BIN" web_app/app.py
