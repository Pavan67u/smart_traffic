#!/bin/bash
# Helper script to deploy the latest finetuned model

# Find the latest finetune run
LATEST_RUN=$(ls -td runs/finetune/* | grep -v "finetune.py" | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "No finetuning runs found in runs/finetune/"
    exit 1
fi

MODEL_PATH="$LATEST_RUN/weights/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model weights not found at $MODEL_PATH"
    echo "Training might still be in progress or failed."
    exit 1
fi

echo "Found latest model: $MODEL_PATH"
echo "Copying to models/best_finetuned.pt..."

mkdir -p models
cp "$MODEL_PATH" models/best_finetuned.pt

echo "---------------------------------------------------"
echo "Model deployed to: models/best_finetuned.pt"
echo ""
echo "To run the web app with this model:"
echo "export MODEL_PATH=$(pwd)/models/best_finetuned.pt"
echo "python web_app/app.py"
echo "---------------------------------------------------"
