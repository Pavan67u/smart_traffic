# Saved model artifacts

This directory stores exported/packaged model artifacts so you can run inference later without re-running training.

Files placed here by the project automation:

- `best.pt` — Ultralytics / PyTorch checkpoint saved by the trainer (optimizer stripped).
- `best.onnx` — ONNX export of the best model (if export succeeded during training).

Quick run (from project root) using the helper script:

```bash
# activate the project venv
source .venv_mac/bin/activate

# run predictions with the saved model on the test split
python scripts/run_saved_model.py --model models/best.pt --source data/vehicles_yolo/test/images --out runs/detect/predict_saved
```

Notes:
- The helper script requires `ultralytics` to be installed in the active Python environment.
- You can also load `best.pt` directly using Ultralytics/torch in your own code.
