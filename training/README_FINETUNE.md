# Finetune YOLOv8 (vehicles) – Quick guide

**Prerequisites:** Use a virtualenv with `ultralytics`, `opencv-python`, and `tqdm` (e.g. `pip install -r infra/requirements.txt` or the same deps as the web app). Run all commands from the **repo root**.

**Indian roads & vehicles:** For datasets suited to Indian traffic (two-wheelers, auto-rickshaws, mixed conditions), see **[INDIAN_ROAD_DATASETS.md](INDIAN_ROAD_DATASETS.md)** for a list of public datasets and how to use them.

## 1. Prepare the dataset

**Option A – From videos (recommended first time)**  
Builds a dataset from `training/videos/*.mp4`: extract frames → auto-label → train/val/test split.

From repo root:

```bash
export PYTHONPATH=/Users/Pavan/projects/smart_traffic
python training/prepare_dataset.py
```

Options: `--every 10` (frames), `--out data/vehicles_yolo`, `--train 0.8 --val 0.15`, `--model yolov8n.pt`, `--conf 0.4`. Use `--skip-label` only if you already have labels in the staging folder.

**Option B – You already have YOLO-format data**  
Put images and labels in this layout (dataset root = `data/vehicles_yolo` by default):

- `train/images/`, `train/labels/`
- `val/images/`, `val/labels/`
- `test/images/`, `test/labels/` (optional)

Class names must match `training/yolo/data_vehicles.yaml`: `[car, bus, truck, motorbike, person]`.

## 2. Dry run (check config and model load)

```bash
# From repo root
PYTHONPATH=/Users/Pavan/projects/smart_traffic python training/finetune.py --dry-run
# or
./scripts/run_finetune.sh --dry-run
```

## 3. Run training

```bash
# Example: 30 epochs, batch 16, MPS (Apple Silicon) or GPU
./scripts/run_finetune.sh --epochs 30 --batch 16 --device mps

# CPU only (slower)
./scripts/run_finetune.sh --epochs 20 --batch 8 --device cpu

# With TensorBoard
./scripts/run_finetune.sh --epochs 50 --device mps --tensorboard
# Then open http://127.0.0.1:6006
```

Finetune script options: `--model`, `--data`, `--epochs`, `--batch`, `--device`, `--project`, `--name`, `--tensorboard`. After training it runs validation and exports the best model to ONNX.

## 4. Use the finetuned model

Weights are saved under `runs/finetune/<name>/weights/best.pt`. To use in the web app:

```bash
export MODEL_PATH=/Users/Pavan/projects/smart_traffic/runs/finetune/finetune_<timestamp>/weights/best.pt
python web_app/app.py
```

Or copy `best.pt` to `models/` and set `MODEL_PATH=models/best.pt`.

---

Notes:
- For long runs use a GPU or MPS; export to ONNX/TensorRT for deployment.
- If your dataset is elsewhere, pass `--data /path/to/data.yaml` and set `path:` in that YAML to your dataset root.
