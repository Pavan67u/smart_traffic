#!/usr/bin/env python3
"""
Finetune YOLOv8 model on the project's dataset.

Usage examples:
  # Dry run: checks config and loads model
  PYTHONPATH=/Users/Pavan/projects/smart_traffic .venv_mac/bin/python training/finetune.py --dry-run

  # Full train (example)
  PYTHONPATH=/Users/Pavan/projects/smart_traffic .venv_mac/bin/python training/finetune.py --epochs 50 --batch 16 --device mps

This script is intentionally small and uses Ultralytics API.
"""
import argparse
import os
import time
from pathlib import Path
import subprocess

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_PATH', 'yolov8n.pt'), help='Base model or weights')
    p.add_argument('--data', default='training/yolo/data_vehicles.yaml', help='Path to data YAML')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='cpu', help='Device: cpu, mps, 0, 0,1 etc')
    p.add_argument('--project', default='runs/finetune', help='Ultralytics project folder')
    p.add_argument('--name', default=None, help='Run name. If not set, timestamped name will be used')
    p.add_argument('--dry-run', action='store_true', help='Only validate config and load model')
    p.add_argument('--tensorboard', action='store_true', help='Launch tensorboard for the run (background)')
    return p.parse_args()


def validate_data_yaml(data_yaml_path: Path):
    if not data_yaml_path.exists():
        print(f"Data YAML not found: {data_yaml_path}")
        return False
    # read and do a lightweight parse to check 'path' key
    try:
        import yaml
        with open(data_yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        base = cfg.get('path')
        if not base:
            print(f"No 'path' entry in {data_yaml_path}")
            return False
        base_path = Path(base)
        if not base_path.exists():
            print(f"Dataset path does not exist: {base_path}")
            return False
        # basic check for train/val folders
        train_dir = base_path / cfg.get('train', 'train/images')
        val_dir = base_path / cfg.get('val', 'val/images')
        print(f"Dataset train dir: {train_dir}")
        print(f"Dataset val dir:   {val_dir}")
        return True
    except Exception as e:
        print('Failed to parse data yaml:', e)
        return False


def main():
    args = parse_args()
    args.project = str(Path(args.project))
    args.name = args.name or f"finetune_{int(time.time())}"

    print('Finetune config:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    data_yaml = Path(args.data)
    ok = validate_data_yaml(data_yaml)
    if not ok:
        print('\nAborting: data YAML validation failed. Create dataset or update --data path.')
        return

    print('\nLoading model...')
    model = YOLO(args.model)
    print(f'Loaded model: {args.model}')

    if args.dry_run:
        print('Dry run complete. Exiting.')
        return

    tb_proc = None
    # Optionally launch tensorboard pointing at the project's runs folder
    if args.tensorboard:
        try:
            tb_cmd = [
                'tensorboard',
                '--logdir', str(Path(args.project)),
                '--host', '127.0.0.1',
                '--port', '6006'
            ]
            print('Starting TensorBoard:',' '.join(tb_cmd))
            tb_proc = subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('TensorBoard launched (background) at http://127.0.0.1:6006')
        except Exception as e:
            print('Failed to start TensorBoard:', e)

    print('Starting training...')
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    print('Training finished. Running validation...')
    model.val(data=str(data_yaml))
    print('Exporting best model to ONNX...')
    out = Path(args.project) / args.name
    out.mkdir(parents=True, exist_ok=True)
    try:
        model.export(format='onnx', opset=12, dynamic=True, overwrite=True)
        print('Export complete.')
    except Exception as e:
        print('ONNX export failed:', e)

    # If we started TensorBoard, do not kill it automatically; inform the user
    if tb_proc is not None:
        print('\nTensorBoard is running in background (PID: {})'.format(tb_proc.pid))
        print('To stop it: kill {}'.format(tb_proc.pid))


if __name__ == '__main__':
    main()
