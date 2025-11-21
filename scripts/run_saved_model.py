#!/usr/bin/env python3
"""Run inference with a saved Ultralytics YOLO model.

Usage:
  python scripts/run_saved_model.py --model models/best.pt --source data/vehicles_yolo/test/images --out outputs/saved_predict

This script loads a saved `best.pt` (Ultralytics) and runs prediction, saving images and .txt labels.
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run inference with saved YOLO model')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to model (.pt)')
    parser.add_argument('--source', type=str, default='data/vehicles_yolo/test/images', help='Image/video source for prediction')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--out', type=str, default='runs/detect/predict_saved', help='Output project directory (project/name will be created)')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print('Failed to import ultralytics. Make sure to run this inside the project venv where ultralytics is installed.')
        raise

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    print(f'Loading model: {model_path}')
    model = YOLO(str(model_path))
    print(f'Running prediction on source: {args.source} -> saving to: {args.out}')

    # Use Ultralytics predict API: save images and text
    model.predict(source=args.source, conf=args.conf, save=True, save_txt=True, project=args.out, name='from_saved')

    print('Prediction completed.')

if __name__ == '__main__':
    main()
