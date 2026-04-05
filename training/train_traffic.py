"""
Train YOLOv8 model for Smart Traffic Detection.
Fine-tunes a pretrained model on the custom dataset.
"""

from ultralytics import YOLO
import torch

def train_model():
    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Training on: {device}")

    # Load pretrained model (YOLOv8s - good balance of speed/accuracy)
    model = YOLO('yolov8s.pt')

    # Train the model
    results = model.train(
        data='training/dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        patience=20,  # Early stopping
        save=True,
        project='training/runs',
        name='traffic_detection',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("\nTraining complete!")
    print(f"Best model saved at: training/runs/traffic_detection/weights/best.pt")

    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return model


if __name__ == "__main__":
    train_model()
