"""
Auto-label images using pretrained YOLOv8 model.
Maps COCO classes to our custom classes.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

# COCO class IDs for vehicles/pedestrians
COCO_TO_CUSTOM = {
    2: 0,   # car -> car
    7: 1,   # truck -> truck
    5: 2,   # bus -> bus
    3: 3,   # motorcycle -> motorcycle
    0: 4,   # person -> pedestrian
}

CUSTOM_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'pedestrian']


def auto_label(images_dir, labels_dir, model_name='yolov8s.pt', conf=0.25):
    """
    Auto-label images using pretrained YOLO model.

    Args:
        images_dir: Directory containing images
        labels_dir: Output directory for YOLO format labels
        model_name: Pretrained model to use (yolov8n/s/m/l/x.pt)
        conf: Confidence threshold
    """
    os.makedirs(labels_dir, exist_ok=True)

    # Load pretrained model
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Get all images
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in Path(images_dir).iterdir()
              if f.suffix.lower() in img_extensions]
    images.sort()

    print(f"Found {len(images)} images to label")
    print(f"Confidence threshold: {conf}")
    print(f"Output: {labels_dir}")
    print("-" * 50)

    labeled = 0
    total_boxes = 0

    for i, img_path in enumerate(images):
        # Run inference
        results = model(str(img_path), verbose=False, conf=conf)

        # Process detections
        label_lines = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])

                # Skip if not in our target classes
                if cls_id not in COCO_TO_CUSTOM:
                    continue

                # Map to custom class ID
                custom_cls = COCO_TO_CUSTOM[cls_id]

                # Get normalized xywh coordinates
                x_center, y_center, width, height = box.xywhn[0].tolist()

                # YOLO format: class x_center y_center width height
                label_lines.append(f"{custom_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save label file
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

        if label_lines:
            labeled += 1
            total_boxes += len(label_lines)

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f"Processed {i + 1}/{len(images)} images | {total_boxes} boxes detected")

    print("-" * 50)
    print(f"Done! Labeled {labeled}/{len(images)} images with {total_boxes} total boxes")
    print(f"Labels saved to: {labels_dir}")

    # Print class distribution
    print("\nClass distribution:")
    for cls_name in CUSTOM_CLASSES:
        print(f"  {cls_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images with pretrained YOLO")
    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("labels_dir", help="Output directory for labels")
    parser.add_argument("--model", default="yolov8s.pt", help="Model to use (default: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")

    args = parser.parse_args()
    auto_label(args.images_dir, args.labels_dir, args.model, args.conf)
