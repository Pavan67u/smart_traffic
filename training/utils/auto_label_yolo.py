"""
Auto-label extracted frames using pre-trained YOLOv8 model.
Generates YOLO format annotations for quick dataset creation.
Manual review recommended for edge cases.
"""

import cv2
import os
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def auto_label_images(image_dir, output_dir, model_name='yolov8n.pt', conf_threshold=0.5):
    """
    Automatically label images using pre-trained YOLOv8 model.
    
    Args:
        image_dir: Directory containing images to label
        output_dir: Directory to save YOLO format labels
        model_name: YOLOv8 model size ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l')
        conf_threshold: Confidence threshold (0.0-1.0)
    """
    
    # Load pre-trained model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(image_dir) 
                   if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to label")
    print(f"Using confidence threshold: {conf_threshold}")
    
    labeled_count = 0
    empty_count = 0
    
    # Define the target class names we want in our dataset and their order
    # These should match `training/yolo/data_vehicles.yaml` names order.
    target_names = ['car', 'bus', 'truck', 'motorbike', 'person']

    # Get pretrained model's class name map (if available)
    try:
        pretrained_names = model.names
    except Exception:
        pretrained_names = None

    # Process each image
    for image_file in tqdm(image_files, desc="Auto-labeling images"):
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_file}")
                continue
            
            height, width = image.shape[:2]
            
            # Run inference
            results = model(image_path, conf=conf_threshold, verbose=False)
            
            # Extract detections
            detections = results[0].boxes
            
            # Create label file
            label_file = os.path.join(
                output_dir, 
                Path(image_file).stem + '.txt'
            )
            
            with open(label_file, 'w') as f:
                if len(detections) > 0:
                    # Write detections in YOLO format
                    # Try to read class ids from detections and map to our target classes
                    # We'll iterate by index so we can access detection arrays safely.
                    try:
                        # detection arrays exposed on results[0].boxes
                        xyxy_arr = detections.xyxy.cpu().numpy()
                        conf_arr = None
                        cls_arr = None
                        if hasattr(detections, 'conf'):
                            try:
                                conf_arr = detections.conf.cpu().numpy()
                            except Exception:
                                conf_arr = None
                        if hasattr(detections, 'cls'):
                            try:
                                cls_arr = detections.cls.cpu().numpy().astype(int)
                            except Exception:
                                cls_arr = None
                    except Exception:
                        # Fallback: iterate object-wise
                        xyxy_arr = None
                        cls_arr = None

                    for i, det in enumerate(detections):
                        # Get bounding box
                        if xyxy_arr is not None:
                            x1, y1, x2, y2 = xyxy_arr[i][:4].tolist()
                        else:
                            x1, y1, x2, y2 = det.xyxy[0].tolist()

                        # Convert to center format and normalize
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height

                        # Determine class id: map pretrained class name to our target classes
                        class_id = None
                        pred_cls = None
                        if cls_arr is not None:
                            pred_cls = int(cls_arr[i])
                        else:
                            # try attribute access
                            try:
                                raw = det.cls
                                pred_cls = int(raw[0]) if hasattr(raw, '__len__') else int(raw)
                            except Exception:
                                pred_cls = None

                        if pred_cls is not None and pretrained_names is not None:
                            pred_name = pretrained_names.get(pred_cls, None) if isinstance(pretrained_names, dict) else (pretrained_names[pred_cls] if pred_cls < len(pretrained_names) else None)
                            if pred_name in target_names:
                                class_id = target_names.index(pred_name)
                        # If we couldn't map, default to 0 (car) to preserve previous behavior
                        if class_id is None:
                            class_id = 0

                        # Write to file
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    
                    labeled_count += 1
                else:
                    # Empty image - no detections
                    empty_count += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print(f"Auto-labeling Complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with detections: {labeled_count}")
    print(f"Empty images (no vehicles): {empty_count}")
    print(f"Labels saved to: {output_dir}")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Review labels in output directory")
    print("2. Manually fix incorrect detections if needed")
    print("3. Run: python training/utils/split_datasets.py")


def main():
    parser = argparse.ArgumentParser(
        description='Auto-label images using pre-trained YOLOv8 model'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/vehicles_yolo/train/images',
        help='Directory containing images to label'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/vehicles_yolo/train/labels',
        help='Directory to save YOLO format labels'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
        help='YOLOv8 model size (nano is fastest)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold (0.0-1.0, lower = more detections)'
    )
    
    args = parser.parse_args()
    
    auto_label_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        conf_threshold=args.conf
    )


if __name__ == '__main__':
    main()
