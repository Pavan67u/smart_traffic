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
                    for detection in detections:
                        # Get bounding box
                        x1, y1, x2, y2 = detection.xyxy[0].tolist()
                        
                        # Convert to center format and normalize
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Get class (all as vehicle class 0 for now)
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
