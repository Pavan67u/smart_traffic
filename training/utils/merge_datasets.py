"""
Merge multiple datasets into unified YOLO format for training.
Supports: BDD100K, auto-labeled frames, and custom datasets.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Unified class mapping
UNIFIED_CLASSES = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'motorcycle': 3,
    'pedestrian': 4,
}

# BDD100K to unified mapping
BDD100K_MAPPING = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'motor': 'motorcycle',
    'person': 'pedestrian',
    'rider': 'pedestrian',  # Include riders as pedestrians
    'bike': None,  # Skip bicycles (not in our classes)
    'train': None,  # Skip trains
    'traffic light': None,  # Skip for now
    'traffic sign': None,
    'drivable area': None,
    'lane': None,
}


def convert_bdd100k_to_yolo(
    json_path: str,
    images_dir: str,
    output_dir: str,
    max_images: int = None
) -> Tuple[int, int]:
    """
    Convert BDD100K JSON annotations to YOLO format.

    Returns:
        Tuple of (images_processed, total_boxes)
    """
    print(f"Loading BDD100K annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    images_processed = 0
    total_boxes = 0
    skipped_no_labels = 0

    # Limit images if specified
    if max_images:
        data = data[:max_images]

    print(f"Processing {len(data)} images...")

    for i, img_data in enumerate(data):
        img_name = img_data['name']
        img_path = Path(images_dir) / img_name

        if not img_path.exists():
            continue

        # Get image dimensions (BDD100K is 1280x720)
        img_width = 1280
        img_height = 720

        labels = img_data.get('labels', [])
        yolo_labels = []

        for label in labels:
            category = label.get('category', '')
            unified_class = BDD100K_MAPPING.get(category)

            if unified_class is None:
                continue

            class_id = UNIFIED_CLASSES[unified_class]

            # Get bounding box
            box = label.get('box2d')
            if not box:
                continue

            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']

            # Convert to YOLO format (normalized xywh)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Validate bounds
            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0:
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_labels:
            # Copy image
            dst_img = Path(output_dir) / 'images' / img_name
            shutil.copy2(img_path, dst_img)

            # Save label
            label_name = Path(img_name).stem + '.txt'
            label_path = Path(output_dir) / 'labels' / label_name
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            images_processed += 1
            total_boxes += len(yolo_labels)
        else:
            skipped_no_labels += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data)} images...")

    print(f"  Converted {images_processed} images with {total_boxes} boxes")
    print(f"  Skipped {skipped_no_labels} images (no valid labels)")

    return images_processed, total_boxes


def merge_datasets(
    datasets: List[Dict],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Merge multiple YOLO-format datasets into train/val/test splits.

    Args:
        datasets: List of dicts with 'images_dir' and 'labels_dir' keys
        output_dir: Output directory for merged dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    """
    random.seed(seed)

    # Collect all image-label pairs
    all_pairs = []

    for ds in datasets:
        images_dir = Path(ds['images_dir'])
        labels_dir = Path(ds['labels_dir'])
        source = ds.get('source', 'unknown')

        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists() and label_path.stat().st_size > 0:
                    all_pairs.append({
                        'image': img_path,
                        'label': label_path,
                        'source': source
                    })

    print(f"Total image-label pairs: {len(all_pairs)}")

    # Count by source
    source_counts = {}
    for pair in all_pairs:
        src = pair['source']
        source_counts[src] = source_counts.get(src, 0) + 1

    print("By source:")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

    # Shuffle and split
    random.shuffle(all_pairs)

    n_train = int(len(all_pairs) * train_ratio)
    n_val = int(len(all_pairs) * val_ratio)

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]

    print(f"\nSplit: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    # Copy files
    splits = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}

    for split, pairs in splits.items():
        print(f"Copying {split}...")
        for i, pair in enumerate(pairs):
            # Generate unique name to avoid conflicts
            new_name = f"{pair['source']}_{pair['image'].stem}{pair['image'].suffix}"

            shutil.copy2(pair['image'], f"{output_dir}/{split}/images/{new_name}")
            shutil.copy2(pair['label'], f"{output_dir}/{split}/labels/{pair['source']}_{pair['label'].stem}.txt")

    print(f"\nMerged dataset saved to: {output_dir}")

    return {
        'train': len(train_pairs),
        'val': len(val_pairs),
        'test': len(test_pairs),
        'total': len(all_pairs)
    }


def verify_labels(labels_dir: str) -> Dict:
    """
    Verify label consistency and report statistics.
    """
    stats = {
        'total_files': 0,
        'empty_files': 0,
        'invalid_format': 0,
        'class_counts': {i: 0 for i in range(5)},
        'total_boxes': 0,
        'out_of_bounds': 0,
    }

    for label_path in Path(labels_dir).glob('*.txt'):
        stats['total_files'] += 1

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            stats['empty_files'] += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                stats['invalid_format'] += 1
                continue

            try:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                if cls_id not in range(5):
                    stats['invalid_format'] += 1
                    continue

                stats['class_counts'][cls_id] += 1
                stats['total_boxes'] += 1

                # Check bounds
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    stats['out_of_bounds'] += 1

            except ValueError:
                stats['invalid_format'] += 1

    return stats


if __name__ == "__main__":
    BASE_DIR = "/Users/Pavan/projects/smart_traffic"
    DOWNLOADS = "/Users/Pavan/Downloads"

    # Step 1: Convert BDD100K validation set
    print("=" * 50)
    print("Step 1: Converting BDD100K dataset...")
    print("=" * 50)

    bdd_output = f"{BASE_DIR}/training/bdd100k_converted"

    # Try to find BDD100K images
    bdd_images_dirs = [
        f"{DOWNLOADS}/archive/bdd100k/bdd100k/images/100k/val",
        f"{DOWNLOADS}/archive/bdd100k/bdd100k/images/100k/train",
    ]

    bdd_images_dir = None
    for d in bdd_images_dirs:
        if os.path.exists(d):
            bdd_images_dir = d
            break

    if bdd_images_dir:
        convert_bdd100k_to_yolo(
            json_path=f"{DOWNLOADS}/archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
            images_dir=bdd_images_dir,
            output_dir=bdd_output,
            max_images=2000  # Limit to balance dataset
        )
    else:
        print("BDD100K images not found, skipping...")

    # Step 2: Merge all datasets
    print("\n" + "=" * 50)
    print("Step 2: Merging all datasets...")
    print("=" * 50)

    datasets = []

    # Add our auto-labeled dataset
    if os.path.exists(f"{BASE_DIR}/training/frames"):
        datasets.append({
            'images_dir': f"{BASE_DIR}/training/frames",
            'labels_dir': f"{BASE_DIR}/training/labels",
            'source': 'custom'
        })

    # Add BDD100K if converted
    if os.path.exists(f"{bdd_output}/images"):
        datasets.append({
            'images_dir': f"{bdd_output}/images",
            'labels_dir': f"{bdd_output}/labels",
            'source': 'bdd100k'
        })

    if datasets:
        merged_output = f"{BASE_DIR}/training/merged_dataset"
        stats = merge_datasets(datasets, merged_output)

        # Step 3: Verify consistency
        print("\n" + "=" * 50)
        print("Step 3: Verifying label consistency...")
        print("=" * 50)

        for split in ['train', 'val', 'test']:
            labels_dir = f"{merged_output}/{split}/labels"
            if os.path.exists(labels_dir):
                print(f"\n{split.upper()}:")
                verify_stats = verify_labels(labels_dir)
                print(f"  Total files: {verify_stats['total_files']}")
                print(f"  Empty files: {verify_stats['empty_files']}")
                print(f"  Invalid format: {verify_stats['invalid_format']}")
                print(f"  Out of bounds: {verify_stats['out_of_bounds']}")
                print(f"  Total boxes: {verify_stats['total_boxes']}")
                print("  Class distribution:")
                class_names = ['car', 'truck', 'bus', 'motorcycle', 'pedestrian']
                for i, name in enumerate(class_names):
                    print(f"    {name}: {verify_stats['class_counts'][i]}")

        # Create data.yaml
        print("\n" + "=" * 50)
        print("Step 4: Creating data.yaml...")
        print("=" * 50)

        yaml_content = f"""# Merged Traffic Detection Dataset
path: {merged_output}
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: pedestrian
"""
        with open(f"{merged_output}/data.yaml", 'w') as f:
            f.write(yaml_content)

        print(f"Created: {merged_output}/data.yaml")
        print("\nDone!")
    else:
        print("No datasets found to merge!")
