"""
Split dataset into train/val/test folders for YOLO training.
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42
):
    """
    Split images and labels into train/val/test folders.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        output_dir: Output directory for split dataset
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.05)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Get all images with corresponding labels
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []

    for f in Path(images_dir).iterdir():
        if f.suffix.lower() in img_extensions:
            label_file = Path(labels_dir) / f"{f.stem}.txt"
            if label_file.exists():
                # Check if label file is not empty
                if label_file.stat().st_size > 0:
                    images.append(f.name)

    print(f"Found {len(images)} images with valid labels")

    # Shuffle and split
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    print(f"Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    # Copy files
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split, img_list in splits.items():
        print(f"Copying {split}...")
        for img_name in img_list:
            # Copy image
            src_img = Path(images_dir) / img_name
            dst_img = Path(output_dir) / split / 'images' / img_name
            shutil.copy2(src_img, dst_img)

            # Copy label
            label_name = Path(img_name).stem + '.txt'
            src_label = Path(labels_dir) / label_name
            dst_label = Path(output_dir) / split / 'labels' / label_name
            shutil.copy2(src_label, dst_label)

    print(f"\nDataset split complete!")
    print(f"Output directory: {output_dir}")

    return {
        'train': len(train_images),
        'val': len(val_images),
        'test': len(test_images)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset for YOLO training")
    parser.add_argument("--images", default="training/frames", help="Images directory")
    parser.add_argument("--labels", default="training/labels", help="Labels directory")
    parser.add_argument("--output", default="training/dataset", help="Output directory")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.05, help="Test ratio")

    args = parser.parse_args()

    split_dataset(
        args.images,
        args.labels,
        args.output,
        args.train,
        args.val,
        args.test
    )
