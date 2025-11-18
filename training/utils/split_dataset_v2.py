"""
Split dataset into train/val/test with proper handling of existing directories.
"""

import os
import shutil
import random
from pathlib import Path


def split_dataset(img_dir, labels_dir, out_root, train=0.7, val=0.2, seed=42):
    """
    Split images and labels into train/val/test sets.
    
    Args:
        img_dir: Source images directory
        labels_dir: Source labels directory
        out_root: Output root directory (will create train/val/test subdirs)
        train: Proportion for training (default 0.7)
        val: Proportion for validation (default 0.2)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    test = 1 - train - val
    
    # Ensure output directories exist
    for split_name in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            dir_path = os.path.join(out_root, split_name, subdir)
            os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    imgs.sort()  # Sort for consistency
    random.shuffle(imgs)
    
    print(f"Total images found: {len(imgs)}")
    
    # Calculate split sizes
    n_train = int(len(imgs) * train)
    n_val = int(len(imgs) * val)
    
    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train+n_val],
        "test": imgs[n_train+n_val:]
    }
    
    print(f"Train: {len(splits['train'])} images")
    print(f"Val: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")
    
    # Copy files to split directories
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name}...")
        for i, f in enumerate(files):
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)} files processed...")
            
            # Copy image
            src_img = os.path.join(img_dir, f)
            dst_img = os.path.join(out_root, split_name, "images", f)
            
            # Skip if source and destination are the same
            if os.path.abspath(src_img) != os.path.abspath(dst_img):
                shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label
            base_name = Path(f).stem
            label_file = base_name + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(out_root, split_name, "labels", label_file)
            
            if os.path.exists(src_label):
                if os.path.abspath(src_label) != os.path.abspath(dst_label):
                    shutil.copy2(src_label, dst_label)
        
        print(f"✓ {split_name} split complete")
    
    print("\n" + "="*60)
    print("Dataset splitting completed!")
    print("="*60)
    print(f"\nStructure created:")
    print(f"  {out_root}/train/images/ ({len(splits['train'])} images)")
    print(f"  {out_root}/train/labels/ ({len(splits['train'])} labels)")
    print(f"  {out_root}/val/images/ ({len(splits['val'])} images)")
    print(f"  {out_root}/val/labels/ ({len(splits['val'])} labels)")
    print(f"  {out_root}/test/images/ ({len(splits['test'])} images)")
    print(f"  {out_root}/test/labels/ ({len(splits['test'])} labels)")


if __name__ == "__main__":
    # Configuration
    img_dir = "data/vehicles_yolo/train/images"
    labels_dir = "data/vehicles_yolo/train/labels"
    out_root = "data/vehicles_yolo"
    
    print("Starting dataset split...")
    print(f"Source images: {img_dir}")
    print(f"Source labels: {labels_dir}")
    print(f"Output root: {out_root}")
    print()
    
    split_dataset(img_dir, labels_dir, out_root, train=0.7, val=0.2)
