#!/usr/bin/env python3
"""
Prepare a YOLO vehicle dataset from training videos: extract frames → auto-label → split.

Run from repo root:
  PYTHONPATH=/path/to/smart_traffic python training/prepare_dataset.py
  PYTHONPATH=/path/to/smart_traffic python training/prepare_dataset.py --every 10 --out data/vehicles_yolo

Then run finetune:
  python training/finetune.py --data training/yolo/data_vehicles.yaml --epochs 30
"""
import argparse
import os
import sys
from pathlib import Path

# Resolve repo root (parent of training/)
REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Allow running without PYTHONPATH from repo root
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    p = argparse.ArgumentParser(description="Prepare vehicle YOLO dataset from videos")
    p.add_argument(
        "--videos",
        type=Path,
        default=REPO_ROOT / "training" / "videos",
        help="Directory containing .mp4 videos",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "vehicles_yolo",
        help="Output dataset root (train/val/test will be created here)",
    )
    p.add_argument(
        "--every",
        type=int,
        default=10,
        help="Extract every Nth frame (default 10)",
    )
    p.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Train split ratio (default 0.8)",
    )
    p.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation split ratio (default 0.15)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model for auto-labeling (default yolov8n.pt)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for auto-labeling (default 0.4)",
    )
    p.add_argument(
        "--skip-label",
        action="store_true",
        help="Skip auto-labeling (use existing labels in staging/labels)",
    )
    args = p.parse_args()

    videos_dir = args.videos.resolve()
    out_root = args.out.resolve()
    staging = out_root.parent / (out_root.name + "_staging")
    staging_images = staging / "images"
    staging_labels = staging / "labels"

    if not videos_dir.is_dir():
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi")) + list(videos_dir.glob("*.mov"))
    if not videos:
        print(f"No video files in {videos_dir}")
        sys.exit(1)

    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)

    # 1) Extract frames
    from training.utils.extract_frames import extract

    print("Step 1: Extracting frames from videos...")
    for v in videos:
        prefix = v.stem
        extract(str(v), str(staging_images), every_n=args.every, prefix=prefix)
    image_files = [f for f in os.listdir(staging_images) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  Extracted {len(image_files)} images to {staging_images}")

    if not image_files:
        print("No images extracted. Check video paths and --every.")
        sys.exit(1)

    # 2) Auto-label (unless --skip-label)
    if not args.skip_label:
        print("Step 2: Auto-labeling with YOLO...")
        from training.utils.auto_label_yolo import auto_label_images

        auto_label_images(
            image_dir=str(staging_images),
            output_dir=str(staging_labels),
            model_name=args.model,
            conf_threshold=args.conf,
        )
    else:
        # Check that we have some labels
        label_files = list(staging_labels.glob("*.txt"))
        if len(label_files) < len(image_files) * 0.1:
            print("Warning: few labels found. Run without --skip-label to auto-label.")
    print(f"  Labels in {staging_labels}")

    # 3) Split into train/val/test
    print("Step 3: Splitting dataset...")
    from training.utils.split_dataset_v2 import split_dataset

    split_dataset(
        img_dir=str(staging_images),
        labels_dir=str(staging_labels),
        out_root=str(out_root),
        train=args.train,
        val=args.val,
        seed=42,
    )

    # Optionally remove staging to save space (keep by default for manual label edits)
    print(f"\nDataset ready at: {out_root}")
    print(f"  train/images, train/labels")
    print(f"  val/images, val/labels")
    print(f"  test/images, test/labels")
    print(f"\nNext: run finetune from repo root:")
    print(f"  PYTHONPATH={REPO_ROOT} python training/finetune.py --data training/yolo/data_vehicles.yaml --epochs 30 --device mps")
    print(f"  Or: ./scripts/run_finetune.sh --epochs 30 --device mps")


if __name__ == "__main__":
    main()
