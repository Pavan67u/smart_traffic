import os
import shutil
import random
import argparse


random.seed(7)


def split(img_dir, labels_dir, out_root, train=0.8, val=0.1):
    test = 1 - train - val
    for s in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(out_root, s, sub), exist_ok=True)
            
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    random.shuffle(imgs)
    n = int(len(imgs) * train)
    m = int(len(imgs) * val)
    
    splits = {
        "train": imgs[:n],
        "val": imgs[n:n+m],
        "test": imgs[n+m:]
    }
    
    for split_name, files in splits.items():
        for f in files:
            base = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(img_dir, f), os.path.join(out_root, split_name, "images", f))
            lbl = base + ".txt"
            if os.path.exists(os.path.join(labels_dir, lbl)):
                shutil.copy2(os.path.join(labels_dir, lbl), os.path.join(out_root, split_name, "labels", lbl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir", help="Directory containing images")
    ap.add_argument("labels_dir", help="Directory containing labels")
    ap.add_argument("out_root", help="Output directory for split datasets")
    ap.add_argument("--train", type=float, default=0.8, help="Proportion of training set")
    ap.add_argument("--val", type=float, default=0.1, help="Proportion of validation set")
    args = ap.parse_args()
    split(args.img_dir, args.labels_dir, args.out_root, args.train, args.val)