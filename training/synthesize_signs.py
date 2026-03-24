import os
import cv2
import random
import numpy as np
import csv
import glob
from pathlib import Path
import shutil

# Configuration
BACKGROUND_DIR = "data/vehicles_yolo/train/images"
SIGNS_Source_DIR = "/Users/Pavan/Downloads/archive-3/traffic_Data/DATA"
LABELS_CSV = "/Users/Pavan/Downloads/archive-3/labels.csv"
OUTPUT_DIR = "data/signs_yolo"
SIGNS_PER_IMAGE = (1, 3) # Min, Max signs per background
OUTPUT_SIZE = (640, 640)
TRAIN_SPLIT = 0.8

def load_sign_classes():
    classes = {}
    with open(LABELS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes[int(row['ClassId'])] = row['Name']
    return classes

def get_sign_images(source_dir):
    # Returns dict: class_id -> list of image paths
    signs = {}
    if not os.path.exists(source_dir):
        print(f"Error: Signs source directory not found: {source_dir}")
        return {}
        
    for class_id_str in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_id_str)
        if os.path.isdir(class_path):
            try:
                cid = int(class_id_str)
                images = glob.glob(os.path.join(class_path, "*.png")) + glob.glob(os.path.join(class_path, "*.jpg"))
                if images:
                    signs[cid] = images
            except ValueError:
                continue
    return signs

def overlay_sign(background, sign_img):
    bh, bw = background.shape[:2]
    sh, sw = sign_img.shape[:2]
    
    # Resize sign to be realistic (between 5% and 15% of background width)
    scale_factor = random.uniform(0.05, 0.15)
    target_sw = int(bw * scale_factor)
    aspect = sh / sw
    target_sh = int(target_sw * aspect)
    
    if target_sh >= bh or target_sw >= bw:
        return background, None # Skip if too big
        
    sign_resized = cv2.resize(sign_img, (target_sw, target_sh))
    
    # Random position
    max_x = bw - target_sw
    max_y = bh - target_sh
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    # Overlay (handle alpha channel if present, otherwise just copy)
    # Most signs in this dataset seem to be simple png/jpg crops, likely rectangular.
    # To make it look better, we could ignore black pixels but simple copy is fine for v1.
    
    roi = background[y:y+target_sh, x:x+target_sw]
    
    # Basic overlay
    background[y:y+target_sh, x:x+target_sw] = sign_resized
    
    # YOLO Format: class x_center y_center width height (normalized)
    x_center = (x + target_sw / 2) / bw
    y_center = (y + target_sh / 2) / bh
    w_norm = target_sw / bw
    h_norm = target_sh / bh
    
    return background, (x_center, y_center, w_norm, h_norm)

def main():
    print("Loading resources...")
    sign_classes = load_sign_classes()
    sign_images = get_sign_images(SIGNS_Source_DIR)
    
    if not sign_images:
        print("No sign images found!")
        return

    # Prepare output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

    backgrounds = glob.glob(os.path.join(BACKGROUND_DIR, "*.jpg")) + glob.glob(os.path.join(BACKGROUND_DIR, "*.png"))
    if not backgrounds:
        print(f"No background images found in {BACKGROUND_DIR}")
        return

    print(f"Found {len(backgrounds)} backgrounds and {len(sign_images)} sign classes.")
    
    # Generate
    generated_count = 0
    for bg_path in backgrounds:
        img = cv2.imread(bg_path)
        if img is None: continue
        
        # Decide if train or val
        split = 'train' if random.random() < TRAIN_SPLIT else 'val'
        
        new_img = img.copy()
        labels = []
        
        num_signs = random.randint(*SIGNS_PER_IMAGE)
        
        for _ in range(num_signs):
            # Pick random class
            cid = random.choice(list(sign_images.keys()))
            # Pick random image
            s_path = random.choice(sign_images[cid])
            s_img = cv2.imread(s_path)
            
            if s_img is None: continue
            
            new_img, label = overlay_sign(new_img, s_img)
            if label:
                labels.append(f"{cid} {' '.join(map(str, label))}")
        
        if labels:
            # Save
            filename = os.path.basename(bg_path)
            out_img_path = os.path.join(OUTPUT_DIR, split, 'images', filename)
            out_lbl_path = os.path.join(OUTPUT_DIR, split, 'labels', os.path.splitext(filename)[0] + ".txt")
            
            cv2.imwrite(out_img_path, new_img)
            with open(out_lbl_path, 'w') as f:
                f.write("\n".join(labels))
            generated_count += 1
            
            if generated_count % 100 == 0:
                print(f"Generated {generated_count} images...", end='\r')

    print(f"\nSynthesis complete! Generated {generated_count} images in {OUTPUT_DIR}")
    
    # verify class count
    print(f"Total Sign Classes: {max(sign_classes.keys()) + 1}")

if __name__ == "__main__":
    main()
