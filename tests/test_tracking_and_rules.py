"""
Simple test to validate tracker + red-light rule using a saved annotated run's labels.
It will:
- load one image from web_app/static/results/<run>
- load the corresponding remapped YOLO label file
- call tracking_manager.update_and_check() repeatedly (simulate frames)

Run: python tests/test_tracking_and_rules.py
"""
import cv2, json, time
from pathlib import Path
from web_app.utils.tracking_manager import update_and_check

RUN = Path("web_app/static/results/run_58504c84ac01491cb458111f10269f1d")
IMG = RUN / "58504c84ac01491cb458111f10269f1d_58504c84ac01491cb458111f10269f1d_1_000141.jpg"
LABEL = RUN / "labels" / "58504c84ac01491cb458111f10269f1d_58504c84ac01491cb458111f10269f1d_1_000141.txt"

def load_labels(lbl_path):
    out = []
    img = cv2.imread(str(IMG))
    h,w = img.shape[:2]
    for line in open(lbl_path):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc = float(parts[1]); yc = float(parts[2]); ww = float(parts[3]); hh = float(parts[4])
        x1 = int((xc - ww/2) * w)
        x2 = int((xc + ww/2) * w)
        y1 = int((yc - hh/2) * h)
        y2 = int((yc + hh/2) * h)
        out.append({"bbox":[x1,y1,x2,y2],"score":0.8,"class":cls})
    return out

def run():
    if not IMG.exists() or not LABEL.exists():
        print('Required run image or label not found:', IMG, LABEL)
        return
    img = cv2.imread(str(IMG))
    detections = load_labels(LABEL)
    # simulate 3 frames identical (for demonstration)
    for frame_idx in range(3):
        tracks, violations = update_and_check(run_id="debug_run", camera_id="default", frame_idx=frame_idx, frame_img=img, detections=detections)
        print("Frame", frame_idx, "tracks:", tracks)
        print("Frame", frame_idx, "violations:", violations)
        time.sleep(0.2)

if __name__ == "__main__":
    run()
