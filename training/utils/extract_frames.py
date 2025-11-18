import os
import argparse
import cv2


def extract(video, out_dir, every_n=5, prefix="f"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    i = fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_{fidx:06d}.jpg"), frame)
            fidx += 1
        i += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to input video file")
    ap.add_argument("out_dir", help="Output directory for frames")
    ap.add_argument("--every", type=int, default=5, help="Extract every nth frame")
    ap.add_argument("--prefix", type=str, default=None, help="Prefix for output image filenames")
    args = ap.parse_args()
    prefix = args.prefix if args.prefix is not None else os.path.splitext(os.path.basename(args.video))[0]
    extract(args.video, args.out_dir, args.every, prefix=prefix)