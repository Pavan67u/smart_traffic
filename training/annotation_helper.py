#!/usr/bin/env python3
"""
Simple annotation helper to create ground-truth CSV for stop-line/lane violations.

Usage:
  python training/annotation_helper.py --video path/to/video.mp4 --out ground_truth.csv --fps 2

What it does:
 - Extracts frames at the specified fps into a temporary folder
 - Prints instructions to inspect frames (open with your image viewer)
 - Allows you to enter frame ranges and labels via CLI
 - Saves a CSV with columns: video, frame_start, frame_end, bbox, stopped, ground_truth

This is intentionally minimal to create small test datasets quickly.
"""
import argparse
import csv
import os
from pathlib import Path
import cv2
import tempfile


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', default='ground_truth.csv')
    p.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract')
    p.add_argument('--frames_dir', default=None, help='Optional folder to write frames')
    return p.parse_args()


def extract_frames(video_path: Path, out_dir: Path, extract_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / extract_fps)))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = out_dir / f'frame_{saved:06d}.jpg'
            cv2.imwrite(str(fname), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def annotate_cli(frames_dir: Path, out_csv: Path, video_path: Path):
    print('\nFrames are extracted to:', frames_dir)
    print('Open the frames directory with your image viewer (e.g. open on macOS)')
    print('\nWhen ready, follow prompts to add annotation rows:')
    print('Enter frame_start,frame_end,bbox(x1:y1:x2:y2),stopped(true/false),VIOLATION/LEGAL')
    print('Example: 12,15,320:400:480:550,false,VIOLATION')
    print('Type DONE when finished')

    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video','frame_start','frame_end','bbox','stopped','ground_truth'])
        while True:
            line = input('> ').strip()
            if not line:
                continue
            if line.upper() == 'DONE':
                break
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 5:
                print('Invalid input. Expected 5 comma-separated fields.')
                continue
            frame_start, frame_end, bbox, stopped, gt = parts
            writer.writerow([str(video_path), frame_start, frame_end, bbox, stopped, gt])
    print('Saved ground truth to', out_csv)


def main():
    args = parse_args()
    video = Path(args.video)
    out_csv = Path(args.out)
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = Path(tempfile.mkdtemp(prefix='frames_'))
    print('Extracting frames... (this may take a while)')
    n = extract_frames(video, frames_dir, args.fps)
    print(f'Extracted {n} frames to {frames_dir}')
    print('\nOn macOS you can run: open', frames_dir)
    annotate_cli(frames_dir, out_csv, video)

if __name__ == '__main__':
    main()
