# Smart Traffic Violation Detection System - Implementation Guide

## Table of Contents

### Part I: Technical Implementation
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Detection System](#5-detection-system)
6. [Tracking System](#6-tracking-system)
7. [Rules Engine](#7-rules-engine)
8. [Evidence Management](#8-evidence-management)
9. [Web Application](#9-web-application)
10. [Database Schema](#10-database-schema)
11. [API Endpoints](#11-api-endpoints)
12. [Configuration](#12-configuration)
13. [Training Pipeline](#13-training-pipeline)
14. [File Structure](#14-file-structure)

### Part II: Academic Supplement (See [ACADEMIC_SUPPLEMENT.md](ACADEMIC_SUPPLEMENT.md))
15. Related Work & ITEMS Analysis
16. Design Justification (Why YOLOv8? Why ByteTrack?)
17. Evaluation Framework (Metrics, Ground Truth Protocol)
18. Failure Analysis
19. Limitations & Constraints
20. Future Work
21. Viva Defense Guide (12 Q&A with answers)
22. References

---

## 1. Project Overview

### Purpose
This system automatically detects traffic violations (stop-line violations, lane violations) from video feeds or images using computer vision and machine learning.

### Key Features
- **Real-time Detection**: Process video streams or uploaded files
- **Multi-class Vehicle Detection**: Cars, buses, trucks, motorbikes, persons
- **Violation Detection**: Stop-line crossing, lane violations
- **Evidence Collection**: Automatic cropping and saving of violation evidence
- **Dashboard**: Web-based interface for reviewing violations
- **Export**: CSV and PDF reports

### Technology Stack
| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack (via Ultralytics) + IoU Fallback |
| Backend | Flask (Python) |
| Database | SQLite + SQLAlchemy |
| Frontend | HTML/CSS/JavaScript |
| CV Processing | OpenCV |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Upload Form  │  │  Dashboard   │  │   Export (CSV/PDF)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK WEB SERVER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   /predict   │  │  /dashboard  │  │   /api/violations    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐   │
│  │ YOLOv8   │ → │ Mapping  │ → │ Tracking │ → │ Rules Check │   │
│  │Detection │   │(COCO→Tgt)│   │(ByteTrack)│   │(Violations)│   │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   SQLite DB  │  │ Evidence Imgs│  │   YOLO Labels        │   │
│  │ (violations) │  │  (crops)     │  │   (.txt files)       │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Main Application (`web_app/app.py`)

The main Flask application handles:

```python
# Key imports
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from ultralytics import YOLO
import cv2

# Global model loading at startup
model = YOLO('yolov8n.pt')  # Pretrained COCO model

# Class mapping: COCO classes → Our target classes
TARGET_NAMES = ['car', 'bus', 'truck', 'motorbike', 'person']
model_to_target_map = {
    2: 0,   # COCO 'car' (id=2) → our 'car' (id=0)
    5: 1,   # COCO 'bus' (id=5) → our 'bus' (id=1)
    7: 2,   # COCO 'truck' (id=7) → our 'truck' (id=2)
    3: 3,   # COCO 'motorcycle' (id=3) → our 'motorbike' (id=3)
    0: 4,   # COCO 'person' (id=0) → our 'person' (id=4)
}
```

### 3.2 Database Model (`web_app/models.py`)

```python
class Violation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    violation_type = db.Column(db.String(50))     # 'stop_line_violation', 'lane_violation'
    image_path = db.Column(db.String(200))        # Path to evidence crop
    vehicle_type = db.Column(db.String(50))       # 'car', 'bus', etc.
    track_id = db.Column(db.Integer)              # Tracking ID
    confidence = db.Column(db.Float)              # Detection confidence
    status = db.Column(db.String(20), default='New')  # New, Reviewed, Sent
```

### 3.3 Tracking Manager (`web_app/utils/tracking_manager.py`)

Connects detection → tracking → rules:

```python
def update_and_check(run_id, camera_id, frame_idx, frame_img, detections):
    """
    Main pipeline function called for each frame.
    
    Args:
        run_id: Unique identifier for this processing run
        camera_id: Camera configuration key
        frame_idx: Current frame number
        frame_img: BGR numpy array (OpenCV image)
        detections: List of detected objects
        
    Returns:
        tracks: List of tracked objects with persistent IDs
        violations: List of detected violations
    """
    # 1. Update tracker with new detections
    tracks = TRACKER.update(detections, key=tracker_key, frame_id=frame_idx)
    
    # 2. Check each rule engine for violations
    violations = []
    for engine in rule_engines:
        violations.extend(engine.process_track(...))
    
    return tracks, violations
```

---

## 4. Data Flow Pipeline

### Step-by-Step Processing

```
INPUT (Image/Video)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: FILE UPLOAD                                          │
│ - User uploads via web form                                  │
│ - File saved to web_app/static/uploads/                      │
│ - Unique run_id generated (UUID)                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: YOLO DETECTION                                       │
│ - For images: model.predict(image)                           │
│ - For videos: model.track(frame, persist=True)               │
│ - Returns bounding boxes, class IDs, confidence scores       │
│ - ByteTrack provides persistent track IDs for videos         │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: CLASS MAPPING                                        │
│ - COCO class ID → Target class ID                            │
│ - Example: COCO 'car'(2) → Target 'car'(0)                   │
│ - Unmapped classes (animals, furniture, etc.) are filtered   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: TRACKING (Video only)                                │
│ - ByteTrack assigns persistent IDs across frames             │
│ - Fallback: Simple IoU tracker if ByteTrack unavailable      │
│ - Tracks maintain history of positions                       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: RULES EVALUATION                                     │
│ - StopLineRule: Check if vehicle crossed stop line           │
│ - LaneViolationRule: Check if vehicle crossed lane divider   │
│ - Returns violation events with metadata                     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: EVIDENCE COLLECTION                                  │
│ - Crop vehicle from frame using bounding box                 │
│ - Save to results/run_xxx/events/<event_id>.jpg              │
│ - Store relative path in database                            │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 7: DATABASE STORAGE                                     │
│ - Create Violation record with all metadata                  │
│ - Commit to SQLite database                                  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 8: VISUALIZATION                                        │
│ - Draw bounding boxes on image/video                         │
│ - Add class labels and confidence scores                     │
│ - Add track IDs for video                                    │
│ - Save annotated output                                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
OUTPUT (Annotated Image/Video + Violations in DB)
```

---

## 5. Detection System

### 5.1 Model Loading

```python
# At application startup (app.py lines 35-75)
MODEL_PATH = os.environ.get('MODEL_PATH', 'yolov8n.pt')

from ultralytics import YOLO
model = YOLO(MODEL_PATH)

# Build class mapping
TARGET_NAMES = ['car', 'bus', 'truck', 'motorbike', 'person']
for model_id, model_name in model.names.items():
    if model_name.lower() in TARGET_NAMES:
        model_to_target_map[model_id] = TARGET_NAMES.index(model_name.lower())
```

### 5.2 Image Detection

```python
# In /predict route (app.py lines 265-330)
results = model.predict(source=str(upload_path), conf=0.25, save=False)

# Parse results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box
        cls_id = int(box.cls)                         # Class ID
        conf = float(box.conf)                        # Confidence
        
        # Map to target class
        mapped_cls = model_to_target_map.get(cls_id)
        if mapped_cls is not None:
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': conf,
                'class': mapped_cls
            })
```

### 5.3 Video Detection with Tracking

```python
# In process_video() (app.py lines 85-195)
def process_video(input_path, output_path, run_id, camera_id='default'):
    cap = cv2.VideoCapture(input_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use ByteTrack for persistent tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25)
        
        for box in results[0].boxes:
            # Extract track_id (persistent across frames)
            track_id = int(box.id[0]) if box.id is not None else None
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': conf,
                'class': mapped_cls,
                'track_id': track_id  # Persistent ID from ByteTrack
            })
```

---

## 6. Tracking System

### 6.1 ByteTrack Integration

ByteTrack is used via Ultralytics' built-in tracking:

```python
# Primary tracking method (via Ultralytics)
results = model.track(
    frame, 
    persist=True,           # Keep track IDs across frames
    tracker="bytetrack.yaml",  # Use ByteTrack algorithm
    conf=0.25,
    verbose=False
)

# Each detection now has a track ID
track_id = int(box.id[0])  # Persistent across the video
```

### 6.2 IoU Fallback Tracker

When ByteTrack is unavailable, a simple IoU-based tracker is used:

```python
# workers/iou_tracker.py
class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.iou_thresh = iou_threshold
        self.tracks = {}  # id -> {bbox, cls, lost_frames}
    
    def update(self, detections):
        # 1. Compute IoU between existing tracks and new detections
        # 2. Match detections to tracks using greedy IoU matching
        # 3. Create new tracks for unmatched detections
        # 4. Age out tracks that haven't been matched for max_lost frames
        return updated_tracks
```

### 6.3 Tracker Manager

```python
# web_app/utils/tracking_manager.py
class _LocalTrackerManager:
    def get_tracker(self, key):
        # Try ByteTrack first
        if create_tracker is not None:
            return create_tracker()
        # Fallback to IoU tracker
        if create_fallback_tracker is not None:
            return create_fallback_tracker()
    
    def update(self, detections, key, frame_id):
        tracker = self.get_tracker(key)
        return tracker.update(detections)
```

---

## 7. Rules Engine

### 7.1 Stop Line Rule (`rules/red_light.py`)

Detects vehicles crossing a stop line without stopping:

```python
class StopLineRule:
    def __init__(self, stop_line, required_stop_seconds=1.5):
        """
        Args:
            stop_line: ((x1,y1), (x2,y2)) - line coordinates
            required_stop_seconds: Time vehicle must stop before crossing
        """
        self.stop_line = stop_line
        self.required_stop_seconds = required_stop_seconds
        self.track_states = {}  # Stores state per track_id
    
    def _crosses_line(self, prev_centroid, curr_centroid):
        """Check if movement crossed the stop line."""
        (x1, y1), (x2, y2) = self.stop_line
        def side(point):
            return (x2-x1)*(point[1]-y1) - (y2-y1)*(point[0]-x1)
        # Crossed if signs are different (on opposite sides)
        return side(prev_centroid) * side(curr_centroid) < 0
    
    def process_track(self, track_id, bbox, timestamp, frame_id, ...):
        centroid = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
        state = self.track_states.get(track_id)
        
        if state and self._crosses_line(state['last_centroid'], centroid):
            # Check if vehicle stopped before crossing
            if not self._was_stopped_long_enough(state):
                # VIOLATION!
                return {
                    'event_id': uuid.uuid4(),
                    'event_type': 'stop_line_violation',
                    'track_id': track_id,
                    'bbox': bbox,
                    'frame_id': frame_id,
                    ...
                }
        
        # Update state
        self.track_states[track_id] = {
            'last_centroid': centroid,
            'last_time': timestamp,
            ...
        }
        return None
```

### 7.2 Lane Violation Rule (`rules/lane.py`)

Detects vehicles crossing lane dividers:

```python
class LaneViolationRule:
    def __init__(self, lane_line):
        """
        Args:
            lane_line: ((x1,y1), (x2,y2)) - lane divider coordinates
        """
        self.lane_line = lane_line
        self.track_states = {}
    
    def process_track(self, track_id, bbox, timestamp, frame_id, ...):
        centroid = self._get_centroid(bbox)
        state = self.track_states.get(track_id)
        
        if state and self._crosses_line(state['last_centroid'], centroid):
            # LANE VIOLATION!
            return {
                'event_id': uuid.uuid4(),
                'event_type': 'lane_violation',
                ...
            }
        
        self.track_states[track_id] = {'last_centroid': centroid, ...}
        return None
```

### 7.3 Rule Engine Integration

```python
# tracking_manager.py
def get_rule_engine(camera_id):
    cfg = CAMERA_CONFIG.get(camera_id, CAMERA_CONFIG["default"])
    engines = []
    
    # Add stop line rule
    y = cfg.get("stop_line_y", 350)
    stop_line = ((0, y), (1920, y))
    engines.append(StopLineRule(stop_line, required_stop_seconds=1.5))
    
    # Add lane rule if configured
    lane_line = cfg.get("lane_line")
    if lane_line:
        engines.append(LaneViolationRule(lane_line))
    
    return engines
```

---

## 8. Evidence Management

### 8.1 Evidence Cropping

When a violation is detected, the vehicle is cropped from the frame:

```python
# In process_video() and /predict route
if violations:
    evdir = output_path.parent / 'events'
    evdir.mkdir(parents=True, exist_ok=True)
    
    for ev in violations:
        bbox = ev.get('bbox')
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop the vehicle from the frame
        crop = frame[y1:y2, x1:x2]
        
        # Save with unique event ID
        crop_path = evdir / f"{ev['event_id']}.jpg"
        cv2.imwrite(str(crop_path), crop)
        
        # Store relative path for database
        ev['_crop_path'] = os.path.relpath(str(crop_path), STATIC_DIR)
```

### 8.2 Evidence Storage Structure

```
web_app/static/results/
├── run_abc123/
│   ├── abc123_video.mp4        # Annotated output video
│   ├── labels/
│   │   └── frame_001.txt       # YOLO format labels
│   ├── events/
│   │   ├── event-uuid-1.jpg    # Violation evidence crop
│   │   └── event-uuid-2.jpg
│   ├── tracks.json             # Track data
│   └── violations.json         # Violation metadata
```

---

## 9. Web Application

### 9.1 Main Page (`index.html`)

Features:
- Camera geometry selection dropdown
- File upload form (images/videos)
- Live stream URL input (RTSP/HTTP)
- Result display with annotated output
- Detection counts
- Recent violations panel (auto-refreshes every 5 seconds)

```html
<!-- Upload Form -->
<form action="/predict" method="post" enctype="multipart/form-data">
    <select name="camera_id">
        <option value="default">Default Camera</option>
        <!-- Camera options from config -->
    </select>
    <input type="file" name="file" accept="image/*,video/*">
    <button type="submit">Process</button>
</form>

<!-- Recent Violations (AJAX auto-refresh) -->
<div id="recent-violations">
    <!-- Populated by JavaScript calling /api/recent_violations -->
</div>
```

### 9.2 Dashboard (`dashboard.html`)

Features:
- Table of all violations
- Evidence thumbnails (clickable to enlarge)
- Status management (New → Reviewed → Sent)
- Auto-refresh every 10 seconds
- Export buttons (CSV/PDF)

```html
<table>
    <tr>
        <th>ID</th>
        <th>Timestamp</th>
        <th>Type</th>
        <th>Vehicle</th>
        <th>Evidence</th>
        <th>Status</th>
        <th>Actions</th>
    </tr>
    {% for v in violations %}
    <tr>
        <td>{{ v.id }}</td>
        <td>{{ v.timestamp }}</td>
        <td>{{ v.violation_type }}</td>
        <td>{{ v.vehicle_type }}</td>
        <td><img src="/static/{{ v.image_path }}" class="thumb"></td>
        <td>{{ v.status }}</td>
        <td>
            <form action="/dashboard/update/{{ v.id }}" method="post">
                <button name="status" value="Reviewed">Ack</button>
            </form>
        </td>
    </tr>
    {% endfor %}
</table>
```

---

## 10. Database Schema

### Violation Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key, auto-increment |
| `timestamp` | DATETIME | When violation was detected |
| `violation_type` | VARCHAR(50) | 'stop_line_violation' or 'lane_violation' |
| `image_path` | VARCHAR(200) | Relative path to evidence crop |
| `vehicle_type` | VARCHAR(50) | 'car', 'bus', 'truck', 'motorbike', 'person' |
| `track_id` | INTEGER | Tracking ID from ByteTrack |
| `confidence` | FLOAT | Detection confidence (0-1) |
| `status` | VARCHAR(20) | 'New', 'Reviewed', 'Sent' |

### Database Location
```
web_app/static/violations.db
```

---

## 11. API Endpoints

### 11.1 Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with upload form |
| `/predict` | POST | Process uploaded image/video |
| `/predict_video` | POST | Process video file or stream URL |
| `/dashboard` | GET | Violations management dashboard |
| `/dashboard/update/<id>` | POST | Update violation status |

### 11.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recent_violations` | GET | Last 10 violations as JSON |
| `/violations` | GET | All violations (optional `?run=run_xxx`) |
| `/_result_counts` | GET | Detection counts for result image |

### 11.3 Export Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/export/csv` | GET | Download all violations as CSV |
| `/export/pdf` | GET | Download all violations as PDF report |

### 11.4 Static Files

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/static/<path>` | GET | Serve uploaded files, results, evidence |

---

## 12. Configuration

### 12.1 Camera Configuration (`config/cameras.json`)

```json
{
    "default": {
        "description": "Default camera settings",
        "stop_line_y": 350,
        "lane_line": [[960, 0], [960, 1080]]
    },
    "intersection_north": {
        "description": "North-facing intersection camera",
        "stop_line_y": 400,
        "lane_line": [[640, 0], [640, 720]]
    }
}
```

### 12.2 Class Mapping (`config/mapping.json`)

```json
{
    "coco_to_target": {
        "person": "person",
        "car": "car",
        "motorcycle": "motorbike",
        "bus": "bus",
        "truck": "truck"
    },
    "target_classes": ["car", "bus", "truck", "motorbike", "person"]
}
```

### 12.3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `yolov8n.pt` | Path to YOLO model weights |

---

## 13. Training Pipeline

### 13.1 Dataset Preparation

```bash
# 1. Extract frames from video
python training/utils/extract_frames.py input_video.mp4 output_frames/ --fps 5

# 2. Auto-label using pretrained model
python training/utils/auto_label_yolo.py output_frames/ labels/

# 3. Split into train/val/test
python training/utils/split_dataset_v2.py images/ labels/ dataset/ --train 0.7 --val 0.2
```

### 13.2 Training Configuration (`training/yolo/data_vehicles.yaml`)

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

names:
  0: car
  1: bus
  2: truck
  3: motorbike
  4: person
```

### 13.3 Training Script (`training/yolo/train_vehicles.py`)

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Start from pretrained

results = model.train(
    data='data_vehicles.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='mps',  # Apple Silicon GPU
    project='runs/detect',
    name='train_vehicles'
)
```

---

## 14. File Structure

```
smart_traffic/
├── web_app/
│   ├── app.py                    # Main Flask application
│   ├── models.py                 # SQLAlchemy database models
│   ├── templates/
│   │   ├── index.html            # Main inference page
│   │   └── dashboard.html        # Violations dashboard
│   ├── utils/
│   │   └── tracking_manager.py   # Tracking + rules integration
│   └── static/
│       ├── uploads/              # Uploaded files
│       ├── results/              # Processing results
│       └── violations.db         # SQLite database
│
├── rules/
│   ├── red_light.py              # StopLineRule class
│   └── lane.py                   # LaneViolationRule class
│
├── workers/
│   ├── iou_tracker.py            # Fallback IoU tracker
│   ├── evidence/
│   │   └── builder.py            # Evidence packet builder (legacy)
│   ├── rules/
│   │   └── red_light.py          # Legacy shapely-based rule
│   └── tracking/
│       └── bytetrack_wrapper.py  # Legacy ByteTrack stub
│
├── config/
│   ├── cameras.json              # Camera geometry configs
│   └── mapping.json              # Class mapping config
│
├── training/
│   ├── config.yaml               # Training configuration
│   ├── yolo/
│   │   ├── data_vehicles.yaml    # Vehicle dataset config
│   │   ├── data_lights.yaml      # Traffic light dataset config
│   │   ├── train_vehicles.py     # Vehicle training script
│   │   └── train_lights.py       # Light training script
│   └── utils/
│       ├── extract_frames.py     # Frame extraction utility
│       ├── auto_label_yolo.py    # Auto-labeling utility
│       ├── split_datasets.py     # Dataset splitter v1
│       └── split_dataset_v2.py   # Dataset splitter v2
│
├── scripts/
│   └── run_infer_demo.py         # Legacy standalone demo
│
├── configs/
│   └── camera_sample.json        # Legacy demo config
│
├── .vscode/
│   └── settings.json             # VS Code settings
│
├── .venv_mac/                    # Python virtual environment
├── PROJECT_STRUCTURE.md          # Project structure docs
├── IMPLEMENTATION_GUIDE.md       # This file (Part I)
├── ACADEMIC_SUPPLEMENT.md        # Part II: Evaluation, ITEMS, Viva Guide
└── requirements.txt              # Python dependencies
```

---

## Summary

This Smart Traffic system provides:

1. **Automated Detection**: YOLOv8 detects vehicles in images/videos
2. **Persistent Tracking**: ByteTrack maintains vehicle IDs across frames
3. **Rule-based Violations**: Configurable rules for stop-line and lane violations
4. **Evidence Collection**: Automatic cropping and storage of violation evidence
5. **Web Dashboard**: Review, manage, and export violations
6. **Extensible Design**: Easy to add new rules, cameras, or detection classes

The system processes ~30 FPS on modern hardware and can handle both uploaded files and live RTSP streams.

---

## Continue to Part II

For academic evaluation requirements, design justification, metrics, and viva preparation, see:

📄 **[ACADEMIC_SUPPLEMENT.md](ACADEMIC_SUPPLEMENT.md)**

Contents:
- ITEMS (Bengaluru Traffic Police) system analysis
- Why YOLOv8 over YOLOv5/EfficientDet (with benchmarks)
- Why ByteTrack over DeepSORT/SORT (with MOT17 metrics)
- Evaluation framework with ground truth protocol
- Failure analysis (night, occlusion, camera angles)
- Limitations & future work roadmap
- **12 viva questions with model answers**
