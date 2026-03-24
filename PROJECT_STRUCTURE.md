# Smart Traffic - Project Structure

## Overview
This project is a traffic violation detection system using YOLOv8 for object detection and rule-based violation analysis.

## Active Components (Currently Used)

### Web Application (`web_app/`)
```
web_app/
├── app.py              # Main Flask application (detection, tracking, rules, API)
├── models.py           # SQLAlchemy database models (Violation)
├── templates/
│   ├── index.html      # Main inference page (upload + results)
│   └── dashboard.html  # Violations dashboard (with auto-refresh)
├── utils/
│   └── tracking_manager.py  # Connects tracker + rules engine
└── static/
    ├── uploads/        # Uploaded files (temporary)
    ├── results/        # Processed results (per-run folders)
    └── violations.db   # SQLite database
```

### Rules Engine (`rules/`)
```
rules/
├── red_light.py        # StopLineRule - detects stop-line violations
└── lane.py             # LaneViolationRule - detects lane crossing violations
```

### Configuration (`config/`)
```
config/
├── cameras.json        # Camera-specific geometry (stop_line_y, lane_line)
└── mapping.json        # COCO to project class mapping
```

### Training (`training/`)
```
training/
├── config.yaml         # Training configuration
├── yolo/
│   ├── data_vehicles.yaml  # Dataset config for vehicles
│   ├── data_lights.yaml    # Dataset config for traffic lights
│   ├── train_vehicles.py   # Training script for vehicles
│   └── train_lights.py     # Training script for lights
└── utils/
    ├── extract_frames.py   # Extract frames from video
    ├── auto_label_yolo.py  # Auto-label using pretrained model
    ├── split_datasets.py   # Basic dataset splitter
    └── split_dataset_v2.py # Improved dataset splitter (use this one)
```

---

## Legacy/Demo Components (Not Used by Main App)

### Old Demo Script (`scripts/`)
```
scripts/
└── run_infer_demo.py   # Standalone demo (uses workers/ components)
```

### Workers (Legacy) (`workers/`)
```
workers/
├── rules/
│   └── red_light.py    # Shapely-based RedLightRule (needs red_zone_polygon)
├── tracking/
│   └── bytetrack_wrapper.py  # Stub ByteTrack wrapper
├── evidence/
│   └── builder.py      # Evidence packet builder
└── iou_tracker.py      # Simple IoU tracker (fallback, still used)
```

### Legacy Config (`configs/`)
```
configs/
└── camera_sample.json  # Sample config for old demo script
```

---

## Data Flow

```
1. Upload (image/video)
       ↓
2. YOLOv8 Detection (model.predict/model.track)
       ↓
3. Class Mapping (COCO → car/bus/truck/motorbike/person)
       ↓
4. Tracking (ByteTrack via Ultralytics or IoU fallback)
       ↓
5. Rules Check (StopLineRule, LaneViolationRule)
       ↓
6. Violation Events → Database + Evidence Crops
       ↓
7. Dashboard Display + Export (CSV/PDF)
```

---

## Key Configuration

### Target Classes (in app.py)
```python
TARGET_NAMES = ['car', 'bus', 'truck', 'motorbike', 'person']
```

### Camera Config (config/cameras.json)
```json
{
  "default": {
    "stop_line_y": 350,
    "lane_line": [[960, 0], [960, 1080]]
  }
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with upload form |
| `/predict` | POST | Process image/video upload |
| `/predict_video` | POST | Process video/stream URL |
| `/dashboard` | GET | Violations dashboard |
| `/dashboard/update/<id>` | POST | Update violation status |
| `/api/recent_violations` | GET | JSON API for recent violations |
| `/violations` | GET | Get violations (per-run or global) |
| `/_result_counts` | GET | Get detection counts for result |
| `/export/csv` | GET | Download violations as CSV |
| `/export/pdf` | GET | Download violations as PDF |
| `/static/<path>` | GET | Serve static files |

---

## Running the Application

```bash
# Activate virtual environment
source .venv_mac/bin/activate

# Start server
PYTHONPATH=/path/to/smart_traffic python web_app/app.py

# Access at http://127.0.0.1:5050
```
