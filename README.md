# Smart Traffic Detection & Enforcement System

An AI-powered traffic violation detection system using computer vision for real-time monitoring, violation detection, and automated evidence generation.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API Documentation](#api-documentation)
8. [Database Schema](#database-schema)
9. [ML Pipeline](#ml-pipeline)
10. [Dataset](#dataset)
11. [Project Structure](#project-structure)
12. [Performance Metrics](#performance-metrics)
13. [RBAC Permissions](#rbac-permissions)
14. [Future Enhancements](#future-enhancements)

---

## Overview

### Problem Statement
- Manual traffic monitoring is expensive and error-prone
- Human operators cannot monitor multiple cameras 24/7
- Traditional systems lack intelligent violation detection
- Evidence collection is often inconsistent

### Solution
Smart Traffic automates traffic enforcement by:
- Detecting vehicles using YOLOv8 deep learning
- Tracking vehicles across frames with ByteTrack
- Applying rule-based violation detection
- Generating court-admissible evidence packages
- Providing a web dashboard for review and management

---

## Features

### Core Detection (Phase 1)
| Feature | Description |
|---------|-------------|
| YOLOv8 Inference | Real-time vehicle detection (car, truck, bus, motorcycle, pedestrian) |
| Multi-Object Tracking | ByteTrack for consistent vehicle IDs across frames |
| Red Light Violation | Detects vehicles crossing stop line on red signal |
| Lane Violation | Detects illegal lane changes |
| Evidence Cropping | Auto-crop vehicle images for documentation |
| Priority Scoring | 0-100 severity score for each violation |

### Data Management
| Feature | Description |
|---------|-------------|
| SQLite Database | Persistent violation storage |
| Status Workflow | New → Reviewed → Sent pipeline |
| CSV Export | Download violations as spreadsheet |
| PDF Reports | Generate evidence packages |
| GDPR Compliance | Data export and retention policies |

### Web Interface
| Feature | Description |
|---------|-------------|
| Upload Interface | Image/video upload with preview |
| Dashboard | Violations table with filtering |
| Evidence Carousel | Image gallery modal |
| Analytics Charts | Visual statistics |
| Dark Mode | Light/dark theme toggle |
| Calibration Manager | Camera geometry profiles |

### Security & Compliance (Phase 2)
| Feature | Description |
|---------|-------------|
| RBAC | Admin/Officer/Viewer roles |
| Authentication | Login/logout with sessions |
| Audit Logging | Track all user actions |
| Auto-Cleanup | Configurable retention policy |

### Signal Integration
| Feature | Description |
|---------|-------------|
| Signal Manager | Track RED/GREEN/YELLOW states |
| EWMA Timing | Adaptive signal suggestions |
| Anti-Flap Hold | 45-second stability window |
| Peak Hour Weighting | Time-based adjustments |

### Alerts & Monitoring
| Feature | Description |
|---------|-------------|
| Webhook Alerts | HTTP POST notifications |
| Email Alerts | SMTP notifications |
| Cooldown Throttling | 180-second alert cooldown |
| Model Drift Tracking | Confidence trend analysis |
| Camera Health | Monitor camera status |

### Advanced AI
| Feature | Description |
|---------|-------------|
| Active Learning | Learn from corrections |
| Model Ensemble | Multi-model voting |
| Video Streaming | RTSP/MJPEG support |
| ANPR/LPR | License plate recognition |

---

## Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| Flask | Web framework |
| Flask-SQLAlchemy | Database ORM |
| SQLite | Primary database |
| Celery + Redis | Distributed task processing |
| OpenCV 4.6+ | Image/video processing |

### AI/ML
| Technology | Purpose |
|------------|---------|
| Ultralytics YOLOv8 | Object detection |
| ByteTrack | Multi-object tracking |
| PyTorch | Deep learning framework |
| NumPy & SciPy | Numerical computations |

### Frontend
| Technology | Purpose |
|------------|---------|
| Jinja2 | Server-side templating |
| JavaScript | Client interactivity |
| CSS3 | Styling with dark mode |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Docker Compose | Service orchestration |
| Redis | Message broker & caching |

---

## System Architecture

### Data Flow
```
Video Input → Frame Extraction → YOLOv8 Detection → ByteTrack Tracking
                                                           ↓
Dashboard ← Database Storage ← Evidence Generation ← Rule Engine
                                                           ↓
                                                    Alert System
```

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                     FLASK WEB SERVER                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Routes  │  │   Auth   │  │ Analytics│  │  Export  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      AI/ML PIPELINE                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  YOLOv8  │→ │ByteTrack │→ │  Rules   │→ │ Evidence │    │
│  │Detection │  │ Tracking │  │  Engine  │  │Generator │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  SQLite  │  │ Evidence │  │  Redis   │                  │
│  │ Database │  │ Storage  │  │  Cache   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Quick Start (Docker)
```bash
# Build and run
docker compose -f infra/docker-compose.yml up --build

# Access at http://localhost:5050
```

### Local Development
```bash
# Clone repository
git clone <repo-url>
cd smart_traffic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r infra/requirements.txt

# Run the application
python web_app/app.py
```

Access the application at `http://localhost:5050`

---

## Usage

### Web Interface
1. Open `http://localhost:5050`
2. Upload image/video or provide stream URL
3. Select camera profile
4. Click "Process" to run detection
5. View results and violations on dashboard

### API Usage
```python
import requests

# Upload and process image
files = {'file': open('traffic.jpg', 'rb')}
response = requests.post('http://localhost:5050/predict', files=files)
print(response.json())
```

### Training Custom Model
```bash
# Activate environment
source .venv/bin/activate

# Train model
yolo detect train data=training/merged_dataset/data.yaml model=yolov8n.pt epochs=30
```

---

## API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main inference page |
| `/predict` | POST | Process image/video |
| `/dashboard` | GET | Violations dashboard |
| `/api/recent_violations` | GET | Get recent violations |
| `/api/health` | GET | System health check |
| `/api/signal/status` | GET | Traffic signal state |
| `/export/csv` | GET | Export as CSV |
| `/export/pdf` | GET | Export as PDF |

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | POST | User login |
| `/logout` | GET | User logout |
| `/register` | POST | User registration |

### Request Example
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
camera_id: cam_01
```

### Response Example
```json
{
  "status": "success",
  "detections": 12,
  "violations": 2,
  "result_image": "/static/results/result_001.jpg",
  "violations_list": [
    {
      "id": 145,
      "type": "red_light_violation",
      "vehicle": "car",
      "confidence": 0.92,
      "priority": "high"
    }
  ]
}
```

---

## Database Schema

### User Table
```sql
CREATE TABLE user (
    id INTEGER PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    role VARCHAR(20) DEFAULT 'viewer',  -- admin, officer, viewer
    created_at DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    last_login DATETIME
);
```

### Violation Table
```sql
CREATE TABLE violation (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    violation_type VARCHAR(50) NOT NULL,
    image_path VARCHAR(255),
    vehicle_type VARCHAR(50),
    track_id INTEGER,
    confidence FLOAT,
    status VARCHAR(20) DEFAULT 'New',
    priority_score INTEGER DEFAULT 50,
    priority_level VARCHAR(10),
    camera_id VARCHAR(50)
);
```

### AuditLog Table
```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES user(id),
    action VARCHAR(100),
    resource_type VARCHAR(50),
    resource_id INTEGER,
    details TEXT,
    timestamp DATETIME,
    ip_address VARCHAR(45)
);
```

---

## ML Pipeline

### Detection Model
- **Architecture**: YOLOv8 (Ultralytics)
- **Variants**: YOLOv8n (fast), YOLOv8s (balanced), YOLOv8m (accurate)
- **Classes**: car, truck, bus, motorcycle, pedestrian
- **Input Size**: 640x640
- **Performance**: 80-90% mAP50, 30+ FPS

### Tracking Algorithm
- **Primary**: ByteTrack (high accuracy)
- **Fallback**: IoU Tracker (CPU-friendly)

### Violation Rules

**Red Light Rule**:
```
IF signal == RED AND vehicle_crosses_stop_line THEN VIOLATION
```

**Lane Rule**:
```
IF vehicle_crosses_lane_boundary AND no_turn_signal THEN VIOLATION
```

### Priority Scoring Algorithm
```python
score = 50  # Base score
score += vehicle_type_weight  # bus:+20, truck:+15
score += confidence_bonus     # >90%:+15, >70%:+10
score += peak_hour_bonus      # +10 during rush hour
score += repeat_offender      # +15 if same vehicle
return min(score, 100)        # Cap at 100
```

---

## Dataset

### Overview
The model is trained on a merged dataset combining custom traffic videos and the BDD100K driving dataset, providing diverse traffic scenarios from multiple sources.

### Dataset Sources

#### 1. Custom Video Collection
| Video Source | Resolution | Duration | Frames Extracted |
|--------------|------------|----------|------------------|
| Traffic Camera 1 (1.mp4) | 1920x1080 | 81s | 406 |
| Traffic Camera 2 (2.mp4) | 640x352 | 75s | 378 |
| Traffic Camera 3 (3.mp4) | 640x360 | 4s | 20 |
| Traffic Camera 10 (10.mp4) | 640x360 | 81s | 406 |
| Traffic Camera 11 (11.mp4) | 640x352 | 27s | 138 |
| Traffic Camera 12 (12.mp4) | 640x352 | 20s | 100 |
| WhatsApp Video 1 (cam_new_1) | 832x464 | 21s | 104 |
| WhatsApp Video 2 (cam_new_2) | 1024x576 | 62s | 310 |
| WhatsApp Video 3 (cam_new_3) | 1024x576 | 25s | 126 |
| iStock Video 1 | 768x432 | 19s | 95 |
| iStock Video 2 | 768x432 | 18s | 89 |
| iStock Video 3 | 768x432 | 19s | 97 |
| iStock Video 4 | 768x432 | 13s | 66 |
| iStock Video 5 | 768x432 | 30s | 150 |
| iStock Video 6 | 768x432 | 24s | 120 |
| iStock Video 7 | 768x432 | 10s | 51 |
| iStock Video 8 | 768x432 | 10s | 50 |
| iStock Video 9 | 768x432 | 23s | 114 |
| iStock Video 10 | 768x432 | 15s | 75 |
| iStock Video 11 | 768x432 | 10s | 50 |
| **Subtotal** | - | ~590s | **2,945 frames** |

#### 2. BDD100K Dataset
- **Source**: Berkeley DeepDrive 100K Dataset
- **Type**: Large-scale driving video dataset
- **Images Used**: 1,983 (validation set)
- **Original Labels**: JSON format with box2d coordinates
- **Converted To**: YOLO format (normalized xywh)

### Final Merged Dataset

| Split | Images | Bounding Boxes | Percentage |
|-------|--------|----------------|------------|
| **Train** | 3,791 | 62,896 | 80% |
| **Validation** | 710 | 12,197 | 15% |
| **Test** | 238 | 4,153 | 5% |
| **Total** | **4,739** | **79,246** | 100% |

### Class Distribution

| Class ID | Class Name | Train | Val | Test | Total | Percentage |
|----------|------------|-------|-----|------|-------|------------|
| 0 | car | 45,193 | 8,550 | 3,062 | 56,805 | 71.7% |
| 1 | truck | 3,893 | 773 | 231 | 4,897 | 6.2% |
| 2 | bus | 4,532 | 1,007 | 268 | 5,807 | 7.3% |
| 3 | motorcycle | 1,537 | 315 | 97 | 1,949 | 2.5% |
| 4 | pedestrian | 7,741 | 1,552 | 495 | 9,788 | 12.3% |

### Class Distribution Visualization
```
car          ████████████████████████████████████████████████████████████  71.7%
pedestrian   ████████████                                                  12.3%
bus          ███████                                                        7.3%
truck        ██████                                                         6.2%
motorcycle   ███                                                            2.5%
```

### Dataset Preparation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Video Files    │────▶│ Frame Extraction│────▶│  Raw Frames     │
│  (.mp4)         │     │ (FFmpeg, 5 FPS) │     │  (.jpg)         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  YOLO Labels    │◀────│  Auto-Labeling  │◀────│  Pretrained     │
│  (.txt)         │     │  (YOLOv8s)      │     │  Model          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  BDD100K JSON   │────▶│  Format Convert │────▶│  Merged Dataset │
│  (box2d format) │     │  (to YOLO)      │     │  (train/val/test)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Label Format (YOLO)
Each `.txt` label file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example** (`frame_001.txt`):
```
0 0.523438 0.456250 0.125000 0.187500
0 0.234375 0.312500 0.093750 0.156250
4 0.765625 0.687500 0.062500 0.250000
2 0.445312 0.234375 0.187500 0.312500
```

### Class Mapping

#### COCO to Custom Mapping
| COCO Class ID | COCO Name | Custom Class ID | Custom Name |
|---------------|-----------|-----------------|-------------|
| 2 | car | 0 | car |
| 7 | truck | 1 | truck |
| 5 | bus | 2 | bus |
| 3 | motorcycle | 3 | motorcycle |
| 0 | person | 4 | pedestrian |

#### BDD100K to Custom Mapping
| BDD100K Category | Custom Class ID | Custom Name |
|------------------|-----------------|-------------|
| car | 0 | car |
| truck | 1 | truck |
| bus | 2 | bus |
| motor | 3 | motorcycle |
| person | 4 | pedestrian |
| rider | 4 | pedestrian |

### Data Augmentation
Applied during training:
| Augmentation | Value | Description |
|--------------|-------|-------------|
| HSV-Hue | 0.015 | Color hue variation |
| HSV-Saturation | 0.7 | Color saturation variation |
| HSV-Value | 0.4 | Brightness variation |
| Translation | 0.1 | Random position shift |
| Scale | 0.5 | Random zoom |
| Flip LR | 0.5 | Horizontal flip (50%) |
| Mosaic | 1.0 | 4-image mosaic augmentation |

### Dataset Quality Checks
All labels verified for:
- ✅ No empty label files
- ✅ No invalid class IDs (only 0-4)
- ✅ No out-of-bounds coordinates
- ✅ Consistent YOLO format
- ✅ Matching image-label pairs

### Dataset Configuration (data.yaml)
```yaml
path: /path/to/training/merged_dataset
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: pedestrian
```

### Storage Requirements
| Component | Size |
|-----------|------|
| Training Images | ~450 MB |
| Validation Images | ~85 MB |
| Test Images | ~30 MB |
| Label Files | ~15 MB |
| **Total Dataset** | **~580 MB** |

---

## Project Structure

```
smart_traffic/
├── web_app/                    # Flask application
│   ├── app.py                  # Main app (routes, inference)
│   ├── models.py               # Database models
│   ├── templates/              # HTML templates
│   │   ├── index.html          # Inference page
│   │   ├── dashboard.html      # Violations dashboard
│   │   └── login.html          # Auth pages
│   ├── static/                 # CSS, JS, images
│   └── utils/                  # Helper modules
│       ├── auth.py             # Authentication
│       ├── analytics.py        # Analytics engine
│       └── tracking_manager.py # Tracking integration
│
├── rules/                      # Violation rule engines
│   ├── red_light.py            # Red light detection
│   └── lane.py                 # Lane violation detection
│
├── tracking/                   # Object tracking
│   └── bytetrack_wrapper.py    # ByteTrack integration
│
├── models/                     # ML model weights
│   └── best.pt                 # Trained YOLOv8 model
│
├── training/                   # Model training
│   ├── train_traffic.py        # Training script
│   ├── merged_dataset/         # Combined dataset
│   │   ├── data.yaml           # Dataset config
│   │   ├── train/              # Training set (3,791 images)
│   │   ├── val/                # Validation set (710 images)
│   │   └── test/               # Test set (238 images)
│   └── utils/                  # Training utilities
│       ├── auto_label.py       # Auto-labeling with YOLOv8
│       └── merge_datasets.py   # Dataset merger
│
├── config/                     # Configuration
│   └── cameras.json            # Camera geometries
│
├── infra/                      # Infrastructure
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker config
│   └── docker-compose.yml      # Service orchestration
│
└── tests/                      # Test suite
    └── test_suite.py           # Pytest tests
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Detection mAP50 | 80-90% |
| Inference Speed | 30+ FPS (GPU) / 10+ FPS (CPU) |
| API Response | <100ms |
| Concurrent Users | 100+ |

---

## RBAC Permissions

| Role | Permissions |
|------|-------------|
| Admin | Full access: users, settings, all violations, system config |
| Officer | View/update violations, generate reports, export data |
| Viewer | View-only access to dashboard |

---

## Future Enhancements

- Speed detection (radar/camera-based)
- Helmet detection for motorcycles
- Wrong-way driving detection
- Parking violation monitoring
- Mobile app for field officers
- Cloud deployment (AWS/GCP)

---

## License

MIT License
