# Smart Traffic System - Technologies and Libraries Documentation

## Project Overview
A comprehensive intelligent traffic management system that uses computer vision and deep learning to detect vehicles, track their movement, enforce traffic rules (red light violations), and provide real-time evidence collection.

---

## Core Technologies

### 1. **Deep Learning & Computer Vision**
- **YOLO v8 (You Only Look Once)** - Object detection framework for vehicle detection
- **Ultralytics** v8.3.58 - PyTorch-based implementation of YOLOv8
- **OpenCV** v4.10.0.84 - Real-time computer vision library for video processing
- **NumPy** v1.26.4 - Numerical computing for array operations
- **SciPy** v1.11.4 - Scientific computing library

### 2. **Tracking & Motion Analysis**
- **ByteTrack** - Multi-object tracking algorithm for vehicle tracking
- **DeepSORT** - Deep learning-based sorting for tracking
- **LAPX** v0.5.9 - Linear assignment problem solver for tracking
- **Shapely** v2.1.2 - Geometric operations for traffic rule validation

### 3. **Backend & API**
- **FastAPI** v0.115.4 - Modern Python web framework for REST API
- **Uvicorn** v0.31.1 - ASGI web server
- **Pydantic** v2.9.2 - Data validation using Python type annotations

### 4. **Data Processing**
- **Pandas** v2.2.2 - Data manipulation and analysis
- **PyYAML** v6.0.2 - YAML configuration file parsing

### 5. **Database & Caching**
- **PostgreSQL** v15 - Primary relational database
- **psycopg2-binary** v2.9.9 - PostgreSQL database adapter for Python
- **Redis** v7 - In-memory data cache for real-time processing
- **redis** (Python) v5.0.8 - Redis client library

### 6. **Object Storage**
- **MinIO** - S3-compatible object storage server
- **boto3** v1.35.31 - AWS SDK for S3 and MinIO integration

### 7. **OCR (Optical Character Recognition)**
- **PaddleOCR** v2.7.3 - Text detection and recognition from images

### 8. **Containerization & Orchestration**
- **Docker** - Containerization platform
- **Docker Compose** v3.9 - Multi-container orchestration

### 9. **Programming Language**
- **Python** v3.11 - Primary programming language
- **Conda** - Environment and package management

---

## System Architecture Components

### Data Pipeline
```
Video Input 
    ↓
Frame Extraction (OpenCV)
    ↓
Image Labeling (LabelImg)
    ↓
Dataset Splitting (Train/Val/Test)
    ↓
YOLO Model Training (Ultralytics)
    ↓
Model Validation & Export (ONNX)
    ↓
Inference Pipeline
```

### Inference Pipeline
```
Live Video/Stream
    ↓
YOLO v8 Detection (Ultralytics)
    ↓
ByteTrack Multi-Object Tracking
    ↓
Rule Engine (Shapely for geometric analysis)
    ↓
Event Detection (Red light violations, overspeeding, etc.)
    ↓
Evidence Collection & Storage (MinIO S3)
    ↓
Database Recording (PostgreSQL)
    ↓
REST API (FastAPI)
    ↓
Frontend Dashboard/Reports
```

---

## Service Architecture (Docker Compose)

### Services Deployed
1. **API Service** - FastAPI application on port 8000
2. **Worker Service** - Background processing and detection
3. **PostgreSQL** - Database on port 5432
4. **Redis** - Caching layer on port 6379
5. **MinIO** - Object storage on ports 9000 & 9001

---

## Key Features & Technologies Used

### Vehicle Detection
- YOLOv8 nano model for real-time performance
- Multiple model sizes available (nano, small, medium, large)
- Custom training on traffic datasets

### Vehicle Tracking
- Continuous tracking of vehicles across frames
- Trajectory analysis using tracked paths
- Track ID management

### Traffic Rule Enforcement
- **Red Light Detection**: Geometric analysis using Shapely
- Stop line crossing detection
- Red zone polygon analysis
- Traffic light state tracking with debouncing

### Evidence Collection
- Bounding box extraction
- Timestamp recording
- S3 storage (MinIO) for evidence images
- Metadata recording in PostgreSQL

### Data Management
- YAML configuration files for data paths
- Structured dataset organization
- Train/validation/test splits

---

## Development & Deployment Tools

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Programming Language | 3.11 |
| Conda | Environment Management | Latest |
| Docker | Containerization | Latest |
| Docker Compose | Orchestration | 3.9 |
| Git | Version Control | Latest |
| LabelImg | Image Annotation | Latest |

---

## Model Export & Deployment Formats
- **ONNX** - For cross-platform inference
- **PyTorch** - Native model format
- **TensorFlow** - Optional conversion

---

## Performance Optimizations
- **Batch Processing** - Configured batch size of 8 for training
- **Image Size** - 640x640 for optimal speed/accuracy tradeoff
- **Data Augmentation** - HSV, translation, scale, perspective transforms
- **Mixed Precision** - Supported for faster training on compatible GPUs
- **Multi-worker Processing** - 8 workers for data loading

---

## Dataset & Training Configuration
- **Model**: YOLOv8 Nano
- **Epochs**: 10-100 (configurable)
- **Batch Size**: 8
- **Learning Rate**: 0.003
- **Optimizer**: AdamW
- **Augmentation**: Mosaic, HSV distortion, spatial transforms
- **Validation Split**: Standard train/val/test configuration

---

## Data Format & Standards
- **Image Format**: JPEG, PNG
- **Label Format**: YOLO format (.txt files with class and normalized coordinates)
- **Configuration Format**: YAML
- **API Format**: JSON (via Pydantic models)

---

## Additional Libraries & Dependencies
- **NumPy** - Numerical operations
- **SciPy** - Scientific computing
- **Pandas** - Data analysis and manipulation
- **PyYAML** - YAML parsing
- **Shapely** - Geometric operations and spatial analysis

---

## System Requirements for Development
- **Python**: 3.11+
- **RAM**: 8GB minimum (16GB recommended for training)
- **Storage**: 50GB+ (for datasets and models)
- **GPU** (Optional): CUDA-capable GPU for accelerated training
- **OS**: Windows, Linux, macOS

---

## References & Documentation
- [YOLO Documentation](https://docs.ultralytics.com)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PostgreSQL](https://www.postgresql.org/docs/)
- [Redis](https://redis.io/documentation)
- [Docker](https://docs.docker.com/)
- [Shapely](https://shapely.readthedocs.io/)

---

*Last Updated: November 17, 2025*
