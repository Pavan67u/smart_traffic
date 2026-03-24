# Smart Traffic Violation Detection System — Complete Draft (Report + Presentation)

> This draft is designed to give **full knowledge** of the project and can be copied directly into a **final report** or a **presentation**. It includes every feature, updated features, architecture, workflow, evaluation metrics, and extra notes.

---

# Part A — Full Project Draft (Report Style)

## 1) Project Overview
**Goal:** Build a complete end‑to‑end smart traffic violation detection system that can detect vehicles, track them across frames, apply rule‑based logic to identify violations, and store evidence in a dashboard for review and export.

**Key Outcomes:**
- Detect vehicles and people using YOLOv8.
- Track objects using ByteTrack (fallback IoU tracker).
- Detect stop‑line and lane violations.
- Save evidence crops + metadata in SQLite.
- Provide a full web dashboard for review and export.

---

## 2) System Architecture

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

## 3) Complete Workflow (End‑to‑End)

1. **Upload / Stream Input**
   - Image/video upload OR live webcam/RTSP.

2. **Object Detection**
   - YOLOv8 detects vehicles & people.

3. **Tracking**
   - ByteTrack assigns persistent track IDs.
   - Fallback IoU tracker if ByteTrack is unavailable.

4. **Rule Evaluation**
   - Stop‑line rule checks movement inside stop zone.
   - Lane rule checks line‑crossing.

5. **Evidence Capture**
   - Vehicle crop saved to `/static/results/run_xxx/events/`.

6. **Database Logging**
   - Violations stored in SQLite (`violations.db`).

7. **Dashboard Review**
   - Evidence preview, status change, export.

8. **Exports**
   - CSV and PDF generation.

---

## 4) Features (Full List)

### Core Features (Original)
- Image and video processing.
- Vehicle detection with YOLOv8.
- ByteTrack integration for tracking.
- Stop‑line and lane violation detection.
- Evidence cropping for violations.
- SQLite database for violations.
- Dashboard with review flow.
- CSV and PDF exports.

### Updated Features (New)
- **Modern UI redesign** (hero, cards, charts, modal).
- **Dark mode toggle** (persistent across sessions).
- **Recent violations live panel** (AJAX polling).
- **Charts + analytics** (violations by type, activity).
- **Chart PNG download** (canvas export).
- **Filters** (type, vehicle, date range) in dashboard.
- **Auto‑refresh toggle** on dashboard.
- **Evidence modal upgrade** (carousel + zoom).
- **Fine‑tuning tools** (finetune script, monitor, annotation helper).
- **Stop‑line rule fix** (prevents false negative on new track).
- **Class propagation fix** in tracking manager.

---

## 5) Technologies Used

| Layer | Tech |
|------|------|
| Backend | Flask, SQLAlchemy |
| Detection | Ultralytics YOLOv8 |
| Tracking | ByteTrack (YOLOv8), IoU fallback |
| CV | OpenCV |
| Database | SQLite |
| Exports | FPDF + CSV |
| Frontend | HTML / CSS / JS |
| Training | Ultralytics training API |
| Visualization | Canvas export (PNG) |
| Optional | TensorBoard |

---

## 6) Metrics (Verified from Current Project)

**Detection validation run:**
- Model: `yolov8n.pt`
- Dataset: `data/vehicles_yolo/val` (334 images)
- Command: `YOLO('yolov8n.pt').val(data='training/yolo/data_vehicles.yaml', imgsz=640, batch=8, device='cpu')`

### Detection Metrics
| Class | Precision (%) | Recall (%) | F1 (%) | AP@50 (%) |
|-------|---------------|------------|--------|-----------|
| Car | 0.15% | 100.00% | 0.29% | 0.77% |
| Bus | 0.00% | 0.00% | 0.00% | 0.00% |
| Truck | 0.00% | 0.00% | 0.00% | 0.00% |
| Motorbike | 0.00% | 0.00% | 0.00% | 0.00% |
| Person | 1.74% | 9.68% | 2.95% | 0.97% |
| **Mean (overall)** | **0.47%** | **27.42%** | **0.93%** | **0.44%** |

> Note: These are baseline metrics **before fine‑tuning**. The low precision/mAP indicates the pretrained model is not well aligned with the custom dataset and needs fine‑tuning.

### Violation Metrics
| Metric | Value |
|--------|-------|
| Stop‑line Precision | N/A (no labeled violation dataset) |
| Stop‑line Recall | N/A (no labeled violation dataset) |
| Stop‑line F1 | N/A (no labeled violation dataset) |
| Lane Precision | N/A (no labeled violation dataset) |
| Lane Recall | N/A (no labeled violation dataset) |
| Lane F1 | N/A (no labeled violation dataset) |

> Violation metrics require a ground‑truth violation dataset (frame‑level annotations). A manual annotation pipeline is included via `training/annotation_helper.py` to generate these values.

---

## 7) Module‑Wise Explanation (Key Files)

- **`web_app/app.py`** — Main Flask app, routes, detection pipeline.
- **`web_app/utils/tracking_manager.py`** — Tracking + rules integration.
- **`rules/red_light.py`** — Stop‑line violation logic.
- **`rules/lane.py`** — Lane violation logic.
- **`web_app/models.py`** — Database schema.
- **`training/finetune.py`** — Fine‑tune pipeline.
- **`training/annotation_helper.py`** — Ground truth annotation helper.
- **`web_app/templates/index.html`** — Inference UI.
- **`web_app/templates/dashboard.html`** — Violations dashboard.

---

## 8) Deployment Considerations
- Needs camera‑specific calibration.
- Requires fine‑tuning on local traffic data.
- Edge deployment (Jetson/RTX) for real‑time.
- Legal compliance requires evidence integrity.

---

# Part B — Presentation Style (Copy‑Paste Slides)

## Slide 1 — Title
**Smart Traffic Violation Detection System**
- YOLOv8 + ByteTrack + Rule Engine
- Evidence‑based violation logging

---

## Slide 2 — Problem Statement
- Manual traffic monitoring is inefficient.
- Need automated detection of violations.
- Focus: stop‑line + lane violations.

---

## Slide 3 — Architecture
- Input → Detection → Tracking → Rules → Evidence → Dashboard

---

## Slide 4 — Workflow
1. Upload or stream input
2. YOLOv8 detection
3. ByteTrack tracking
4. Rule engine (stop‑line, lane)
5. Evidence + DB logging
6. Dashboard review + export

---

## Slide 5 — Core Features
- YOLOv8 detection
- ByteTrack tracking
- Rule‑based violations
- Evidence cropping
- DB logging + exports

---

## Slide 6 — Updated Features
- Dark mode UI
- Analytics charts
- PNG chart export
- Filters + auto‑refresh toggle
- Evidence modal (carousel + zoom)
- Fine‑tuning tools

---

## Slide 7 — Technologies Used
- Flask, SQLAlchemy, SQLite
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- HTML/CSS/JS

---

## Slide 8 — Metrics (Template)
- Precision / Recall / F1 (stop‑line, lane)
- Detection AP@50

---

## Slide 9 — Limitations
- Domain gap (COCO vs local roads)
- Manual calibration
- No ANPR / legal enforcement

---

## Slide 10 — Future Scope
- ANPR integration
- Helmet / mobile detection
- Speed detection
- Edge deployment

---

# Part C — Extra Notes (Optional)

- Add demo video for viva.
- Show dashboard with evidence previews.
- Mention ITEMS system to validate real‑world relevance.
- Use evaluation results to finalize metrics tables.

---

*This is a complete draft. I can merge this into `IMPLEMENTATION_GUIDE.md` or convert it to a formal IEEE/UG report if needed.*
