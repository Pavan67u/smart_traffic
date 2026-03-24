# Smart Traffic System — Draft: New Features, Technologies, Workflow, and Full Explanation

## 1) Draft of New Features Added (Latest Changes)

### A. Frontend Experience (Inference + Dashboard)
- **Modern UI refresh** for `index.html` and `dashboard.html`.
  - New hero section, cards, structured layout, responsive grid.
  - Cleaner typography, shadows, gradients, and spacing for a professional look.
- **Dark mode toggle** with persistent preference (saved in `localStorage`).
- **Evidence modal upgrade**:
  - Carousel navigation (Prev/Next).
  - Zoom in/out.
  - Consistent design across inference and dashboard.
- **Charts (no external libraries)**:
  - **Inference page**: Violations by Type + Recent Activity.
  - **Dashboard**: Violation Mix (per-type distribution).
- **PNG chart export**:
  - Download charts as PNG using an offscreen `<canvas>`.
- **Loading skeletons** for recent violations, to avoid “blank” UI during fetch.
- **Auto-refresh toggle** on dashboard (replaces fixed 10s meta refresh).
- **Filters in dashboard**:
  - Violation type
  - Vehicle type
  - Date range
  - Filters update stats + chart automatically.

### B. Detection & Rules (Logic Improvements)
- **Stop-line rule fix**: Prevents new tracks from being marked as “stopped” on the first frame, which previously suppressed valid violations.
- **Correct class_id propagation** into rule engine (`tracking_manager.py`).
- **Duplicate text rendering removed** in video overlay.

### C. Training & Fine-tuning Tools
- **`training/finetune.py`**: Unified finetune script with arguments (device, epochs, batch, dry-run).
- **TensorBoard optional launch** from the finetune script.
- **`scripts/run_finetune.sh`**: one-command finetune runner.
- **`scripts/monitor_finetune.sh`**: live log tail + artifact visibility.
- **`training/annotation_helper.py`**: frame extraction + CLI annotation for ground-truth CSV.
- **`training/README_FINETUNE.md`**: clear instructions for training usage.

---

## 2) Technologies Used (With New Additions)

### Core Stack
- **Flask** — Web server, API routing, template rendering.
- **SQLAlchemy + SQLite** — Violation storage and retrieval.
- **OpenCV** — Image/video I/O, cropping, overlays.
- **Ultralytics YOLOv8** — Object detection + built-in tracking.
- **ByteTrack** (via YOLOv8 track API) — Multi-object tracking.

### Frontend
- **HTML/CSS/Vanilla JS** — Full UI (no external frameworks).
- **Canvas API** — Chart PNG export.
- **LocalStorage** — Save dark mode and auto-refresh preference.

### Training & Evaluation
- **Ultralytics training API** — Fine-tuning pipeline.
- **TensorBoard (optional)** — Training metrics visualization.

---

## 3) Complete Workflow (End-to-End)

1. **Input Source**
   - User uploads image/video OR uses live stream URL (RTSP/webcam).

2. **Detection**
   - YOLOv8 detects vehicles/persons.
   - Output: bounding boxes, class IDs, confidence.

3. **Tracking**
   - ByteTrack assigns consistent IDs per object across frames.

4. **Rules Engine**
   - Stop-line rule: detects crossing without proper stop.
   - Lane rule: detects lane boundary crossing.

5. **Evidence Capture**
   - Vehicle crop is saved in `static/results/run_xxx/events/`.

6. **Database Update**
   - Each violation stored in SQLite with metadata:
     - time, vehicle type, confidence, evidence path.

7. **Visualization**
   - Result annotated image/video saved per run.
   - Recent violations displayed in UI (auto fetch).

8. **Dashboard Review**
   - Violations table + evidence preview.
   - Status update workflow (New → Reviewed → Sent).

9. **Exports**
   - CSV and PDF reports from dashboard.

---

## 4) Detailed Explanation of the Application

### 4.1 Application Entry (`web_app/app.py`)
- Loads YOLO model once at startup.
- Initializes Flask + SQLite via SQLAlchemy.
- Defines key routes:
  - `/` → inference UI
  - `/predict` → upload image/video
  - `/predict_video` → live stream
  - `/dashboard` → violation review
  - `/api/recent_violations` → AJAX feed
  - `/export/csv` + `/export/pdf`

### 4.2 Detection Pipeline (Image)
- `model.predict()` returns bounding boxes.
- Draws overlays and saves output image.
- Creates YOLO format label files.
- Runs `update_and_check()` to evaluate violations.
- Saves crops and writes DB records.

### 4.3 Detection Pipeline (Video)
- `model.track()` returns detections with track IDs.
- Every frame is processed:
  - detections → tracking → rules → evidence
- Annotated video saved to `results/run_xxx/`.

### 4.4 Tracking & Rules (`tracking_manager.py`)
- Provides unified interface for detection → tracking → rule evaluation.
- Uses ByteTrack if available, else IoU fallback.
- Passes track state into rules with class, score, and bbox.

### 4.5 Rules Engine
- **StopLineRule** (`rules/red_light.py`)
  - Uses polygon stop-zone + velocity checks.
  - Uses signal state and stop duration.
- **LaneViolationRule** (`rules/lane.py`)
  - Detects crossing of configured line.

### 4.6 Evidence & DB
- Crops are saved under `static/results/run_xxx/events/`.
- DB stores path + metadata for review/export.

### 4.7 UI/UX
- **Inference Page**: upload, stream, results, recent violations, charts.
- **Dashboard**: filters, stats, charts, evidence modal, status updates.

---

## 5) Notes for Report / Viva
- Emphasize the **engineered workflow** rather than just demo UI.
- Show metrics or ground-truth evaluation once dataset is ready.
- Mention future integration with **ANPR and edge inference**.

---

*This is a draft. I can convert this into a full formal report section, or merge it into `IMPLEMENTATION_GUIDE.md` and `ACADEMIC_SUPPLEMENT.md` if needed.*
