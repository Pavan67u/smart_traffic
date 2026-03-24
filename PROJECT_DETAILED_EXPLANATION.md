# Smart Traffic Project: Complete Technical Explanation

Last updated: 24 March 2026

## Progress Status

**All 28 core features fully implemented and deployed.**

### Recent Completions (24 March 2026)

✅ **Commit 7018027** — Frontend UI cleanup (removes all raw JSON/code visibility)
- Replaced all `<pre>` JSON dump blocks with human-readable `<div>` summaries
- Added formatted output helpers: `metricSummaryHtml()`, `rulesSummaryHtml()`, `signalSummaryHtml()`, `autoPolicySummaryHtml()`
- Changed export behavior to auto-download preset files instead of displaying JSON inline
- Updated `/dashboard` drift-health panel to show "Drift Risk | Confidence Trend | Fallback Ratio" instead of raw JSON
- Impact: 3 templates updated, 107 insertions, 20 deletions

✅ **Commit 77b9833** — Batch feature implementation (all 6 remaining features)
1. Auto-policy loop daemon for periodic signal suggestions per camera
2. Peak-hour weighting (1.15-1.20x boost during rush hours: 7-10 AM, 5-9 PM UTC)
3. Alert hooks (webhook + SMTP email) with 180s cooldown throttling
4. Incident priority scoring (0-100 scale) ranking violations by severity
5. Model drift health API tracking confidence trends and fallback ratio
6. Dashboard integration with priority-based violation sorting

✅ **Commit ca3a62d** — Signal policy expansion
- EWMA-smoothed signal timing suggestions (alpha=0.35)
- 45-second anti-flap hold window preventing rapid profile changes
- Peak-hour adaptive weighting integrated into suggestion algorithm

### Feature Implementation Summary

**Backend APIs** (22):
- Inference and detection (3)
- Tracking and rules (4)
- Database and persistence (3)
- Signal management (4)
- Alerts and notifications (3)
- Model health monitoring (2)
- Metrics and diagnostics (3)

**Frontend UI** (6):
- Inference page with camera selection and stream support
- Dashboard with status workflow and violation table
- Evidence carousel modal with zoom
- Charts (no external libs) with PNG export
- Dark mode toggle with persistence
- Calibration manager for camera profiles

### System State
- App running on port 5050 with full REST API
- SQLite violation database with 7 core fields + priority scoring
- YOLOv8n baseline + fine-tuning support for vehicle detection
- Production-ready UI (no visible raw code/JSON to end users)
- All diagnostics converted to human-readable summaries

---

## 1. What This Project Is

Smart Traffic is a computer vision pipeline for traffic violation analysis. It combines:

- object detection (YOLOv8),
- multi-object tracking (Ultralytics ByteTrack when available, IoU fallback otherwise),
- geometric rule evaluation (stop-line/red-light and lane-crossing style rules),
- evidence generation (cropped violation images),
- persistence (SQLite via SQLAlchemy),
- and a Flask web UI for upload, review, and report export.

In short: upload an image/video (or stream), run detection + tracking + rules, and produce actionable violation records with visual evidence.

## 2. Current Runtime Architecture

The active runtime path is centered on the Flask app in `web_app/app.py`.

### 2.1 Top-Level Flow

1. User uploads image/video or submits a live stream/webcam source.
2. Server runs YOLO inference:
   - `model.predict(...)` for image or fallback video mode,
   - `model.track(..., tracker="bytetrack.yaml", persist=True)` for tracked video mode.
3. Detections are normalized into a common structure:
   - `bbox=[x1,y1,x2,y2]`,
   - `score`,
   - `class`,
   - optional `track_id`.
4. `web_app/utils/tracking_manager.py` receives detections and runs:
   - tracking (if track IDs were not already provided),
   - rule engines per camera profile.
5. Violations (if any) generate crops and metadata.
6. Violations are written to DB (`Violation` model) and run output folders.
7. UI renders:
   - inference result media,
   - counts and recent events,
   - dashboard table with statuses and export actions.

### 2.2 Key Runtime Components

- `web_app/app.py`: Main server, routes, inference orchestration, DB writes, exports.
- `web_app/models.py`: SQLAlchemy model (`Violation`) and database object.
- `web_app/utils/tracking_manager.py`: Tracker selection, camera config loading, rule invocation.
- `rules/red_light.py`: `StopLineRule` implementation (polygon-zone + movement/stop logic).
- `rules/lane.py`: `LaneViolationRule` implementation (line crossing test).
- `web_app/utils/signal_manager.py`: Optional signal state provider (`RED/GREEN/YELLOW`) with timer/manual modes.

## 3. File/Folder Inventory and Purpose

## 3.1 Core Application

- `web_app/app.py`
  - Initializes Flask and SQLite DB.
  - Loads YOLO model from `MODEL_PATH` env var or fallback to `models/yolov8n.pt` then `yolov8n.pt`.
  - Builds class mapping from model classes to same-index target classes (identity map).
  - Handles image/video/stream inference paths.
  - Saves labels, events, result media.
  - Exposes API + export routes.

- `web_app/models.py`
  - `Violation` schema:
    - `id`, `timestamp`, `violation_type`, `image_path`, `vehicle_type`, `track_id`, `confidence`, `status`.
  - includes `to_dict()`.

- `web_app/templates/index.html`
  - inference page,
  - camera profile dropdown,
  - upload and live stream forms,
  - result media rendering,
  - dynamic count fetch,
  - theme toggle,
  - evidence modal (carousel + zoom),
  - lightweight visual analytics and chart PNG export.

- `web_app/templates/dashboard.html`
  - violations table,
  - status transition actions (`New -> Reviewed -> Sent`),
  - filters (type, vehicle, date range),
  - auto-refresh toggle,
  - summary stats,
  - chart PNG export,
  - evidence modal navigation.

- `web_app/static/`
  - `uploads/`: user inputs,
  - `results/`: run artifacts and per-run outputs,
  - `violations.db`: SQLite data file.

## 3.2 Rule Engines

- `rules/red_light.py`
  - Contains `StopLineRule`.
  - Works with `stop_zone` polygon (not just one line).
  - Uses centroid and bottom-center logic for movement/zone tests.
  - Rejects violations if signal is not red.
  - Applies stop-duration logic and minimum speed threshold.
  - Emits `.jsonl` events in `results/events`.

- `rules/lane.py`
  - Contains `LaneViolationRule`.
  - Detects side change across configured lane divider line.
  - Includes track age gating to reduce early false positives.
  - Emits lane events to `results/events/*_lane.jsonl`.

## 3.3 Tracking

- `web_app/utils/tracking_manager.py`
  - Chooses tracker implementation:
    - `tracking/bytetrack_wrapper.py` if available,
    - fallback `workers/iou_tracker.py`.
  - Loads camera geometry from `config/cameras.json`.
  - Builds rule engines per `camera_id` (cached).
  - `update_and_check(...)`:
    - accepts detection list,
    - reuses external `track_id` when present,
    - otherwise updates local tracker,
    - dispatches to each rule engine,
    - returns `tracks, violations`.

- `tracking/bytetrack_wrapper.py`
  - minimal adapter around external ByteTrack package (`bytetrack` import).
  - exposes `create_tracker()`.
  - may require installation/adaptation depending on ByteTrack package variant.

- `workers/iou_tracker.py`
  - CPU-friendly fallback tracker with greedy IoU matching.
  - Maintains track IDs, `lost` counters, and configurable `max_lost`.

## 3.4 Legacy/Prototype Worker Path

These are not the main web runtime path but are useful references:

- `workers/rules/red_light.py`: Shapely-based `RedLightRule` for another pipeline style.
- `workers/tracking/bytetrack_wrapper.py`: placeholder tracker wrapper.
- `workers/evidence/builder.py`: creates evidence packets with image hashes.
- `scripts/run_infer_demo.py`: old standalone flow combining vehicle model, light model, tracker, rule, evidence builder.

## 3.5 Config Files

- `config/cameras.json`
  - camera-specific geometry:
    - `stop_line_y` (legacy scalar representation),
    - `lane_line` endpoints,
    - optional `stop_zone` polygon.

- `config/mapping.json`
  - detector class remapping and thresholds for utility/smoke scripts.

- `configs/camera_sample.json`
  - sample geometry for legacy worker demo.

## 3.6 Training/Data Engineering

- `training/prepare_dataset.py`
  - end-to-end dataset bootstrap:
    - frame extraction,
    - auto-labeling,
    - train/val/test split.

- `training/finetune.py`
  - command-line model fine-tuning wrapper over Ultralytics.
  - supports `--dry-run`, custom model/data/project/name/device,
  - optional TensorBoard launch,
  - post-train validation and ONNX export attempt.

- `training/annotation_helper.py`
  - extracts frames and helps build ground-truth CSV from CLI annotations.

- `training/utils/auto_label_yolo.py`
  - auto labels images using pretrained YOLO.
  - maps predicted names into project target names:
    - `car`, `bus`, `truck`, `motorbike`, `person`.

- `training/utils/extract_frames.py`, `training/utils/split_dataset_v2.py`, `training/utils/split_datasets.py`
  - frame extraction and dataset split utilities.

- `training/yolo/data_vehicles.yaml`
  - main vehicles dataset config.

- `training/yolo/data_lights.yaml`
  - lights dataset config.

- `training/yolo/data_signs.yaml`
  - sign classification/detection class names (expanded label set).

- `training/yolo/train_vehicles.py`, `training/yolo/train_lights.py`
  - direct training scripts with fixed params.

## 3.7 Operations and Utilities

- `ops/retention.py`
  - archives old run folders and enforces `KEEP_LAST` policy.

- `scripts/run_saved_model.py`
  - quick inference using saved `models/best.pt`.

- `scripts/evaluate_violations.py`
  - computes precision/recall/F1 by matching predicted events to ground truth frame windows.

- `scripts/run_finetune.sh`
  - convenience wrapper for `training/finetune.py`.

- `scripts/monitor_finetune.sh`
  - tails latest run log and lists artifacts.

- `scripts/start_tensorboard.sh`
  - launches TensorBoard.

- `scripts/deploy_latest_model.sh`
  - copies latest run `best.pt` to `models/best_finetuned.pt`.

- `scripts/switch_model.sh`
  - toggles startup model mode (vehicles/signs), then starts web app with `MODEL_PATH`.

## 3.8 Infrastructure

- `Dockerfile`
  - Python 3.10 slim image,
  - OpenCV system dependencies,
  - pip install runtime packages,
  - copies project and starts Flask app.

- `docker-compose.yml`
  - single `web_app` service,
  - binds host port 5000 to container 5000,
  - mounts models and static upload/result volumes.

- `infra/requirements.txt`
  - dependency baseline for local/dev use.

## 4. API and Web Endpoints

Implemented in `web_app/app.py`:

- `GET /`
  - inference UI + camera config chooser.

- `POST /predict`
  - upload image/video and process.

- `POST /predict_video`
  - process uploaded video or stream/webcam URL.

- `GET /dashboard`
  - review violations table.

- `POST /dashboard/update/<id>`
  - status update workflow.

- `GET /_result_counts?path=/static/...`
  - aggregate class counts from YOLO txt labels.

- `GET /violations`
  - return per-run violations JSON (or aggregate jsonl events).

- `GET /api/recent_violations`
  - latest violations as JSON for AJAX polling.

- `GET /export/csv`
  - download full DB report as CSV.

- `GET /export/pdf`
  - download tabular PDF report.

- `GET /api/signal/status`, `POST /api/signal/set`
  - optional signal state APIs.

- `GET /api/health`
  - health response with model and signal info.

- `GET /static/<path>`
  - custom static file serving rooted at `web_app/static`.

## 5. Data Formats and Storage Contracts

## 5.1 Detection Object (internal)

Typical dict shape:

```json
{
  "bbox": [x1, y1, x2, y2],
  "score": 0.91,
  "class": 2,
  "track_id": 14
}
```

## 5.2 Track Object (internal)

Typical dict shape:

```json
{
  "track_id": 14,
  "bbox": [x1, y1, x2, y2],
  "cls": 2,
  "score": 0.91
}
```

## 5.3 Rule Event / Violation Object

Typical fields:

- `event_id`
- `event_type` (for example `red_light_violation`, `lane_violation`)
- `track_id`
- `class_id`
- `score`
- `frame_id`
- `timestamp`
- `bbox`
- `meta`
- optional `_crop_path` (added when evidence crop is saved)

## 5.4 DB Record

`Violation` stores a durable subset for dashboard/reporting.

## 5.5 Output Folder Layout (per run)

Representative structure:

```text
web_app/static/results/run_<uuid>/
  <annotated image or video>
  labels/*.txt
  events/*.jpg
  tracks.json
  violations.json
```

## 6. Model Handling Strategy

- Model path resolution:
  - `MODEL_PATH` environment variable if set,
  - else `models/yolov8n.pt` if present,
  - else Ultralytics default `yolov8n.pt`.

- Class handling:
  - runtime uses the loaded model's own class list (`model.names`),
  - `model_to_target_map` is identity by default.

Implication: if you switch to custom weights with different classes, UI labels and rule `vehicle_type` mapping follow that class list automatically.

## 7. Rule Logic Details

## 7.1 StopLineRule (`rules/red_light.py`)

- Requires red signal for violation checks.
- Computes movement speed between timestamps.
- Uses bottom-center point for zone inclusion.
- Violation criteria (high-level):
  - in stop zone,
  - moving above min speed,
  - did not stop long enough before crossing context,
  - track age above short ghost suppression threshold,
  - not in cooldown window (prevents repeated same-track spam).

## 7.2 LaneViolationRule (`rules/lane.py`)

- Maintains previous centroid per track.
- Calculates side-of-line sign for previous and current points.
- Opposite signs imply line crossing.
- Emits event only after minimum track age (false positive suppression).

## 8. Signal State Subsystem

`web_app/utils/signal_manager.py` provides:

- global singleton `SIGNAL_MANAGER`,
- `manual` and `timer` modes,
- cyclic transitions (`RED -> GREEN -> YELLOW -> RED`) in timer mode,
- thread-safe getters/setters via lock.

This lets rules react to red-phase semantics even without external traffic controller integration.

## 9. Testing and Validation Artifacts

- `tests/smoke_test.py`
  - validates mapping utilities and threshold access.

- `tests/test_tracking_and_rules.py`
  - simulates repeated frames from saved labels and exercises tracker/rule integration.

- `scripts/evaluate_violations.py`
  - computes TP/FP/FN and precision/recall/F1 against ground-truth CSV.

## 10. Known Inconsistencies and Engineering Debt

This repository has evolved through phases. Important realities:

- Multiple architecture generations coexist:
  - active Flask + `rules/*` path,
  - legacy `workers/*` + `scripts/run_infer_demo.py` path.

- Event naming differs in places:
  - code may emit `red_light_violation` while older docs mention `stop_line_violation`.

- Path conventions vary:
  - some utilities use `results/...`,
  - active web app writes to `web_app/static/results/...`.

- `Dockerfile` exposes/runs on port 5000 while direct app run uses 5050 in `app.run(...)`.

- ByteTrack wrappers are intentionally thin and may require package-specific adaptation.

These are not fatal, but they matter for deployment consistency and reproducibility.

## 11. End-to-End Operational Workflows

## 11.1 Run Web App (local)

1. Create/activate venv.
2. Install dependencies.
3. Optionally set `MODEL_PATH`.
4. Run `python web_app/app.py`.
5. Open UI and process media.

## 11.2 Prepare + Finetune Vehicles Model

1. Put videos in `training/videos/`.
2. Run `training/prepare_dataset.py`.
3. Verify `training/yolo/data_vehicles.yaml` paths.
4. Dry run `training/finetune.py --dry-run`.
5. Run training with device config.
6. Deploy latest `best.pt` to `models/` or set `MODEL_PATH`.

## 11.3 Evaluate Rule Quality

1. Create GT CSV (`training/annotation_helper.py`).
2. Produce predictions (`violations.json`).
3. Run `scripts/evaluate_violations.py --gt ... --pred ...`.
4. Inspect precision/recall/F1 and adjust geometry/rule thresholds.

## 12. Documentation Map

- `README.md`: quick start.
- `PROJECT_STRUCTURE.md`: active vs legacy split.
- `IMPLEMENTATION_GUIDE.md`: technical architecture narrative.
- `ACADEMIC_SUPPLEMENT.md`: benchmark/academic framing and evaluation templates.
- `NEW_FEATURES_DRAFT.md`: UI and feature enhancement summary.
- `PRESENTATION_DRAFT.md`: presentation-oriented talking points.

## 13. Practical Recommendations

If this project is being prepared for production/presentation grading:

1. Standardize event taxonomy (`red_light_violation` vs `stop_line_violation`).
2. Consolidate output paths under one canonical results root.
3. Align app runtime port with Docker compose mapping.
4. Add unit tests for rule edge cases and tracker fallback behavior.
5. Add migration/versioning for DB schema if expanding fields.
6. Decide and document one canonical ByteTrack backend.

## 14. One-Line Summary

This is a full-stack CV enforcement prototype where YOLO detections are tracked over time, interpreted through geometric traffic rules, persisted as violation records with evidence, and served through a modern Flask dashboard with export/reporting capabilities.