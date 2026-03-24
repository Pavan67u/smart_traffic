from flask import Flask, render_template, request, send_from_directory, redirect, url_for, make_response, jsonify
from pathlib import Path
import uuid
import os
import cv2  # type: ignore
import numpy as np
import json
import io
import csv
from fpdf import FPDF  # type: ignore
from web_app.models import db, Violation
from typing import Any, Optional

# Type aliases for clarity
YOLO_Model = Any  # Ultralytics YOLO model
import time
try:
    from web_app.utils.signal_manager import SIGNAL_MANAGER
except ImportError:
    SIGNAL_MANAGER = None


APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / 'web_app' / 'static'
UPLOAD_DIR = STATIC_DIR / 'uploads'
RESULT_DIR = STATIC_DIR / 'results'

for d in (UPLOAD_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(APP_ROOT / 'web_app' / 'templates'))
# Use portable path for DB: inside web_app/static/ (which is already created)
db_path = APP_ROOT / 'web_app' / 'static' / 'violations.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()


# Load model at startup. Allow overriding via the MODEL_PATH env var.
# If MODEL_PATH is not set, default to the working pretrained model 'yolov8n.pt'
# which provides multi-class vehicle detections out of the box.
MODEL_PATH = os.environ.get('MODEL_PATH', None)
if MODEL_PATH is None:
    # prefer a local copy in models/ if present, else fall back to the ultralytics
    # pretrained name 'yolov8n.pt' (will be downloaded/used by ultralytics if needed)
    local_pref = APP_ROOT / 'models' / 'yolov8n.pt'
    if local_pref.exists():
        MODEL_PATH = str(local_pref)
    else:
        MODEL_PATH = 'yolov8n.pt'
else:
    # allow environment var to be either a relative/local path or a full path
    MODEL_PATH = str(MODEL_PATH)
# Target dataset class names (the app will map model output names to these)
# Target dataset class names (the app will use model output names)
TARGET_NAMES = []

# model -> target mapping will be computed at startup
model: Optional[Any] = None
model_to_target_map: dict = {}  # maps model_cls_id -> target_cls_id

# Only these class names are used for tracking/rule evaluation by default.
MONITORED_TRAFFIC_CLASSES = {
    'car', 'bus', 'truck', 'motorcycle', 'motorbike', 'bicycle',
    'auto', 'auto-rickshaw', 'autorickshaw'
}


def is_monitored_class(class_name: str) -> bool:
    return class_name.strip().lower() in MONITORED_TRAFFIC_CLASSES

try:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print('Loaded model:', MODEL_PATH)
    
    # Use model's own names
    if hasattr(model, 'names') and isinstance(model.names, dict):
        # Sort by ID to ensure consistent list order
        sorted_ids = sorted(model.names.keys())
        TARGET_NAMES = [str(model.names[i]) for i in sorted_ids]
        
        # Identity mapping since we are using the model's own classes
        for mid in model.names.keys():
            model_to_target_map[int(mid)] = int(mid)
            
    print('Model class names:', TARGET_NAMES)
except Exception as e:
    print('Failed to load model at startup:', MODEL_PATH, e)

# tracking + rules integration
try:
    from web_app.utils.tracking_manager import (
        update_and_check,
        CAMERA_CONFIG,
        get_runtime_metrics,
        get_rule_geometry,
        reload_camera_config,
    )
except Exception:
    update_and_check = None
    CAMERA_CONFIG = {}
    get_runtime_metrics = None
    get_rule_geometry = None
    reload_camera_config = None


DRAW_RULE_OVERLAY = os.environ.get('DRAW_RULE_OVERLAY', '1').strip() not in {'0', 'false', 'False'}


def draw_rule_overlay(frame, camera_id='default'):
    if frame is None or get_rule_geometry is None:
        return
    geom = get_rule_geometry(camera_id=camera_id, frame_shape=getattr(frame, 'shape', None)) or {}

    stop_zone = geom.get('stop_zone')
    if stop_zone and len(stop_zone) >= 3:
        pts = np.array(stop_zone, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.putText(frame, 'STOP ZONE', tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    lane_line = geom.get('lane_line')
    if lane_line and len(lane_line) == 2:
        p1 = tuple(map(int, lane_line[0]))
        p2 = tuple(map(int, lane_line[1]))
        cv2.line(frame, p1, p2, (255, 0, 255), 2)
        cv2.putText(frame, 'LANE LINE', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def process_video(input_path, output_path, run_id, camera_id='default'):
    """
    Process video frame by frame: inference -> track -> check rules -> draw -> save.
    Returns list of violations found.
    """
    # Accept pathlib.Path, string filepath/URL, or integer camera index
    from pathlib import Path as _Path
    if isinstance(input_path, _Path):
        input_path = str(input_path)
    # Check if input_path is a digit string (webcam index)
    if isinstance(input_path, str) and input_path.isdigit():
        cap = cv2.VideoCapture(int(input_path))
    else:
        cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Use standard mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    frame_idx = 0
    all_violations = []
    
    # FPS Benchmarking
    t_start = time.monotonic()
    frame_count_bf = 0
    inference_times = []
    
    while True:
        loop_start = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference - Use ByteTrack
        # persist=True ensures track IDs persist across frames in this loop
        try:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)
        except Exception:
            # fallback if track fails (e.g. strict config)
            results = model.predict(frame, conf=0.25, verbose=False)
        
        # Parse detections
        detections = []
        if results:
            r = results[0]
            if r.boxes:
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    b = box.xyxy[0].cpu().numpy()
                    c = int(box.cls)
                    s = float(box.conf)
                    # Extract track ID if available
                    tid = int(box.id[0]) if box.id is not None else None
                    
                    # Map to target class
                    mapped = model_to_target_map.get(c)
                    if mapped is None:
                        continue

                    cname = TARGET_NAMES[mapped] if mapped < len(TARGET_NAMES) else str(mapped)
                    if not is_monitored_class(cname):
                        continue
                    
                    x1, y1, x2, y2 = map(int, b[:4])
                    detections.append({
                        "bbox": [x1, y1, x2, y2], 
                        "score": s, 
                        "class": mapped,
                        "track_id": tid
                    })
                    
                    # Draw simple bbox for visual feedback
                    label = f"{cname} {s:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if DRAW_RULE_OVERLAY:
                    draw_rule_overlay(frame, camera_id=camera_id)

        # Tracking + Rules
        if update_and_check:
            # We pass frame_img=frame so rules can crop violations
            tracks, violations = update_and_check(run_id, camera_id, frame_idx, frame, detections, fps=fps)
            
            # Save evidence crops for any violations detected in this frame
            if violations:
                # Create events directory for this run
                evdir = output_path.parent / 'events'
                evdir.mkdir(parents=True, exist_ok=True)
                
                for ev in violations:
                    try:
                        bx = ev.get('bbox') or []
                        x1, y1, x2, y2 = map(int, bx)
                        if frame is not None and y2 > y1 and x2 > x1:
                            crop = frame[y1:y2, x1:x2]
                            cpath = evdir / f"{ev.get('event_id')}.jpg"
                            cv2.imwrite(str(cpath), crop)
                            ev['_crop_path'] = os.path.relpath(str(cpath), STATIC_DIR)
                    except Exception as e:
                        print(f"Error saving evidence crop: {e}")
                        ev['_crop_path'] = None
            
            all_violations.extend(violations)
            
            # Optional: Draw track IDs
            for t in tracks:
                tid = t['track_id']
                bbox = t['bbox']
                # Draw ID
                tx, ty = int(bbox[0]), int(bbox[1])
                cv2.putText(frame, f"ID: {tid}", (tx, ty+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Measure FPS
        loop_end = time.monotonic()
        proc_dur = loop_end - loop_start
        proc_fps = 1.0 / proc_dur if proc_dur > 0 else 0.0
        inference_times.append(proc_dur)
        
        # Overlay Signal State and FPS
        if SIGNAL_MANAGER:
            sig_status = SIGNAL_MANAGER.get_status()
            s_state = sig_status['state']
            s_color = (0, 0, 255) if s_state == 'RED' else (0, 255, 0)
            if s_state == 'YELLOW': s_color = (0, 255, 255)
            
            # Draw Traffic Light Indicator (Circle)
            cv2.circle(frame, (50, 50), 30, s_color, -1)
            cv2.putText(frame, s_state, (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, s_color, 2)
        
        cv2.putText(frame, f"Proc FPS: {proc_fps:.1f}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    return all_violations

@app.route('/')
def index():
    # Pass available camera keys for the dropdown
    cameras = CAMERA_CONFIG
    return render_template('index.html', cameras=cameras)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    camera_id = request.form.get('camera_id', 'default')

    uid = uuid.uuid4().hex
    filename = f'{uid}_{file.filename}'
    upload_path = UPLOAD_DIR / filename
    file.save(upload_path)

    if model is None:
        return 'Model not loaded on server. Check server logs.', 500

    # Run prediction in-memory and save outputs ourselves. Avoid Ultralytics' auto-save
    # to prevent duplicate annotations — we'll draw a single overlay using the
    # mapped target names and write remapped YOLO .txt labels.
    name = f'run_{uid}'
    out_dir = RESULT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = out_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    # Check for video file extensions
    ext = upload_path.suffix.lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video Processing Path
        output_video_path = out_dir / f"{filename}"
        
        try:
            violations_found = process_video(upload_path, output_video_path, name, camera_id=camera_id)
            
            # Persist violations to DB
            try:
                for ev in violations_found:
                    # _crop_path is relative to STATIC_DIR, set by tracking_manager/rules
                    new_violation = Violation(
                        violation_type=ev.get('event_type', 'unknown'),
                        image_path=ev.get('_crop_path'),
                        vehicle_type=TARGET_NAMES[ev.get('class_id')] if ev.get('class_id') is not None else 'unknown',
                        status='New',
                        track_id=ev.get('track_id'),
                        confidence=ev.get('score')
                    )
                    db.session.add(new_violation)
                db.session.commit()
            except Exception as e:
                print('DB Error saving video violations:', e)
                
            # Redirect to result
            video_rel = os.path.relpath(str(output_video_path), STATIC_DIR)
            return redirect(url_for('index', result=video_rel))
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return f"Error processing video: {e}", 500

    # Image Processing Path (Fallthrough)
    try:
        results = model.predict(source=str(upload_path), conf=0.25, save=False)

    except Exception:
        # fallback single-call
        results = model(str(upload_path), conf=0.25)

    img_rel = None
    try:
        res = results[0]
        b = getattr(res, 'boxes', None)
        boxes_xyxy = None
        confs = None
        cls_ids = None
        if b is not None:
            try:
                boxes_xyxy = b.xyxy.cpu().numpy()
            except Exception:
                boxes_xyxy = np.array(b.xyxy) if hasattr(b, 'xyxy') else None
            try:
                confs = b.conf.cpu().numpy()
            except Exception:
                confs = np.array(b.conf) if hasattr(b, 'conf') else None
            try:
                cls_ids = b.cls.cpu().numpy()
            except Exception:
                cls_ids = np.array(b.cls) if hasattr(b, 'cls') else None

        # keep an unmodified copy for crops/evidence; we'll draw overlays on `img`
        img = cv2.imread(str(upload_path))
        orig_img = img.copy() if img is not None else None
        h, w = img.shape[:2]
        saved_img_path = out_dir / (f"{uid}_{upload_path.name}")
        yolo_lines = []
        detections = []
        if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            for i, xy in enumerate(boxes_xyxy):
                try:
                    x1, y1, x2, y2 = map(int, xy[:4])
                except Exception:
                    continue
                cid = int(cls_ids[i]) if cls_ids is not None else None
                conf = float(confs[i]) if confs is not None else None
                mapped = None
                if cid is not None:
                    mapped = model_to_target_map.get(int(cid)) if model_to_target_map else None
                if mapped is None:
                    continue
                cname = TARGET_NAMES[mapped] if mapped < len(TARGET_NAMES) else str(mapped)
                if not is_monitored_class(cname):
                    continue
                text = f"{cname} {conf:.2f}" if conf is not None else f"{cname}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                ty = max(0, y1 - th - baseline - 4)
                cv2.rectangle(img, (x1, ty), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
                yolo_lines.append(f"{mapped} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                # normalized detection format for tracker: bbox ints, score, class
                detections.append({"bbox": [x1, y1, x2, y2], "score": conf if conf is not None else 1.0, "class": mapped})

        if DRAW_RULE_OVERLAY:
            draw_rule_overlay(img, camera_id=camera_id)

        cv2.imwrite(str(saved_img_path), img)
        img_rel = os.path.relpath(str(saved_img_path), STATIC_DIR)
        if yolo_lines:
            label_file = label_dir / (saved_img_path.stem + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')

        # call tracker+rules (if available)
        if update_and_check is not None:
            try:
                tracks, violations = update_and_check(run_id=name, camera_id=camera_id, frame_idx=0, frame_img=img, detections=detections)
                # persist tracks and violations for quick lookup by UI
                try:
                    (out_dir / 'tracks.json').write_text(json.dumps(tracks, indent=2))
                except Exception:
                    pass
                if violations:
                    # per-run events folder (crops & index)
                    evdir = out_dir / 'events'
                    evdir.mkdir(parents=True, exist_ok=True)
                    ev_index = []
                    for ev in violations:
                        # save crop if bbox available
                        try:
                            bx = ev.get('bbox') or []
                            x1, y1, x2, y2 = map(int, bx)
                            if orig_img is not None and y2 > y1 and x2 > x1:
                                crop = orig_img[y1:y2, x1:x2]
                                cpath = evdir / (f"{ev.get('event_id')}.jpg")
                                cv2.imwrite(str(cpath), crop)
                                ev['_crop_path'] = os.path.relpath(str(cpath), STATIC_DIR)
                        except Exception:
                            ev['_crop_path'] = None
                        ev_index.append(ev)
                    try:
                        (out_dir / 'violations.json').write_text(json.dumps(ev_index, indent=2))
                    except Exception:
                        pass
                    
                    # Persist to Database
                    try:
                        for ev in ev_index:
                            new_violation = Violation(
                                violation_type=ev.get('event_type', 'unknown'),
                                image_path=ev.get('_crop_path'),
                                vehicle_type=TARGET_NAMES[ev.get('class_id')] if ev.get('class_id') is not None and ev.get('class_id') < len(TARGET_NAMES) else 'unknown',
                                status='New',
                                track_id=ev.get('track_id'),
                                confidence=ev.get('score')
                            )
                            db.session.add(new_violation)
                        db.session.commit()
                    except Exception as e:
                        print('DB Error:', e)
                        
            except Exception:

                # don't break prediction on tracker/rule errors
                tracks, violations = [], []
    except Exception:
        try:
            saved = out_dir / upload_path.name
            upload_path.replace(saved)
            img_rel = os.path.relpath(str(saved), STATIC_DIR)
        except Exception:
            img_rel = None

    if img_rel:
        return redirect(url_for('index', result=img_rel))

    # fallback: if no image saved, render index with an error
    return render_template('index.html', img_url=None, class_counts={},)


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Accept a video file upload (form field 'file') or a camera/stream URL (form field 'url')
    and process it frame-by-frame using the same pipeline as `process_video`.
    Returns the annotated video result page on success.
    """
    if model is None:
        return 'Model not loaded on server. Check server logs.', 500

    # allow either an uploaded file or a stream URL
    input_path = None
    filename = None
    uid = uuid.uuid4().hex
    camera_id = request.form.get('camera_id', 'default')
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        filename = f"{uid}_{file.filename}"
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)
        input_path = upload_path
    else:
        url = request.form.get('url')
        if not url:
            return 'No file or url provided', 400
        # use URL directly (e.g., rtsp/http) — VideoCapture can open many stream URLs
        input_path = url
        # create a friendly filename for saving results
        filename = f"{uid}_stream.mp4"

    name = f'run_{uid}'
    out_dir = RESULT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = out_dir / filename
    try:
        violations_found = process_video(input_path, output_video_path, name, camera_id=camera_id)

        # Persist violations to DB
        try:
            for ev in violations_found:
                new_violation = Violation(
                    violation_type=ev.get('event_type', 'unknown'),
                    image_path=ev.get('_crop_path'),
                    vehicle_type=TARGET_NAMES[ev.get('class_id')] if ev.get('class_id') is not None and ev.get('class_id') < len(TARGET_NAMES) else 'unknown',
                    status='New',
                    track_id=ev.get('track_id'),
                    confidence=ev.get('score')
                )
                db.session.add(new_violation)
            db.session.commit()
        except Exception as e:
            print('DB Error saving video violations:', e)

        # Redirect to result page same as /predict
        video_rel = os.path.relpath(str(output_video_path), STATIC_DIR)
        return redirect(url_for('index', result=video_rel))

    except Exception as e:
        print(f"Error processing video/stream: {e}")
        return f"Error processing video/stream: {e}", 500

@app.route('/dashboard')
def dashboard():
    # Fetch all violations ordered by latest first
    violations = Violation.query.order_by(Violation.timestamp.desc()).all()
    return render_template('dashboard.html', violations=violations)

@app.route('/dashboard/update/<int:vid>', methods=['POST'])
def update_status(vid):
    v = Violation.query.get_or_404(vid)
    new_status = request.form.get('status')
    if new_status:
        v.status = new_status
        db.session.commit()
    return redirect(url_for('dashboard'))


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve project static files rooted at `web_app/static`.

    We use a custom route so that `/static/<path>` serves files from
    the `STATIC_DIR` directory. The template constructs URLs like
    `/static/results/run_xxx/imagename.jpg` where the path after
    `/static/` is relative to `STATIC_DIR`.
    """
    return send_from_directory(str(STATIC_DIR), filename)


@app.route('/_result_counts')
def result_counts():
    """Return JSON counts for a given result image path (img_url prefixed with /static/...).
    Query param: path=/static/....jpg
    """
    p = request.args.get('path')
    if not p:
        return jsonify({})

    # p is expected to be something like '/static/results/run_xxx/imagename.jpg'
    # remove the leading '/static/' and compute the labels directory under STATIC_DIR
    if p.startswith('/static/'):
        rel = p[len('/static/'):]
    else:
        rel = p
    rel_path = Path(rel)
    out_dir = STATIC_DIR / rel_path.parent
    label_dir = out_dir / 'labels'
    class_counts = {}
    if label_dir.exists():
        for txt in label_dir.glob('*.txt'):
            with open(txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except Exception:
                        continue
                    cname = TARGET_NAMES[cid] if cid < len(TARGET_NAMES) else str(cid)
                    class_counts[cname] = class_counts.get(cname, 0) + 1
    return jsonify(class_counts)


@app.route('/violations')
def violations():
    """Return recent violations. Optional query param `run` to return per-run violations.json
    If `run` is not provided, aggregate events from the global events folder (rules output).
    """
    run = request.args.get('run')
    if run:
        # expect run to be like 'run_<uid>'
        out = RESULT_DIR / run / 'violations.json'
        if out.exists():
            try:
                return jsonify(json.loads(out.read_text()))
            except Exception:
                return jsonify([])
        return jsonify([])

    # aggregate recent from rules' EVENTS_DIR
    try:
        from rules import red_light
        ev_files = list(red_light.EVENTS_DIR.glob('*.jsonl'))
        events = []
        for f in ev_files:
            with open(f, 'r') as fh:
                for line in fh:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        return jsonify(events)
    except Exception:
        return jsonify([])

@app.route('/api/recent_violations')
def api_recent_violations():
    """Return recent violations from DB as JSON for AJAX polling."""
    # Get last 10 violations
    violations = Violation.query.order_by(Violation.timestamp.desc()).limit(10).all()
    result = []
    for v in violations:
        result.append({
            'id': v.id,
            'timestamp': v.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'violation_type': v.violation_type,
            'vehicle_type': v.vehicle_type or 'Unknown',
            'confidence': f"{v.confidence:.2f}" if v.confidence else None,
            'track_id': v.track_id,
            'status': v.status,
            'image_path': v.image_path
        })
    return jsonify(result)

@app.route('/export/csv')
def export_csv():
    violations = Violation.query.order_by(Violation.timestamp.desc()).all()
    
    si = io.StringIO()
    cw = csv.writer(si)
    # Header
    cw.writerow(['ID', 'Timestamp', 'Type', 'Vehicle', 'Confidence', 'Track ID', 'Status'])
    
    for v in violations:
        cw.writerow([
            v.id,
            v.timestamp.isoformat(),
            v.violation_type,
            v.vehicle_type,
            f"{v.confidence:.2f}" if v.confidence else "",
            v.track_id if v.track_id else "",
            v.status
        ])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=violations_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/export/pdf')
def export_pdf():
    violations = Violation.query.order_by(Violation.timestamp.desc()).all()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Title
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, "Traffic Violation Report", ln=True, align='C')
    pdf.ln(10)
    
    # Table Header
    pdf.set_font("Arial", 'B', size=10)
    col_widths = [15, 45, 30, 25, 20, 20, 20] # ID, Time, Type, Veh, Conf, Trk, Stat
    headers = ['ID', 'Timestamp', 'Type', 'Vehicle', 'Conf', 'TrkID', 'Status']
    
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, border=1)
    pdf.ln()
    
    # Data
    pdf.set_font("Arial", size=9)
    for v in violations:
        row = [
            str(v.id),
            v.timestamp.strftime('%Y-%m-%d %H:%M'),
            str(v.violation_type)[:15] if v.violation_type else "-",  # Truncate
            str(v.vehicle_type) if v.vehicle_type else "-",
            f"{v.confidence:.2f}" if v.confidence else "-",
            str(v.track_id) if v.track_id else "-",
            str(v.status) if v.status else "-"
        ]
        for i, r in enumerate(row):
            pdf.cell(col_widths[i], 10, str(r), border=1)
        pdf.ln()
    
    # Get PDF bytes - handle both old and new FPDF versions
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        pdf_bytes = pdf_output.encode('latin-1')
    else:
        pdf_bytes = bytes(pdf_output)
    
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=violations_report.pdf'
    return response

# --- Signal API ---
@app.route('/api/signal/status')
def signal_status():
    if SIGNAL_MANAGER:
        return jsonify(SIGNAL_MANAGER.get_status())
    return jsonify({'error': 'Signal Manager not available'}), 503

@app.route('/api/signal/set', methods=['POST'])
def signal_set():
    if not SIGNAL_MANAGER:
        return jsonify({'error': 'Signal Manager not available'}), 503
    data = request.json or {}
    mode = data.get('mode')
    state = data.get('state')
    
    if mode:
        SIGNAL_MANAGER.set_mode(mode)
    if state:
        SIGNAL_MANAGER.set_state(state)
        
    return jsonify(SIGNAL_MANAGER.get_status())

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'signal': SIGNAL_MANAGER.get_status() if SIGNAL_MANAGER else None,
        'db': 'connected',
        'model': str(MODEL_PATH)
    })


@app.route('/api/metrics')
def metrics():
    if get_runtime_metrics is None:
        return jsonify({})
    run_id = request.args.get('run_id')
    camera_id = request.args.get('camera_id')
    if run_id and camera_id:
        return jsonify(get_runtime_metrics(run_id=run_id, camera_id=camera_id))
    return jsonify(get_runtime_metrics())


@app.route('/api/debug/rules')
def debug_rules():
    if get_rule_geometry is None:
        return jsonify({})
    camera_id = request.args.get('camera_id', 'default')
    w = request.args.get('w', type=int)
    h = request.args.get('h', type=int)
    frame_shape = (h, w, 3) if w and h else None
    return jsonify(get_rule_geometry(camera_id=camera_id, frame_shape=frame_shape))


@app.route('/api/camera/calibration', methods=['POST'])
def save_camera_calibration():
    data = request.json or {}
    target_camera_id = (data.get('camera_id') or 'default').strip()
    save_as_camera_id = (data.get('save_as_camera_id') or '').strip()
    camera_id = save_as_camera_id or target_camera_id
    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400

    try:
        width = int(data.get('width'))
        height = int(data.get('height'))
    except Exception:
        return jsonify({'error': 'width and height are required integers'}), 400

    if width <= 0 or height <= 0:
        return jsonify({'error': 'width and height must be > 0'}), 400

    cfg_path = APP_ROOT / 'config' / 'cameras.json'
    try:
        if cfg_path.exists():
            camera_cfg = json.loads(cfg_path.read_text())
        else:
            camera_cfg = {}
    except Exception as e:
        return jsonify({'error': f'Failed to read camera config: {e}'}), 500

    source_camera_id = target_camera_id if target_camera_id in camera_cfg else 'default'
    existing = camera_cfg.get(source_camera_id, {}) if save_as_camera_id else camera_cfg.get(camera_id, {})
    updated = dict(existing)
    updated['reference_resolution'] = [width, height]

    description = data.get('description')
    if description is not None:
        updated['description'] = str(description)

    stop_line_y = data.get('stop_line_y')
    if stop_line_y is not None:
        try:
            stop_line_y = int(round(float(stop_line_y)))
        except Exception:
            return jsonify({'error': 'stop_line_y must be numeric'}), 400
        stop_line_y = max(0, min(height, stop_line_y))
        updated['stop_line_y'] = stop_line_y
        updated['stop_zone'] = [[0, stop_line_y], [width, stop_line_y], [width, height], [0, height]]

    lane_line = data.get('lane_line')
    if lane_line is not None:
        if not isinstance(lane_line, list) or len(lane_line) != 2:
            return jsonify({'error': 'lane_line must be [[x1,y1],[x2,y2]]'}), 400
        try:
            p1 = [int(round(float(lane_line[0][0]))), int(round(float(lane_line[0][1])))]
            p2 = [int(round(float(lane_line[1][0]))), int(round(float(lane_line[1][1])))]
        except Exception:
            return jsonify({'error': 'lane_line points must be numeric'}), 400
        p1[0] = max(0, min(width, p1[0]))
        p1[1] = max(0, min(height, p1[1]))
        p2[0] = max(0, min(width, p2[0]))
        p2[1] = max(0, min(height, p2[1]))
        updated['lane_line'] = [p1, p2]

    camera_cfg[camera_id] = updated
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(camera_cfg, indent=2))
    except Exception as e:
        return jsonify({'error': f'Failed to write camera config: {e}'}), 500

    if reload_camera_config is not None:
        reload_camera_config()

    return jsonify({'ok': True, 'camera_id': camera_id, 'config': updated})


@app.route('/api/camera/profiles')
def camera_profiles():
    profiles = {}
    for cam_id, cfg in (CAMERA_CONFIG or {}).items():
        profiles[cam_id] = {
            'description': cfg.get('description', cam_id),
            'reference_resolution': cfg.get('reference_resolution'),
            'stop_line_y': cfg.get('stop_line_y'),
            'lane_line': cfg.get('lane_line'),
        }
    return jsonify({'profiles': profiles})


@app.route('/api/camera/preset')
def camera_preset():
    camera_id = request.args.get('camera_id', 'default')
    cfg = (CAMERA_CONFIG or {}).get(camera_id)
    if not cfg:
        return jsonify({'error': f'camera_id not found: {camera_id}'}), 404
    return jsonify({'camera_id': camera_id, 'config': cfg})


@app.route('/api/camera/preset/import', methods=['POST'])
def import_camera_preset():
    data = request.json or {}
    camera_id = (data.get('camera_id') or '').strip()
    config = data.get('config')
    overwrite = bool(data.get('overwrite', False))

    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400
    if not isinstance(config, dict):
        return jsonify({'error': 'config must be an object'}), 400

    cfg_path = APP_ROOT / 'config' / 'cameras.json'
    try:
        if cfg_path.exists():
            camera_cfg = json.loads(cfg_path.read_text())
        else:
            camera_cfg = {}
    except Exception as e:
        return jsonify({'error': f'Failed to read camera config: {e}'}), 500

    if (camera_id in camera_cfg) and (not overwrite):
        return jsonify({'error': f'camera_id already exists: {camera_id}. Set overwrite=true to replace.'}), 409

    camera_cfg[camera_id] = config
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(camera_cfg, indent=2))
    except Exception as e:
        return jsonify({'error': f'Failed to write camera config: {e}'}), 500

    if reload_camera_config is not None:
        reload_camera_config()

    return jsonify({'ok': True, 'camera_id': camera_id})

if __name__ == '__main__':
    # Use 0.0.0.0 for Docker visibility, but 5050 to avoid MacOS AirPlay conflict
    app.run(host='0.0.0.0', port=5050, debug=True)
