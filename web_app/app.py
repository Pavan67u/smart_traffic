from flask import Flask, render_template, request, send_from_directory, redirect, url_for, make_response, jsonify, session
from pathlib import Path
import uuid
import os
import urllib.request
import cv2  # type: ignore
import numpy as np
import json
import io
import csv
import smtplib
from fpdf import FPDF  # type: ignore
from email.message import EmailMessage
from web_app.models import db, Violation, User, AuditLog
from web_app.utils.auth import require_login, require_permission, require_role, audit_action, init_default_users
from typing import Any, Optional
import threading

# Type aliases for clarity
YOLO_Model = Any  # Ultralytics YOLO model
import time
try:
    from web_app.utils.signal_manager import SIGNAL_MANAGER
except ImportError:
    SIGNAL_MANAGER = None

# Import heatmap and emergency detector
try:
    from web_app.utils.heatmap import DensityHeatmap, VehicleCounter
except ImportError:
    DensityHeatmap = None
    VehicleCounter = None

EmergencyVehicleDetector = None

try:
    from web_app.utils.alerts import emit_multi_channel_alert, check_twilio_config
except ImportError:
    emit_multi_channel_alert = None
    check_twilio_config = None

# Global instances for heatmap and emergency detection
HEATMAP_INSTANCES = {}  # run_id -> DensityHeatmap
VEHICLE_COUNTERS = {}   # run_id -> VehicleCounter
EMERGENCY_DETECTORS = {}  # run_id -> EmergencyVehicleDetector


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
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
db.init_app(app)

with app.app_context():
    db.create_all()
    init_default_users(app)


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

# Signal timing suggestion smoothing and anti-flapping controls.
SIGNAL_POLICY_SMOOTHING_ALPHA = float(os.environ.get('SIGNAL_POLICY_SMOOTHING_ALPHA', '0.35'))
SIGNAL_POLICY_HOLD_SECONDS = int(os.environ.get('SIGNAL_POLICY_HOLD_SECONDS', '45'))
SIGNAL_POLICY_STATE = {}
SIGNAL_POLICY_LOCK = threading.Lock()

ALERT_WEBHOOK_URL = os.environ.get('ALERT_WEBHOOK_URL', '').strip()
ALERT_COOLDOWN_SECONDS = int(os.environ.get('ALERT_COOLDOWN_SECONDS', '180'))
ALERT_STATE = {}

ALERT_SMTP_HOST = os.environ.get('ALERT_SMTP_HOST', '').strip()
ALERT_SMTP_PORT = int(os.environ.get('ALERT_SMTP_PORT', '587'))
ALERT_SMTP_USER = os.environ.get('ALERT_SMTP_USER', '').strip()
ALERT_SMTP_PASS = os.environ.get('ALERT_SMTP_PASS', '').strip()
ALERT_EMAIL_TO = os.environ.get('ALERT_EMAIL_TO', '').strip()
ALERT_EMAIL_FROM = os.environ.get('ALERT_EMAIL_FROM', ALERT_SMTP_USER).strip()

AUTO_POLICY_DEFAULT_INTERVAL = int(os.environ.get('AUTO_POLICY_DEFAULT_INTERVAL', '20'))
AUTO_POLICY_STATE = {}
AUTO_POLICY_LOCK = threading.Lock()
AUTO_POLICY_STOP_EVENT = threading.Event()
AUTO_POLICY_THREAD = None


def _priority_score(violation):
    vtype = (violation.violation_type or '').lower()
    status = (violation.status or '').lower()
    vehicle = (violation.vehicle_type or '').lower()

    base = 35
    if 'red' in vtype:
        base = 72
    elif 'stop' in vtype:
        base = 58
    elif 'lane' in vtype:
        base = 45

    status_bonus = {'new': 15, 'reviewed': 6, 'sent': 0}.get(status, 0)
    conf = float(violation.confidence or 0.0)
    conf_bonus = int(max(0.0, min(conf, 1.0)) * 15)
    vehicle_bonus = 6 if vehicle in {'bus', 'truck'} else 0

    recency_bonus = 0
    try:
        age_s = max(0.0, time.time() - violation.timestamp.timestamp())
        recency_bonus = max(0, int(20 - (age_s / 3600.0)))
    except Exception:
        recency_bonus = 0

    score = max(0, min(100, base + status_bonus + conf_bonus + vehicle_bonus + recency_bonus))
    if score >= 75:
        level = 'high'
    elif score >= 50:
        level = 'medium'
    else:
        level = 'low'
    return score, level


def _peak_hour_weight(now_ts=None):
    lt = time.localtime(now_ts or time.time())
    # Typical city peaks: morning and evening rush.
    is_peak = (7 <= lt.tm_hour <= 10) or (17 <= lt.tm_hour <= 21)
    if is_peak:
        return {
            'is_peak_hour': True,
            'tracks_weight': 1.15,
            'violation_weight': 1.20,
            'tag': 'rush_hour',
        }
    return {
        'is_peak_hour': False,
        'tracks_weight': 1.0,
        'violation_weight': 1.0,
        'tag': 'normal_hour',
    }


def _should_send_alert(key):
    now_ts = time.time()
    last = float(ALERT_STATE.get(key, 0.0))
    if now_ts - last < ALERT_COOLDOWN_SECONDS:
        return False
    ALERT_STATE[key] = now_ts
    return True


def _send_webhook_alert(payload):
    if not ALERT_WEBHOOK_URL:
        return {'sent': False, 'reason': 'webhook_not_configured'}
    try:
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return {'sent': True, 'status_code': getattr(resp, 'status', 200)}
    except Exception as e:
        return {'sent': False, 'reason': str(e)}


def _send_email_alert(subject, payload):
    if not (ALERT_SMTP_HOST and ALERT_EMAIL_TO and ALERT_EMAIL_FROM):
        return {'sent': False, 'reason': 'email_not_configured'}
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = ALERT_EMAIL_FROM
        msg['To'] = ALERT_EMAIL_TO
        msg.set_content(json.dumps(payload, indent=2))

        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT, timeout=8) as s:
            s.starttls()
            if ALERT_SMTP_USER and ALERT_SMTP_PASS:
                s.login(ALERT_SMTP_USER, ALERT_SMTP_PASS)
            s.send_message(msg)
        return {'sent': True}
    except Exception as e:
        return {'sent': False, 'reason': str(e)}


def _emit_alerts(scope_key, payload):
    if not _should_send_alert(scope_key):
        return {'suppressed': True, 'cooldown_seconds': ALERT_COOLDOWN_SECONDS}
    wh = _send_webhook_alert(payload)
    em = _send_email_alert('Smart Traffic Incident Alert', payload)
    return {'suppressed': False, 'webhook': wh, 'email': em}


def _ensure_auto_policy_thread():
    global AUTO_POLICY_THREAD
    if AUTO_POLICY_THREAD is not None and AUTO_POLICY_THREAD.is_alive():
        return

    def _loop():
        while not AUTO_POLICY_STOP_EVENT.is_set():
            try:
                with AUTO_POLICY_LOCK:
                    items = [(k, dict(v)) for k, v in AUTO_POLICY_STATE.items() if v.get('enabled')]

                now_ts = time.time()
                for camera_id, cfg in items:
                    interval = max(5, int(cfg.get('interval_seconds', AUTO_POLICY_DEFAULT_INTERVAL)))
                    last_applied = float(cfg.get('last_applied_ts', 0.0))
                    if now_ts - last_applied < interval:
                        continue

                    run_id = cfg.get('run_id')
                    mode = (cfg.get('mode') or 'timer').strip().lower()
                    force_change = bool(cfg.get('force', False))

                    metrics = _aggregate_runtime_metrics(run_id=run_id, camera_id=camera_id)
                    scope_key = f"{camera_id or 'global'}::{run_id or 'all'}"
                    suggestion = _build_signal_timing_suggestion(metrics, scope_key=scope_key, force_profile_change=force_change)

                    if SIGNAL_MANAGER:
                        SIGNAL_MANAGER.set_durations(
                            red=suggestion.get('durations', {}).get('RED'),
                            green=suggestion.get('durations', {}).get('GREEN'),
                            yellow=suggestion.get('durations', {}).get('YELLOW'),
                        )
                        if mode in {'manual', 'timer'}:
                            SIGNAL_MANAGER.set_mode(mode)

                    vrate = float(suggestion.get('derived', {}).get('violation_rate_ewma', 0.0) or 0.0)
                    if suggestion.get('profile') == 'high_congestion' or vrate >= 0.06:
                        payload = {
                            'event': 'high_incident_risk',
                            'camera_id': camera_id,
                            'run_id': run_id,
                            'suggestion': suggestion,
                            'signal': SIGNAL_MANAGER.get_status() if SIGNAL_MANAGER else None,
                            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                        }
                        _emit_alerts(f"{scope_key}:incident", payload)

                    with AUTO_POLICY_LOCK:
                        cur = AUTO_POLICY_STATE.get(camera_id, {})
                        cur['last_applied_ts'] = now_ts
                        cur['last_applied'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))
                        cur['last_suggestion'] = suggestion
                        AUTO_POLICY_STATE[camera_id] = cur
            except Exception as e:
                print(f"[auto_policy] loop error: {e}")

            AUTO_POLICY_STOP_EVENT.wait(1.0)

    AUTO_POLICY_THREAD = threading.Thread(target=_loop, daemon=True)
    AUTO_POLICY_THREAD.start()


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

    # Direction Zone for Wrong-Way Detection
    direction_zone = geom.get('direction_zone')
    if direction_zone and len(direction_zone) >= 3:
        pts = np.array(direction_zone, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 100, 0), thickness=2)
        # Draw expected direction arrow
        expected_dir = geom.get('expected_direction')
        if expected_dir:
            # Draw arrow in center of zone
            cx = int(sum(p[0] for p in direction_zone) / len(direction_zone))
            cy = int(sum(p[1] for p in direction_zone) / len(direction_zone))
            arrow_len = 50
            dx, dy = expected_dir
            end_x = int(cx + dx * arrow_len)
            end_y = int(cy + dy * arrow_len)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 100, 0), 3, tipLength=0.3)
        cv2.putText(frame, 'DIRECTION ZONE', tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Zebra Crossing Zone
    zebra_zone = geom.get('zebra_crossing_zone')
    if zebra_zone and len(zebra_zone) >= 3:
        pts = np.array(zebra_zone, dtype=np.int32)
        # Draw with diagonal stripes pattern effect
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        # Fill with semi-transparent overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, 'ZEBRA CROSSING', tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


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


# ============================================================================
# Authentication Routes (RBAC System)
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login endpoint."""
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect('/')
        return render_template('login.html')
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    user = User.query.filter_by(username=username).first()
    if user and user.is_active and user.check_password(password):
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()
        audit_action('login', resource_type='user', resource_id=user.id)
        return redirect('/')
    
    audit_action('login_failed', resource_type='user', details={'username': username})
    return render_template('login.html', error='Invalid credentials')


@app.route('/logout')
def logout():
    """User logout endpoint."""
    if 'user_id' in session:
        audit_action('logout', resource_type='user', resource_id=session.get('user_id'))
    session.clear()
    return redirect('/login')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration (admin-only)."""
    # Check if admin is registering users
    if 'user_id' not in session:
        return redirect('/login')
    
    current_user = User.query.get(session['user_id'])
    if current_user.role != 'admin':
        return jsonify({'error': 'Only admins can register users'}), 403
    
    if request.method == 'GET':
        return render_template('register.html')
    
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    role = request.form.get('role', 'viewer')
    
    if not username or not email or not password:
        return render_template('register.html', error='All fields required')
    
    if User.query.filter_by(username=username).first():
        return render_template('register.html', error='Username already exists')
    
    if User.query.filter_by(email=email).first():
        return render_template('register.html', error='Email already exists')
    
    new_user = User(username=username, email=email, role=role, is_active=True)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    
    audit_action('user_created', resource_type='user', resource_id=new_user.id, 
                details={'username': username, 'role': role})
    
    return render_template('register.html', success=f'User {username} created')


@app.route('/api/users')
@require_role('admin')
def get_users():
    """List all users (admin only)."""
    users = User.query.all()
    audit_action('list_users', resource_type='user')
    return jsonify([u.to_dict() for u in users])


@app.route('/api/audit_logs')
@require_role('admin')
def get_audit_logs():
    """List audit logs (admin only)."""
    limit = request.args.get('limit', 100, type=int)
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return jsonify([log.to_dict() for log in logs])


@app.route('/api/user/profile')
@require_login
def get_user_profile():
    """Get current user profile."""
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict())


from datetime import datetime, timezone

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
    for v in violations:
        score, level = _priority_score(v)
        v.priority_score = score
        v.priority_level = level
    violations.sort(key=lambda x: (getattr(x, 'priority_score', 0), x.timestamp.timestamp() if x.timestamp else 0), reverse=True)
    return render_template('dashboard.html', violations=violations)


@app.route('/calibration')
def calibration_manager():
    return render_template('calibration_manager.html')

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
        score, level = _priority_score(v)
        result.append({
            'id': v.id,
            'timestamp': v.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'violation_type': v.violation_type,
            'vehicle_type': v.vehicle_type or 'Unknown',
            'confidence': f"{v.confidence:.2f}" if v.confidence else None,
            'track_id': v.track_id,
            'status': v.status,
            'image_path': v.image_path,
            'priority_score': score,
            'priority_level': level,
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


def _aggregate_runtime_metrics(run_id=None, camera_id=None):
    if get_runtime_metrics is None:
        return {}
    if run_id and camera_id:
        metrics = get_runtime_metrics(run_id=run_id, camera_id=camera_id)
        return metrics if isinstance(metrics, dict) else {}

    all_metrics = get_runtime_metrics()
    if not isinstance(all_metrics, dict):
        return {}

    agg = {
        'frames_processed': 0,
        'detections_seen': 0,
        'tracks_emitted': 0,
        'violations_emitted': 0,
        'frames_with_external_tracks': 0,
        'frames_with_fallback_tracks': 0,
    }
    for item in all_metrics.values():
        if not isinstance(item, dict):
            continue
        for k in agg:
            agg[k] += int(item.get(k, 0) or 0)
    return agg


def _build_signal_timing_suggestion(metrics, scope_key='global', force_profile_change=False):
    frames = max(int(metrics.get('frames_processed', 0) or 0), 1)
    avg_tracks_raw = float(metrics.get('tracks_emitted', 0) or 0) / frames
    violation_rate_raw = float(metrics.get('violations_emitted', 0) or 0) / frames
    fallback_ratio = float(metrics.get('frames_with_fallback_tracks', 0) or 0) / frames
    peak_cfg = _peak_hour_weight()

    avg_tracks_raw_weighted = avg_tracks_raw * float(peak_cfg['tracks_weight'])
    violation_rate_raw_weighted = violation_rate_raw * float(peak_cfg['violation_weight'])

    alpha = min(max(SIGNAL_POLICY_SMOOTHING_ALPHA, 0.05), 1.0)
    now_ts = time.time()

    with SIGNAL_POLICY_LOCK:
        prev = SIGNAL_POLICY_STATE.get(scope_key, {})
        prev_avg = float(prev.get('ewma_avg_tracks', avg_tracks_raw_weighted))
        prev_vrate = float(prev.get('ewma_violation_rate', violation_rate_raw_weighted))

        avg_tracks = (alpha * avg_tracks_raw_weighted) + ((1.0 - alpha) * prev_avg)
        violation_rate = (alpha * violation_rate_raw_weighted) + ((1.0 - alpha) * prev_vrate)

        # Choose candidate profile from smoothed metrics.
        if avg_tracks >= 6.0 or violation_rate >= 0.08:
            candidate_profile = 'high_congestion'
            candidate_durations = {'RED': 7, 'GREEN': 22, 'YELLOW': 3} if peak_cfg['is_peak_hour'] else {'RED': 8, 'GREEN': 20, 'YELLOW': 3}
        elif avg_tracks >= 3.0 or violation_rate >= 0.03:
            candidate_profile = 'medium_congestion'
            candidate_durations = {'RED': 9, 'GREEN': 17, 'YELLOW': 3} if peak_cfg['is_peak_hour'] else {'RED': 10, 'GREEN': 15, 'YELLOW': 3}
        else:
            candidate_profile = 'low_congestion'
            candidate_durations = {'RED': 12, 'GREEN': 12, 'YELLOW': 3} if peak_cfg['is_peak_hour'] else {'RED': 14, 'GREEN': 10, 'YELLOW': 3}

        last_profile = prev.get('profile')
        last_change_ts = float(prev.get('last_change_ts', 0.0))
        can_change = force_profile_change or (now_ts - last_change_ts >= SIGNAL_POLICY_HOLD_SECONDS)

        hold_active = False
        hold_remaining_seconds = 0
        if last_profile and candidate_profile != last_profile and not can_change:
            # Freeze profile until minimum hold time has passed.
            profile = last_profile
            durations = prev.get('durations', candidate_durations)
            hold_active = True
            hold_remaining_seconds = max(0, int(SIGNAL_POLICY_HOLD_SECONDS - (now_ts - last_change_ts)))
        else:
            profile = candidate_profile
            durations = candidate_durations
            if not last_profile or profile != last_profile:
                last_change_ts = now_ts

        SIGNAL_POLICY_STATE[scope_key] = {
            'profile': profile,
            'durations': durations,
            'ewma_avg_tracks': avg_tracks,
            'ewma_violation_rate': violation_rate,
            'last_change_ts': last_change_ts,
        }

    confidence = 'high'
    if fallback_ratio >= 0.7:
        confidence = 'medium'
    if frames < 40:
        confidence = 'low'

    rationale = [
        f"avg_tracks_per_frame_raw={avg_tracks_raw:.2f}",
        f"avg_tracks_per_frame_raw_weighted={avg_tracks_raw_weighted:.2f}",
        f"avg_tracks_per_frame_ewma={avg_tracks:.2f}",
        f"violation_rate_raw={violation_rate_raw:.3f}",
        f"violation_rate_raw_weighted={violation_rate_raw_weighted:.3f}",
        f"violation_rate_ewma={violation_rate:.3f}",
        f"fallback_ratio={fallback_ratio:.2f}",
        f"sample_frames={frames}",
    ]

    return {
        'profile': profile,
        'confidence': confidence,
        'recommended_mode': 'timer',
        'durations': durations,
        'rationale': rationale,
        'stability': {
            'smoothing_alpha': alpha,
            'hold_seconds': SIGNAL_POLICY_HOLD_SECONDS,
            'hold_active': hold_active,
            'hold_remaining_seconds': hold_remaining_seconds,
        },
        'peak_hour': peak_cfg,
        'derived': {
            'avg_tracks_ewma': avg_tracks,
            'violation_rate_ewma': violation_rate,
            'fallback_ratio': fallback_ratio,
        },
        'metrics_used': {
            'frames_processed': frames,
            'tracks_emitted': int(metrics.get('tracks_emitted', 0) or 0),
            'violations_emitted': int(metrics.get('violations_emitted', 0) or 0),
            'frames_with_fallback_tracks': int(metrics.get('frames_with_fallback_tracks', 0) or 0),
        },
    }


@app.route('/api/signal/suggestion')
def signal_suggestion():
    run_id = request.args.get('run_id')
    camera_id = request.args.get('camera_id')
    metrics = _aggregate_runtime_metrics(run_id=run_id, camera_id=camera_id)
    scope_key = f"{camera_id or 'global'}::{run_id or 'all'}"
    suggestion = _build_signal_timing_suggestion(metrics, scope_key=scope_key, force_profile_change=False)
    return jsonify({
        'scope': {
            'run_id': run_id,
            'camera_id': camera_id,
        },
        'current_signal': SIGNAL_MANAGER.get_status() if SIGNAL_MANAGER else None,
        'suggestion': suggestion,
    })


@app.route('/api/signal/apply_suggestion', methods=['POST'])
def apply_signal_suggestion():
    if not SIGNAL_MANAGER:
        return jsonify({'error': 'Signal Manager not available'}), 503

    data = request.json or {}
    run_id = data.get('run_id')
    camera_id = data.get('camera_id')
    requested_mode = (data.get('mode') or 'timer').strip().lower()
    force_change = bool(data.get('force'))

    metrics = _aggregate_runtime_metrics(run_id=run_id, camera_id=camera_id)
    scope_key = f"{camera_id or 'global'}::{run_id or 'all'}"
    suggestion = _build_signal_timing_suggestion(metrics, scope_key=scope_key, force_profile_change=force_change)
    durations = suggestion.get('durations', {})

    ok = SIGNAL_MANAGER.set_durations(
        red=durations.get('RED'),
        green=durations.get('GREEN'),
        yellow=durations.get('YELLOW'),
    )
    if not ok:
        return jsonify({'error': 'Could not apply suggested durations'}), 400

    if requested_mode in {'manual', 'timer'}:
        SIGNAL_MANAGER.set_mode(requested_mode)

    return jsonify({
        'ok': True,
        'applied': suggestion,
        'signal': SIGNAL_MANAGER.get_status(),
    })


@app.route('/api/signal/auto_policy', methods=['GET', 'POST'])
def auto_policy():
    _ensure_auto_policy_thread()

    if request.method == 'POST':
        data = request.json or {}
        camera_id = (data.get('camera_id') or 'default').strip()
        enabled = bool(data.get('enabled', True))
        interval_seconds = max(5, int(data.get('interval_seconds', AUTO_POLICY_DEFAULT_INTERVAL)))
        run_id = data.get('run_id')
        mode = (data.get('mode') or 'timer').strip().lower()
        force = bool(data.get('force', False))

        with AUTO_POLICY_LOCK:
            cur = AUTO_POLICY_STATE.get(camera_id, {})
            cur.update({
                'camera_id': camera_id,
                'enabled': enabled,
                'interval_seconds': interval_seconds,
                'run_id': run_id,
                'mode': mode if mode in {'manual', 'timer'} else 'timer',
                'force': force,
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            })
            AUTO_POLICY_STATE[camera_id] = cur

        return jsonify({'ok': True, 'policy': dict(AUTO_POLICY_STATE.get(camera_id, {}))})

    with AUTO_POLICY_LOCK:
        policies = {k: dict(v) for k, v in AUTO_POLICY_STATE.items()}
    return jsonify({
        'thread_alive': bool(AUTO_POLICY_THREAD and AUTO_POLICY_THREAD.is_alive()),
        'policies': policies,
    })


@app.route('/api/model/drift')
def model_drift_health():
    rows = Violation.query.order_by(Violation.timestamp.desc()).limit(120).all()
    rows = list(reversed(rows))
    conf_points = [float(v.confidence) for v in rows if v.confidence is not None]

    conf_trend = 'stable'
    confidence_delta = 0.0
    if len(conf_points) >= 8:
        mid = len(conf_points) // 2
        first_avg = sum(conf_points[:mid]) / max(1, len(conf_points[:mid]))
        second_avg = sum(conf_points[mid:]) / max(1, len(conf_points[mid:]))
        confidence_delta = second_avg - first_avg
        if confidence_delta <= -0.08:
            conf_trend = 'degrading'
        elif confidence_delta >= 0.05:
            conf_trend = 'improving'

    confidence_series = [
        {
            'timestamp': v.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': float(v.confidence),
        }
        for v in rows if v.confidence is not None
    ][-20:]

    runtime = get_runtime_metrics() if get_runtime_metrics is not None else {}
    fallback_by_run = []
    agg_frames = 0
    agg_fallback = 0
    if isinstance(runtime, dict):
        for k, m in runtime.items():
            if not isinstance(m, dict):
                continue
            frames = int(m.get('frames_processed', 0) or 0)
            fb = int(m.get('frames_with_fallback_tracks', 0) or 0)
            agg_frames += frames
            agg_fallback += fb
            ratio = (fb / frames) if frames > 0 else 0.0
            run_id, camera_id = (k.split(':', 1) + [''])[:2]
            fallback_by_run.append({
                'key': k,
                'run_id': run_id,
                'camera_id': camera_id,
                'fallback_ratio': round(ratio, 4),
                'frames_processed': frames,
            })

    fallback_by_run.sort(key=lambda x: x['fallback_ratio'], reverse=True)
    aggregate_ratio = (agg_fallback / agg_frames) if agg_frames > 0 else 0.0
    drift_risk = 'low'
    if conf_trend == 'degrading' or aggregate_ratio > 0.6:
        drift_risk = 'high'
    elif aggregate_ratio > 0.35:
        drift_risk = 'medium'

    return jsonify({
        'drift_risk': drift_risk,
        'confidence_trend': {
            'label': conf_trend,
            'delta': round(confidence_delta, 4),
            'sample_count': len(conf_points),
            'series': confidence_series,
        },
        'fallback_tracking': {
            'aggregate_ratio': round(aggregate_ratio, 4),
            'runs': fallback_by_run[:20],
        },
    })

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


@app.route('/api/metrics/summary')
def metrics_summary():
    from datetime import datetime, timedelta, timezone

    def _as_utc(dt):
        if not dt:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    since_24h = now - timedelta(hours=24)
    all_violations = Violation.query.order_by(Violation.timestamp.desc()).all()
    recent_24h = [v for v in all_violations if ((ts := _as_utc(v.timestamp)) and ts >= since_24h)]

    by_type = {}
    by_status = {}
    for v in all_violations:
        by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        by_status[v.status] = by_status.get(v.status, 0) + 1

    runtime = get_runtime_metrics() if get_runtime_metrics is not None else {}
    runtime_totals = {
        'active_runs': len(runtime),
        'frames_processed': sum((m.get('frames_processed', 0) for m in runtime.values())) if isinstance(runtime, dict) else 0,
        'tracks_emitted': sum((m.get('tracks_emitted', 0) for m in runtime.values())) if isinstance(runtime, dict) else 0,
        'violations_emitted': sum((m.get('violations_emitted', 0) for m in runtime.values())) if isinstance(runtime, dict) else 0,
    }

    return jsonify({
        'total_violations': len(all_violations),
        'violations_last_24h': len(recent_24h),
        'by_type': by_type,
        'by_status': by_status,
        'runtime': runtime_totals,
    })


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


@app.route('/api/camera/profile/<camera_id>', methods=['DELETE'])
def delete_camera_profile(camera_id):
    camera_id = (camera_id or '').strip()
    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400
    if camera_id == 'default':
        return jsonify({'error': 'default profile cannot be deleted'}), 400

    cfg_path = APP_ROOT / 'config' / 'cameras.json'
    try:
        if cfg_path.exists():
            camera_cfg = json.loads(cfg_path.read_text())
        else:
            camera_cfg = {}
    except Exception as e:
        return jsonify({'error': f'Failed to read camera config: {e}'}), 500

    if camera_id not in camera_cfg:
        return jsonify({'error': f'camera_id not found: {camera_id}'}), 404

    del camera_cfg[camera_id]
    try:
        cfg_path.write_text(json.dumps(camera_cfg, indent=2))
    except Exception as e:
        return jsonify({'error': f'Failed to write camera config: {e}'}), 500

    if reload_camera_config is not None:
        reload_camera_config()

    return jsonify({'ok': True, 'deleted': camera_id})


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


# ============================================================================
# Analytics, Monitoring, and Advanced Filtering Routes
# ============================================================================

from web_app.utils.analytics import (
    get_analytics_summary, get_camera_health, get_heatmap_data,
    build_filter_query, get_violation_trends, predict_resource_allocation,
    cleanup_old_violations, export_user_data
)

@app.route('/api/analytics/summary')
@require_login
def analytics_summary():
    """Get comprehensive analytics."""
    days = request.args.get('days', 7, type=int)
    summary = get_analytics_summary(days=days)
    audit_action('list_analytics', resource_type='analytics', details={'days': days})
    return jsonify(summary)


@app.route('/api/analytics/camera_health')
@require_login
def analytics_camera_health():
    """Get camera health metrics."""
    health = get_camera_health()
    audit_action('view_camera_health', resource_type='camera')
    return jsonify(health)


@app.route('/api/analytics/heatmap')
@require_login
def analytics_heatmap():
    """Get violation heatmap by time."""
    days = request.args.get('days', 7, type=int)
    heatmap = get_heatmap_data(limit_days=days)
    return jsonify(heatmap)


@app.route('/api/analytics/trends')
@require_login
def analytics_trends():
    """Get violation trends and predictions."""
    days = request.args.get('days', 30, type=int)
    trends = get_violation_trends(days=days)
    return jsonify(trends)


@app.route('/api/analytics/resource_allocation')
@require_permission('view_violations')
def analytics_resource_allocation():
    """Get enforcement resource allocation suggestions."""
    days = request.args.get('days', 7, type=int)
    recommendations = predict_resource_allocation(days=days)
    return jsonify(recommendations)


@app.route('/api/violations/search')
@require_login
def search_violations():
    """Advanced search with multiple filters."""
    filters = request.args.to_dict()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_desc = request.args.get('sort_desc', 'true').lower() == 'true'
    
    query = build_filter_query(filters)
    
    # Sorting
    if sort_by == 'timestamp':
        col = Violation.timestamp
    elif sort_by == 'priority_score':
        col = Violation.priority_score
    elif sort_by == 'confidence':
        col = Violation.confidence
    else:
        col = Violation.timestamp
    
    if sort_desc:
        query = query.order_by(col.desc())
    else:
        query = query.order_by(col.asc())
    
    # Pagination
    paginated = query.paginate(page=page, per_page=per_page)
    
    audit_action('search_violations', resource_type='violation', 
                details={'filter_count': len(filters), 'page': page})
    
    return jsonify({
        'violations': [v.to_dict() for v in paginated.items],
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page
    })


@app.route('/api/maintenance/cleanup', methods=['POST'])
@require_role('admin')
def maintenance_cleanup():
    """Clean up old violations (admin only)."""
    days = request.json.get('days', 90) if request.is_json else 90
    deleted = cleanup_old_violations(days=days)
    audit_action('cleanup_old_violations', resource_type='maintenance', 
                details={'days': days, 'deleted_count': deleted})
    return jsonify({'deleted_count': deleted})


@app.route('/api/gdpr/export_data')
@require_login
def gdpr_export_data():
    """Export user's personal data (GDPR compliance)."""
    user_id = session.get('user_id')
    data = export_user_data(user_id)
    audit_action('export_personal_data', resource_type='gdpr', resource_id=user_id)
    
    return jsonify(data)


# ============================================================================
# Active Learning & Video Streaming Routes
# ============================================================================

from web_app.utils.ml_advanced import ModelEnsemble, VideoStreamWriter, flag_uncertain_detections

@app.route('/api/active_learning/pending_labels')
@require_login
def get_pending_labels():
    """Get violations flagged for human labeling."""
    from web_app.models_ml import LabelingTask
    
    pending = LabelingTask.query.filter_by(status='pending').order_by(
        LabelingTask.created_at.desc()
    ).limit(50).all()
    
    return jsonify([task.to_dict() for task in pending])


@app.route('/api/active_learning/submit_label', methods=['POST'])
@require_permission('update_status')
def submit_label():
    """Submit human label for uncertain detection."""
    from web_app.models_ml import LabelingTask, TrainingBatch
    
    data = request.json
    task_id = data.get('task_id')
    human_label = data.get('label')
    
    task = LabelingTask.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    task.human_label = human_label
    task.is_correct = (human_label == task.predicted_label)
    task.labeled_at = datetime.now(timezone.utc)
    task.labeled_by = session.get('user_id')
    task.status = 'labeled'
    
    db.session.commit()
    audit_action('submit_label', resource_type='labeling_task', resource_id=task_id,
                details={'predicted': task.predicted_label, 'actual': human_label})
    
    return jsonify({'ok': True, 'task_id': task_id})


@app.route('/api/video_stream/validate', methods=['POST'])
@require_permission('manage_profiles')
def validate_stream():
    """Validate and get info about a video stream (RTSP/MJPEG/Webcam)."""
    data = request.json
    stream_url = data.get('stream_url')
    
    if not stream_url:
        return jsonify({'error': 'stream_url required'}), 400
    
    stream_type = VideoStreamWriter.validate_stream_url(stream_url)
    if not stream_type:
        return jsonify({'error': 'Invalid stream URL format'}), 400
    
    info = VideoStreamWriter.get_stream_info(stream_url)
    if not info:
        return jsonify({'error': 'Unable to connect to stream'}), 500
    
    audit_action('validate_stream', resource_type='video_stream', details={'url': stream_url})
    
    return jsonify({
        'valid': True,
        'stream_type': VideoStreamWriter.STREAM_TYPES.get(stream_type),
        'info': info
    })


@app.route('/api/video_stream/record_snippet', methods=['POST'])
@require_permission('manage_profiles')
def record_stream_snippet():
    """Record a short test snippet from a stream."""
    data = request.json
    stream_url = data.get('stream_url')
    duration = data.get('duration_seconds', 10)
    
    if not stream_url:
        return jsonify({'error': 'stream_url required'}), 400
    
    run_id = str(uuid.uuid4())[:8]
    snippet_path = RESULT_DIR / run_id / 'stream_snippet.mp4'
    snippet_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = VideoStreamWriter.record_stream_snippet(stream_url, str(snippet_path), duration)
    
    if success:
        audit_action('record_stream', resource_type='video_stream', details={'duration': duration})
        return jsonify({
            'ok': True,
            'snippet_path': str(snippet_path),
            'run_id': run_id,
            'size_mb': snippet_path.stat().st_size / 1024 / 1024
        })
    return jsonify({'error': 'Failed to record stream'}), 500


# ============================================================================
# Signal Controller Integration Routes
# ============================================================================

@app.route('/api/signal/controller/connect', methods=['POST'])
@require_role('admin')
def signal_controller_connect():
    """Connect to external traffic signal controller API."""
    data = request.json
    controller_url = data.get('controller_url')  # e.g., "http://192.168.1.100:8080"
    camera_id = data.get('camera_id', 'default')
    
    # Store controller config
    config = {
        'controller_url': controller_url,
        'active': True,
        'last_sync': datetime.now(timezone.utc).isoformat(),
        'version': '1.0'
    }
    
    cfg_path = APP_ROOT / 'config' / 'signal_controllers.json'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        existing = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    except:
        existing = {}
    
    existing[camera_id] = config
    cfg_path.write_text(json.dumps(existing, indent=2))
    
    audit_action('connect_signal_controller', resource_type='signal_controller',
                details={'controller_url': controller_url, 'camera_id': camera_id})
    
    return jsonify({'ok': True, 'controller_url': controller_url})


@app.route('/api/signal/controller/apply', methods=['POST'])
@require_permission('manage_profiles')
def signal_controller_apply():
    """Push suggested signal timing to physical traffic controller."""
    data = request.json
    camera_id = data.get('camera_id', 'default')
    signal_profile = data.get('profile')  # e.g., {'RED': 30, 'YELLOW': 5, 'GREEN': 35}
    
    # Load controller config
    cfg_path = APP_ROOT / 'config' / 'signal_controllers.json'
    if not cfg_path.exists():
        return jsonify({'error': 'No signal controller configured'}), 404
    
    controllers = json.loads(cfg_path.read_text())
    if camera_id not in controllers:
        return jsonify({'error': f'No controller for camera {camera_id}'}), 404
    
    controller_url = controllers[camera_id]['controller_url']
    
    # Send HTTP POST to actual signal controller
    try:
        import requests
        response = requests.post(
            f"{controller_url}/api/signal/apply",
            json={'profile': signal_profile, 'camera_id': camera_id},
            timeout=5
        )
        
        if response.status_code == 200:
            audit_action('apply_signal_profile', resource_type='signal_controller',
                        resource_id=None, details={'camera_id': camera_id, 'profile': signal_profile})
            return jsonify({'ok': True, 'result': response.json()})
        else:
            return jsonify({'error': response.text}), response.status_code
    except Exception as e:
        return jsonify({'error': f'Connection error: {str(e)}'}), 500


@app.route('/api/signal/controller/feedback', methods=['POST'])
@require_login
def signal_controller_feedback():
    """Receive feedback from signal controller about effectiveness."""
    data = request.json
    camera_id = data.get('camera_id', 'default')
    before_violation_count = data.get('before_violations', 0)
    after_violation_count = data.get('after_violations', 0)
    
    effectiveness = (before_violation_count - after_violation_count) / max(before_violation_count, 1) * 100
    
    audit_action('signal_feedback', resource_type='signal_controller', resource_id=None,
                details={'camera_id': camera_id, 'effectiveness': effectiveness})
    
    return jsonify({
        'ok': True,
        'effectiveness_percent': effectiveness,
        'improvement': 'POSITIVE' if effectiveness > 0 else 'NEUTRAL'
    })


# =============================================================================
# Live Stream Endpoint for Real-time Webcam Processing
# =============================================================================

from flask import Response

def generate_live_frames(camera_index=0, camera_id='default', enable_heatmap=True):
    """Generator function for live webcam streaming with inference."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    run_id = f"live_{uuid.uuid4().hex[:8]}"

    # Initialize heatmap and counter if enabled
    heatmap = None
    counter = None
    emergency_detector = None

    if enable_heatmap and DensityHeatmap:
        heatmap = DensityHeatmap()
    if VehicleCounter:
        counter = VehicleCounter()
    # Emergency vehicle detection intentionally disabled.

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = []

            # Run inference
            if model is not None:
                try:
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                         conf=0.25, verbose=False)

                    if results and results[0].boxes:
                        for box in results[0].boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            s = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            tid = int(box.id[0].cpu().numpy()) if box.id is not None else -1

                            # Get class name
                            cls_name = TARGET_NAMES[cls_id] if cls_id < len(TARGET_NAMES) else str(cls_id)

                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': s,
                                'class': cls_id,
                                'class_name': cls_name,
                                'track_id': tid
                            })

                            # Draw detection box
                            color = (0, 255, 0)  # Green
                            label = f"{cls_name} {s:.2f}"

                            # Check for emergency vehicle
                            if emergency_detector and cls_name.lower() in ['car', 'truck', 'bus']:
                                is_emergency, score, details = emergency_detector.check_vehicle(tid, [x1, y1, x2, y2], frame)
                                if is_emergency:
                                    color = (0, 0, 255)  # Red for emergency
                                    label = f"EMERGENCY {score:.2f}"
                                    cv2.putText(frame, '!!! EMERGENCY VEHICLE !!!',
                                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # Update counter
                            if counter and tid >= 0:
                                counter.count(tid, cls_id)

                except Exception as e:
                    print(f"Inference error: {e}")

            # Draw rule overlays
            draw_rule_overlay(frame, camera_id=camera_id)

            # Update and render heatmap
            if heatmap and detections:
                heatmap.update(detections, frame.shape)
                frame = heatmap.render_overlay(frame, alpha=0.25, show_grid=False)

            # Render vehicle counter
            if counter:
                counter.render_overlay(frame, position=(10, 30))

            # Add frame info
            cv2.putText(frame, f'Frame: {frame_idx} | Live Stream',
                       (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame_idx += 1

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()


@app.route('/live_stream')
def live_stream():
    """Live webcam stream with real-time inference overlay."""
    camera_index = request.args.get('camera', 0, type=int)
    camera_id = request.args.get('camera_id', 'default')
    enable_heatmap = request.args.get('heatmap', 'true').lower() == 'true'

    return Response(
        generate_live_frames(camera_index, camera_id, enable_heatmap),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/heatmap/<run_id>')
def get_heatmap_data(run_id):
    """Get current heatmap statistics for a run."""
    heatmap = HEATMAP_INSTANCES.get(run_id)
    if heatmap:
        return jsonify(heatmap.get_stats())
    return jsonify({'error': 'No heatmap data for this run'}), 404


@app.route('/api/vehicle_counts/<run_id>')
def get_vehicle_counts(run_id):
    """Get current vehicle counts for a run."""
    counter = VEHICLE_COUNTERS.get(run_id)
    if counter:
        return jsonify(counter.get_counts())
    return jsonify({'error': 'No counter data for this run'}), 404


@app.route('/api/alerts/config')
@require_login
def get_alerts_config():
    """Check Twilio/alerts configuration status."""
    if check_twilio_config:
        return jsonify(check_twilio_config())
    return jsonify({'configured': False, 'reason': 'alerts module not loaded'})


@app.route('/api/alerts/test', methods=['POST'])
@require_login
@require_role('admin')
def test_alerts():
    """Send test alert to configured channels."""
    if not emit_multi_channel_alert:
        return jsonify({'error': 'Alerts module not loaded'}), 500

    test_violation = {
        'event_type': 'test_alert',
        'track_id': 'TEST-001',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'meta': {'message': 'This is a test alert from Smart Traffic System'}
    }

    channels = request.json.get('channels', ['sms', 'whatsapp']) if request.json else ['sms', 'whatsapp']
    results = emit_multi_channel_alert(test_violation, channels=channels)

    return jsonify({'ok': True, 'results': results})


if __name__ == '__main__':
    # Use 0.0.0.0 for Docker visibility, but 5050 to avoid MacOS AirPlay conflict
    app.run(host='0.0.0.0', port=5050, debug=True)
