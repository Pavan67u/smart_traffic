"""
High-level manager that connects inference -> tracker -> rules.
Intended to be called from your inference endpoint after detections are produced.
"""

# rules.red_light is imported when needed (StopLineRule fallback or RedLightRuleEngine if present)
# Try to import bytetrack wrapper factory or fallback tracker
try:
    from tracking.bytetrack_wrapper import create_tracker
except Exception:
    create_tracker = None
try:
    from workers.iou_tracker import create_fallback_tracker
except Exception:
    create_fallback_tracker = None

# Import Lane Rule
try:
    from rules.lane import LaneViolationRule
except ImportError:
    LaneViolationRule = None

# Disabled unstable rules (wrong-way, triple-riding, zebra-crossing, emergency-vehicle).
WrongWayRule = None
TripleRidingRule = None
ZebraCrossingRule = None
EmergencyVehicleDetector = None

from pathlib import Path
import json
import os


def _scale_value(v, src_max, dst_max):
    # Support both absolute pixels and normalized [0,1] values.
    if isinstance(v, (int, float)):
        if 0.0 <= float(v) <= 1.0:
            return int(round(float(v) * dst_max))
        return int(round(float(v) * dst_max / src_max))
    return 0


def _scale_point(pt, ref_w, ref_h, out_w, out_h):
    x, y = pt
    return (_scale_value(x, ref_w, out_w), _scale_value(y, ref_h, out_h))

class _LocalTrackerManager:
    def __init__(self, iou_thr=0.35):
        self.iou_thr = iou_thr
        self.trackers = {}

    def get_tracker(self, key="default"):
        if key in self.trackers:
            return self.trackers[key]
        # prefer bytetrack if available
        if create_tracker is not None:
            try:
                t = create_tracker()
                self.trackers[key] = t
                return t
            except Exception:
                pass
        # fallback
        if create_fallback_tracker is not None:
            t = create_fallback_tracker()
            self.trackers[key] = t
            return t
        raise RuntimeError('No tracker available (bytetrack missing and fallback not present)')

    def update(self, detections, key="default", frame_id=None):
        t = self.get_tracker(key)
        # normalize detections for different tracker implementations
        dets = detections
        if detections and isinstance(detections, list) and isinstance(detections[0], dict):
            dets = []
            for d in detections:
                bbox = d.get('bbox')
                score = d.get('score', 1.0)
                cls = d.get('class', d.get('cls'))
                if bbox is None:
                    continue
                dets.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls])

        # adapt to both tracker.update signatures
        try:
            return t.update(dets, frame_id)
        except TypeError:
            # try without frame_id kw
            return t.update(dets)


TRACKER = _LocalTrackerManager(iou_thr=0.35)  # adjust IoU thr if needed

# Load camera config
CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'cameras.json'


def _default_camera_config():
    return {
        "default": {
            "stop_line_y": 350,
            "lane_line": [[960, 0], [960, 1080]],
            "reference_resolution": [1920, 1080],
        }
    }


def _load_camera_config_file():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return _default_camera_config()


def reload_camera_config():
    """Reload camera geometry config and clear cached rule engines/geometry."""
    global CAMERA_CONFIG
    try:
        latest = _load_camera_config_file()
    except Exception as e:
        print(f"Error loading camera config: {e}")
        latest = _default_camera_config()

    # Mutate in place so imported references (e.g., in app.py) stay valid.
    CAMERA_CONFIG.clear()
    CAMERA_CONFIG.update(latest)

    if 'RULE_ENGINES' in globals() and isinstance(RULE_ENGINES, dict):
        RULE_ENGINES.clear()
    if 'RULE_GEOMETRY' in globals() and isinstance(RULE_GEOMETRY, dict):
        RULE_GEOMETRY.clear()
    return CAMERA_CONFIG


CAMERA_CONFIG = {}
reload_camera_config()

# Create a RedLightRuleEngine per camera / run
RULE_ENGINES = {}
RULE_GEOMETRY = {}
RUNTIME_METRICS = {}


# Import Signal Manager
try:
    from web_app.utils.signal_manager import SIGNAL_MANAGER
except ImportError:
    SIGNAL_MANAGER = None

def get_rule_engine(camera_id="default", frame_shape=None):
    # adapt to StopLineRule implementation in rules/red_light.py
    cache_key = camera_id
    frame_w, frame_h = 1920, 1080
    if frame_shape is not None:
        try:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
            cache_key = f"{camera_id}:{frame_w}x{frame_h}"
        except Exception:
            cache_key = camera_id

    if cache_key not in RULE_ENGINES:
        cfg = CAMERA_CONFIG.get(camera_id, CAMERA_CONFIG.get("default", {}))
        engines = []

        ref = cfg.get("reference_resolution", [1920, 1080])
        try:
            ref_w, ref_h = int(ref[0]), int(ref[1])
        except Exception:
            ref_w, ref_h = 1920, 1080
        
        # 1. Red Light / Stop Line
        y = cfg.get("stop_line_y", 350)
        y_scaled = _scale_value(y, ref_h, frame_h)
        # Convert legacy line Y to a Polygon Zone (Area below the line)
        stop_zone = [(0, y_scaled), (frame_w, y_scaled), (frame_w, frame_h), (0, frame_h)]
        
        # Check if explicit zone defined
        if "stop_zone" in cfg:
            stop_zone = [_scale_point(tuple(p), ref_w, ref_h, frame_w, frame_h) for p in cfg["stop_zone"]]

        try:
            from rules.red_light import StopLineRule
            engines.append(StopLineRule(stop_zone, required_stop_seconds=1.0, min_violation_speed=5.0))
        except Exception as e:
            print(f"Error loading StopLineRule: {e}")
            
        # 2. Lane Violation
        lane_line_scaled = None
        lane_line = cfg.get("lane_line") # ((x1,y1), (x2,y2))
        if lane_line and LaneViolationRule:
            if isinstance(lane_line, list) and len(lane_line) == 2:
                p1 = _scale_point(tuple(lane_line[0]), ref_w, ref_h, frame_w, frame_h)
                p2 = _scale_point(tuple(lane_line[1]), ref_w, ref_h, frame_w, frame_h)
                lane_line_scaled = [list(p1), list(p2)]
                engines.append(LaneViolationRule((p1, p2)))

        # 3/4. Wrong-way and zebra-crossing checks are intentionally disabled.
        direction_zone_scaled = None
        zebra_zone_scaled = None
        expected_direction = cfg.get("expected_direction", [0, 1])

        RULE_GEOMETRY[cache_key] = {
            "camera_id": camera_id,
            "frame_size": [frame_w, frame_h],
            "stop_zone": [list(p) for p in stop_zone],
            "lane_line": lane_line_scaled,
            "direction_zone": direction_zone_scaled,
            "expected_direction": list(expected_direction) if expected_direction else None,
            "zebra_crossing_zone": zebra_zone_scaled,
        }
            
        RULE_ENGINES[cache_key] = engines
    return RULE_ENGINES[cache_key]


def get_rule_geometry(camera_id="default", frame_shape=None):
    cache_key = camera_id
    if frame_shape is not None:
        try:
            h, w = int(frame_shape[0]), int(frame_shape[1])
            cache_key = f"{camera_id}:{w}x{h}"
        except Exception:
            cache_key = camera_id

    # Ensure geometry exists by building/retrieving the engine for this shape.
    get_rule_engine(camera_id=camera_id, frame_shape=frame_shape)
    return RULE_GEOMETRY.get(cache_key, {})


def _metrics_key(run_id, camera_id):
    return f"{run_id}:{camera_id}"


def get_runtime_metrics(run_id=None, camera_id=None):
    if run_id is not None and camera_id is not None:
        return dict(RUNTIME_METRICS.get(_metrics_key(run_id, camera_id), {}))
    return {k: dict(v) for k, v in RUNTIME_METRICS.items()}

def update_and_check(run_id, camera_id, frame_idx, frame_img, detections, fps=30.0):
    """
    detections: list of dicts {"bbox":[x1,y1,x2,y2], "score":float, "class":int}
    frame_img: BGR numpy array (cv2)
    fps: frames per second of input video
    returns:
      tracks: list of track dicts
      violations: list of violation meta dicts (already saved)
    """
    # update tracker
    tracker_key = f"{run_id}_{camera_id}"
    
    # Check if we have external track IDs (ByteTrack from app.py)
    has_external_tracks = detections and any(d.get('track_id') is not None for d in detections)
    
    if has_external_tracks:
        # Convert available external IDs directly to track structure.
        # For detections where external IDs are missing, use a fallback tracker
        # so those objects are not silently dropped from downstream rules.
        tracks = []
        missing_id_dets = []
        for d in detections:
            if d.get('track_id') is not None:
                tracks.append({
                    'track_id': d['track_id'],
                    'bbox': d['bbox'],
                    'cls': d['class'],
                    'score': d['score']
                })
            else:
                missing_id_dets.append(d)

        if missing_id_dets:
            # Keep a stable fallback namespace per run/camera to retain IDs across frames.
            fallback_key = f"{tracker_key}_missing"
            fallback_tracks = TRACKER.update(missing_id_dets, key=fallback_key, frame_id=frame_idx)
            # Offset fallback IDs to avoid colliding with external tracker IDs.
            fallback_offset = 1000000
            for tr in fallback_tracks:
                tracks.append({
                    'track_id': int(tr.get('track_id', 0)) + fallback_offset,
                    'bbox': tr.get('bbox'),
                    'cls': tr.get('cls'),
                    'score': tr.get('score')
                })
    else:
        tracks = TRACKER.update(detections, key=tracker_key, frame_id=frame_idx)

    mkey = _metrics_key(run_id, camera_id)
    m = RUNTIME_METRICS.get(mkey, {
        'run_id': run_id,
        'camera_id': camera_id,
        'frames_processed': 0,
        'detections_seen': 0,
        'tracks_emitted': 0,
        'violations_emitted': 0,
        'frames_with_external_tracks': 0,
        'frames_with_fallback_tracks': 0,
    })
    m['frames_processed'] += 1
    m['detections_seen'] += len(detections or [])
    m['tracks_emitted'] += len(tracks or [])
    if has_external_tracks:
        m['frames_with_external_tracks'] += 1
    if (not has_external_tracks) or any((d.get('track_id') is None) for d in (detections or [])):
        m['frames_with_fallback_tracks'] += 1

    # check rules (list of engines)
    frame_shape = getattr(frame_img, 'shape', None)
    engines = get_rule_engine(camera_id, frame_shape=frame_shape)
    if not isinstance(engines, list):
        engines = [engines]
        
    violations = []
    from datetime import datetime, timezone
    
    # Get Signal State
    signal_state = 'RED'
    if SIGNAL_MANAGER:
        signal_state = SIGNAL_MANAGER.get_status()['state']

    for engine in engines:
        # adapt to either engine API: prefer check_tracks_for_violations if present
        if hasattr(engine, 'check_tracks_for_violations'):
            violations.extend(engine.check_tracks_for_violations(run_id, frame_idx, frame_img, tracks))
        else:
            # fallback: call process_track for each track
            for tr in tracks:
                tid = tr.get('track_id')
                bbox = tr.get('bbox')
                now = datetime.now(timezone.utc)
                ev = None
                try:
                    # Pass signal_state and fps
                    ev = engine.process_track(tid, bbox, now, frame_idx, 
                                            fps=fps, 
                                            signal_state=signal_state,
                                            class_id=tr.get('cls'), 
                                            score=tr.get('score'))
                except Exception:
                    ev = None
                if ev:
                    violations.append(ev)

    m['violations_emitted'] += len(violations)
    RUNTIME_METRICS[mkey] = m

    # Triple-riding check intentionally disabled.

    return tracks, violations
