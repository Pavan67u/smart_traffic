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

from pathlib import Path
import json
import os

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
try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            CAMERA_CONFIG = json.load(f)
    else:
        CAMERA_CONFIG = {
            "default": {
                "stop_line_y": 350,
                "lane_line": [[960, 0], [960, 1080]]
            }
        }
except Exception as e:
    print(f"Error loading camera config: {e}")
    CAMERA_CONFIG = {"default": {"stop_line_y": 350}}

# Create a RedLightRuleEngine per camera / run
RULE_ENGINES = {}


# Import Signal Manager
try:
    from web_app.utils.signal_manager import SIGNAL_MANAGER
except ImportError:
    SIGNAL_MANAGER = None

def get_rule_engine(camera_id="default"):
    # adapt to StopLineRule implementation in rules/red_light.py
    if camera_id not in RULE_ENGINES:
        cfg = CAMERA_CONFIG.get(camera_id, CAMERA_CONFIG["default"])
        engines = []
        
        # 1. Red Light / Stop Line
        y = cfg.get("stop_line_y", 350)
        # Convert legacy line Y to a Polygon Zone (Area below the line)
        # Assuming 1920x1080 resolution for default config
        stop_zone = [(0, y), (1920, y), (1920, 1080), (0, 1080)]
        
        # Check if explicit zone defined
        if "stop_zone" in cfg:
            stop_zone = cfg["stop_zone"]

        try:
            from rules.red_light import StopLineRule
            engines.append(StopLineRule(stop_zone, required_stop_seconds=1.0, min_violation_speed=5.0))
        except Exception as e:
            print(f"Error loading StopLineRule: {e}")
            
        # 2. Lane Violation
        lane_line = cfg.get("lane_line") # ((x1,y1), (x2,y2))
        if lane_line and LaneViolationRule:
            engines.append(LaneViolationRule(lane_line))
            
        RULE_ENGINES[camera_id] = engines
    return RULE_ENGINES[camera_id]

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
        # Convert directly to track structure
        tracks = []
        for d in detections:
            if d.get('track_id') is not None:
                tracks.append({
                    'track_id': d['track_id'],
                    'bbox': d['bbox'],
                    'cls': d['class'],
                    'score': d['score']
                })
    else:
        tracks = TRACKER.update(detections, key=tracker_key, frame_id=frame_idx)

    # check rules (list of engines)
    engines = get_rule_engine(camera_id)
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
    return tracks, violations
