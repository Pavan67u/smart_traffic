"""
Simple rules PoC: detect stop-line crossing without stopping.
This module consumes track updates (track_id, bbox, centroid, timestamp, frame)
and a stop-line definition and emits JSON events appended to an events file.
"""
import json
import uuid
import math
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timezone

EVENTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'events'
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

def centroid_from_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def point_in_polygon(point, polygon):
    # point: (x, y), polygon: list of points [(x,y), ...]
    # cv2.pointPolygonTest requires contour as float32 array
    contour = np.array(polygon, dtype=np.float32)
    # returns +ve (inside), -ve (outside), 0 (on edge)
    return cv2.pointPolygonTest(contour, point, False) >= 0

class StopLineRule:
    def __init__(self, stop_zone, required_stop_seconds=1.0, min_violation_speed=5.0):
        """
        stop_zone: list of (x,y) tuples defining the polygon of the intersection/stop area
        min_violation_speed: pixels per second (or frame) threshold to ignore stationary jitter
        """
        self.stop_zone = stop_zone 
        self.required_stop_seconds = required_stop_seconds
        self.min_violation_speed = min_violation_speed
        self.track_states = {}  # track_id -> {last_centroid, last_time, stopped_since, creation_time}

    def process_track(self, track_id, bbox, timestamp, frame_id, fps=30.0, signal_state='RED', class_id=None, score=None, **kwargs):
        # 1. Signal Check: If GREEN or YELLOW, no red light violation possible
        if signal_state in ['GREEN', 'YELLOW']:
            # Optional: clear state if needed, or just ignore
            return None

        c = centroid_from_bbox(bbox)
        # For stop logic, user bottom-center might be better, but centroid is standard for tracking.
        # Let's use bottom-center for zone check as recommended by expert review.
        x1, y1, x2, y2 = bbox
        bottom_center = ((x1+x2)/2.0, y2)
        
        now = timestamp
        state = self.track_states.get(track_id, {'last_centroid': c, 'last_time': now, 'stopped_since': None, 'creation_time': now})
        prev_c = state['last_centroid']
        prev_t = state['last_time']
        
        # Calculate Speed (pixels/second)
        dt = (now - prev_t).total_seconds()
        speed = 0.0
        if dt > 0:
            dist = math.hypot(c[0]-prev_c[0], c[1]-prev_c[1])
            speed = dist / dt
        
        # 2. Zone Check: Is vehicle INSIDE the stop zone?
        in_zone = point_in_polygon(bottom_center, self.stop_zone)
        
        violation = False
        
        # Logic:
        # If Signal is RED
        # AND Vehicle is IN Zone
        # AND Vehicle is MOVING (speed > threshold)
        # AND Vehicle did NOT stop previously (optional heuristic)
        # -> VIOLATION
        
        if in_zone and dt > 0: # Only check if we have history
            if speed > self.min_violation_speed:
                # Check history: did it stop before?
                stopped_dur = 0
                if state.get('stopped_since'):
                     stopped_dur = (now - state['stopped_since']).total_seconds()
                     
                if stopped_dur < self.required_stop_seconds:
                     violation = True
        
        # Update Stopped State (independent of zone, track globally)
        # Only update if we have a valid time delta to avoid marking a new track
        # as "stopped" on its very first observation.
        if dt > 0:
            if speed < self.min_violation_speed: # Effective stop threshold
                if state.get('stopped_since') is None:
                    state['stopped_since'] = now
            else:
                 # Reset stop state if moving significantly
                 state['stopped_since'] = None

        state['last_centroid'] = c
        state['last_time'] = now
        self.track_states[track_id] = state

        if violation:
            # Ghost suppression
            age = (now - state['creation_time']).total_seconds()
            if age < 0.5: # Allow faster trigger than 1.0s if speed matches
                return None

            ev = {
                'event_id': str(uuid.uuid4()),
                'event_type': 'red_light_violation',
                'track_id': track_id,
                'class_id': class_id,
                'score': score,
                'frame_id': frame_id,
                'timestamp': now.isoformat(),
                'bbox': bbox,
                'meta': {
                    'speed_px_s': speed,
                    'signal_state': signal_state,
                    'stop_zone': self.stop_zone
                }
            }
            # Add cooldown to prevent spamming same car every frame
            if state.get('last_violation_time') and (now - state['last_violation_time']).total_seconds() < 2.0:
                return None
            
            state['last_violation_time'] = now
            self._emit_event(ev)
            return ev
        return None

    def _emit_event(self, ev):
        runid = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = EVENTS_DIR / f'{runid}.jsonl'
        with open(out, 'a') as f:
            f.write(json.dumps(ev) + '\n')
