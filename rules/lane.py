import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Keep rule-level events under project-root/results/events
EVENTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'events'
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

class LaneViolationRule:
    """
    Detects when a vehicle crosses a solid lane divider.
    """
    def __init__(self, lane_line):
        # lane_line: ((x1,y1),(x2,y2)) defining the solid line
        self.lane_line = lane_line
        self.track_states = {} # track_id -> last_centroid

    def _crosses_line(self, prev_c, cur_c):
        (x1,y1),(x2,y2) = self.lane_line
        def side(p):
            return (x2-x1)*(p[1]-y1) - (y2-y1)*(p[0]-x1)
        # Check if they are on opposite sides
        return side(prev_c) * side(cur_c) < 0

    def process_track(self, track_id, bbox, timestamp, frame_id, speed_px_s=None, class_id=None, score=None, **kwargs):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        c = (cx, cy)

        # State: {'last_centroid': (x,y), 'creation_time': timestamp}
        state = self.track_states.get(track_id)
        if state is None:
            state = {'last_centroid': c, 'creation_time': timestamp}
            self.track_states[track_id] = state
            return None

        prev_c = state['last_centroid']
        violation = None

        if self._crosses_line(prev_c, c):
            # False positive suppression
            age = (timestamp - state['creation_time']).total_seconds()
            if age >= 1.0:
                if state.get('last_violation_time') and (timestamp - state['last_violation_time']).total_seconds() < 2.0:
                    state['last_centroid'] = c
                    self.track_states[track_id] = state
                    return None

                # Crossed the solid line
                violation = {
                    'event_id': str(uuid.uuid4()),
                    'event_type': 'lane_violation',
                    'track_id': track_id,
                    'class_id': class_id,
                    'score': score,
                    'frame_id': frame_id,
                    'timestamp': timestamp.isoformat(),
                    'bbox': bbox,
                    'meta': {'lane_line': self.lane_line}
                }
                state['last_violation_time'] = timestamp
                self._emit_event(violation)

        state['last_centroid'] = c
        self.track_states[track_id] = state
        return violation

    def _emit_event(self, ev):
        runid = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = EVENTS_DIR / f'{runid}_lane.jsonl'
        with open(out, 'a') as f:
            f.write(json.dumps(ev) + '\n')
