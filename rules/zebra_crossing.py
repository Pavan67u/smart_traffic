"""
Zebra Crossing Violation Rule.
Detects vehicles stopping or dwelling on pedestrian crossings.
"""
import json
import uuid
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timezone

EVENTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'events'
EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using cv2."""
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


class ZebraCrossingRule:
    """
    Detects vehicles stopping or dwelling on zebra crossings.

    Triggers violation when a vehicle remains stationary on the crossing
    zone for longer than the threshold duration.
    """

    def __init__(self, crossing_zone, dwell_threshold_seconds=2.0, min_speed_threshold=3.0):
        """
        Args:
            crossing_zone: List of (x,y) tuples defining the zebra crossing polygon
            dwell_threshold_seconds: Time vehicle must be stopped to trigger violation
            min_speed_threshold: Speed (px/s) below which vehicle is considered stopped
        """
        self.crossing_zone = crossing_zone
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.min_speed_threshold = min_speed_threshold
        self.track_states = {}  # track_id -> state dict

    def _centroid_from_bbox(self, bbox):
        """Get centroid from bbox."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _calculate_speed(self, prev_centroid, curr_centroid, dt):
        """Calculate speed in pixels per second."""
        if dt <= 0:
            return 0
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance / dt

    def process_track(self, track_id, bbox, timestamp, frame_id, class_id=None, score=None, **kwargs):
        """
        Process a track update and check for zebra crossing violation.

        Returns:
            dict: Violation event if detected, None otherwise
        """
        centroid = self._centroid_from_bbox(bbox)
        now = timestamp

        # Use bottom-center for ground plane detection
        x1, y1, x2, y2 = bbox
        bottom_center = ((x1 + x2) / 2.0, y2)

        # Check if in crossing zone
        in_zone = point_in_polygon(bottom_center, self.crossing_zone)

        # Initialize state for new tracks
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                'creation_time': now,
                'last_centroid': centroid,
                'last_time': now,
                'zone_entry_time': None,
                'stopped_since': None,
                'violation_reported': False
            }
            return None

        state = self.track_states[track_id]

        # Calculate speed
        dt = (now - state['last_time']).total_seconds()
        speed = self._calculate_speed(state['last_centroid'], centroid, dt)

        violation = None

        if in_zone:
            # Track zone entry time
            if state['zone_entry_time'] is None:
                state['zone_entry_time'] = now

            # Check if stopped
            if speed < self.min_speed_threshold:
                if state['stopped_since'] is None:
                    state['stopped_since'] = now
                else:
                    # Calculate dwell time
                    dwell_time = (now - state['stopped_since']).total_seconds()

                    # Check for violation
                    if dwell_time >= self.dwell_threshold_seconds and not state['violation_reported']:
                        violation = {
                            'event_id': str(uuid.uuid4()),
                            'event_type': 'zebra_crossing_violation',
                            'track_id': track_id,
                            'class_id': class_id,
                            'score': score,
                            'frame_id': frame_id,
                            'timestamp': now.isoformat(),
                            'bbox': bbox,
                            'meta': {
                                'dwell_time_seconds': dwell_time,
                                'threshold_seconds': self.dwell_threshold_seconds,
                                'crossing_zone': self.crossing_zone
                            }
                        }
                        state['violation_reported'] = True
                        self._emit_event(violation)
            else:
                # Vehicle moving, reset stopped state
                state['stopped_since'] = None
        else:
            # Not in zone, reset zone-related states
            state['zone_entry_time'] = None
            state['stopped_since'] = None
            state['violation_reported'] = False

        # Update state
        state['last_centroid'] = centroid
        state['last_time'] = now

        return violation

    def _emit_event(self, ev):
        """Write event to JSONL file."""
        runid = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = EVENTS_DIR / f'{runid}.jsonl'
        with open(out, 'a') as f:
            f.write(json.dumps(ev) + '\n')

    def cleanup_old_tracks(self, max_age_seconds=60.0, current_time=None):
        """Remove old tracks from memory."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        to_remove = []
        for track_id, state in self.track_states.items():
            age = (current_time - state['last_time']).total_seconds()
            if age > max_age_seconds:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.track_states[track_id]
