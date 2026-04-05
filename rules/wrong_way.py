"""
Wrong-Way Driving Detection Rule.
Detects vehicles traveling against the expected direction of traffic flow.
"""
import json
import uuid
import math
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

EVENTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'events'
EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using cv2."""
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


class WrongWayRule:
    """
    Detects vehicles moving in the wrong direction within a defined zone.

    Uses track history to calculate movement direction and compares against
    the expected traffic flow direction.
    """

    def __init__(self, direction_zone, expected_direction, angle_tolerance=60.0, min_history_points=5, min_displacement=30.0):
        """
        Args:
            direction_zone: list of (x,y) tuples defining the detection area polygon
            expected_direction: (dx, dy) tuple - normalized vector for expected traffic flow
                               Example: (0, 1) = traffic flows downward (south)
                                       (1, 0) = traffic flows rightward (east)
                                       (0, -1) = traffic flows upward (north)
            angle_tolerance: degrees allowed deviation from expected direction (default 60)
            min_history_points: minimum track points needed to calculate direction
            min_displacement: minimum pixel displacement to calculate direction (avoid noise)
        """
        self.direction_zone = direction_zone
        self.expected_direction = self._normalize(expected_direction)
        self.angle_tolerance = angle_tolerance
        self.min_history_points = min_history_points
        self.min_displacement = min_displacement
        self.track_history = {}  # track_id -> deque of (centroid, timestamp)
        self.track_states = {}   # track_id -> {last_violation_time, creation_time}
        self.max_history = 20    # Keep last N positions

    def _normalize(self, vec):
        """Normalize a vector to unit length."""
        dx, dy = vec
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return (0, 1)  # Default to downward
        return (dx / mag, dy / mag)

    def _centroid_from_bbox(self, bbox):
        """Get centroid from bbox."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _calculate_direction_vector(self, track_id):
        """
        Calculate movement direction from track history.
        Returns normalized direction vector or None if insufficient data.
        """
        history = self.track_history.get(track_id)
        if not history or len(history) < self.min_history_points:
            return None

        # Use first and last points for direction
        start_centroid, _ = history[0]
        end_centroid, _ = history[-1]

        dx = end_centroid[0] - start_centroid[0]
        dy = end_centroid[1] - start_centroid[1]

        # Check minimum displacement
        displacement = math.hypot(dx, dy)
        if displacement < self.min_displacement:
            return None

        return self._normalize((dx, dy))

    def _angle_between_vectors(self, v1, v2):
        """Calculate angle in degrees between two vectors."""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        # Clamp to [-1, 1] for numerical stability
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _is_wrong_way(self, movement_direction):
        """
        Check if movement direction is opposite to expected direction.
        Returns True if vehicle is going wrong way (angle > 180 - tolerance).
        """
        angle = self._angle_between_vectors(movement_direction, self.expected_direction)
        # Wrong way if angle is close to 180 degrees (opposite direction)
        return angle > (180.0 - self.angle_tolerance)

    def process_track(self, track_id, bbox, timestamp, frame_id, fps=30.0, class_id=None, score=None, **kwargs):
        """
        Process a track update and check for wrong-way violation.

        Returns:
            dict: Violation event if detected, None otherwise
        """
        centroid = self._centroid_from_bbox(bbox)
        now = timestamp

        # Initialize state for new tracks
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                'creation_time': now,
                'last_violation_time': None,
                'violation_reported': False
            }

        # Initialize history for new tracks
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.max_history)

        # Add current position to history
        self.track_history[track_id].append((centroid, now))

        state = self.track_states[track_id]

        # Check if vehicle is in the direction zone
        # Use bottom-center for zone check (more accurate for ground plane)
        x1, y1, x2, y2 = bbox
        bottom_center = ((x1 + x2) / 2.0, y2)

        in_zone = point_in_polygon(bottom_center, self.direction_zone)

        if not in_zone:
            return None

        # Calculate movement direction
        movement_dir = self._calculate_direction_vector(track_id)
        if movement_dir is None:
            return None

        # Check if going wrong way
        if not self._is_wrong_way(movement_dir):
            return None

        # Ghost suppression: ignore very new tracks
        age = (now - state['creation_time']).total_seconds()
        if age < 1.0:
            return None

        # Cooldown: prevent spamming violations for same vehicle
        if state['last_violation_time']:
            cooldown = (now - state['last_violation_time']).total_seconds()
            if cooldown < 5.0:  # 5 second cooldown
                return None

        # Create violation event
        angle = self._angle_between_vectors(movement_dir, self.expected_direction)

        ev = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'wrong_way_violation',
            'track_id': track_id,
            'class_id': class_id,
            'score': score,
            'frame_id': frame_id,
            'timestamp': now.isoformat(),
            'bbox': bbox,
            'meta': {
                'movement_direction': movement_dir,
                'expected_direction': self.expected_direction,
                'angle_deviation': angle,
                'direction_zone': self.direction_zone
            }
        }

        state['last_violation_time'] = now
        state['violation_reported'] = True
        self._emit_event(ev)

        return ev

    def _emit_event(self, ev):
        """Write event to JSONL file."""
        runid = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = EVENTS_DIR / f'{runid}.jsonl'
        with open(out, 'a') as f:
            f.write(json.dumps(ev) + '\n')

    def cleanup_old_tracks(self, max_age_seconds=60.0, current_time=None):
        """Remove old tracks from memory to prevent memory leaks."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        to_remove = []
        for track_id, history in self.track_history.items():
            if history:
                _, last_time = history[-1]
                age = (current_time - last_time).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.track_history[track_id]
            if track_id in self.track_states:
                del self.track_states[track_id]
