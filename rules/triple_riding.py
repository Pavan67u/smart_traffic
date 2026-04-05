"""
Triple Riding Detection Rule.
Detects when more than 2 people are on a motorcycle (common violation in India).
"""
import json
import uuid
import math
from pathlib import Path
from datetime import datetime, timezone

EVENTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'events'
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# Class IDs (matching training/classes.txt)
CLASS_MOTORCYCLE = 3  # motorcycle
CLASS_PEDESTRIAN = 4  # pedestrian (person)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def bbox_contains_point(bbox, point):
    """Check if bbox contains a point."""
    x1, y1, x2, y2 = bbox
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def expand_bbox(bbox, factor):
    """Expand bbox by a factor while keeping center fixed."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    nw = w * factor
    nh = h * factor

    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


class TripleRidingRule:
    """
    Detects triple riding (3+ persons on a motorcycle) violation.

    This rule operates on frame-level rather than track-level, analyzing
    the spatial relationship between motorcycles and persons.
    """

    def __init__(self, min_persons=3, iou_threshold=0.15, bbox_expansion=1.3, cooldown_seconds=5.0):
        """
        Args:
            min_persons: Minimum persons to trigger violation (3 = triple riding)
            iou_threshold: Minimum IoU between person and expanded motorcycle bbox
            bbox_expansion: Factor to expand motorcycle bbox for person detection
            cooldown_seconds: Cooldown between violations for same motorcycle
        """
        self.min_persons = min_persons
        self.iou_threshold = iou_threshold
        self.bbox_expansion = bbox_expansion
        self.cooldown_seconds = cooldown_seconds
        self.reported_tracks = {}  # track_id -> last_report_time

    def _should_report(self, track_id, current_time):
        """Check if enough time has passed since last violation report."""
        if track_id not in self.reported_tracks:
            return True

        last_time = self.reported_tracks[track_id]
        elapsed = (current_time - last_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def check_frame(self, frame_idx, timestamp, all_detections):
        """
        Check all detections in a frame for triple riding violations.

        Args:
            frame_idx: Current frame index
            timestamp: Current timestamp (datetime)
            all_detections: List of dicts with keys: bbox, class (int), track_id, score

        Returns:
            List of violation events
        """
        # Separate motorcycles and persons
        motorcycles = []
        persons = []

        for det in all_detections:
            cls = det.get('class')
            # Handle both int class_id and string class name
            if isinstance(cls, str):
                if cls.lower() in ['motorcycle', 'motorbike']:
                    motorcycles.append(det)
                elif cls.lower() in ['person', 'pedestrian']:
                    persons.append(det)
            else:
                if cls == CLASS_MOTORCYCLE:
                    motorcycles.append(det)
                elif cls == CLASS_PEDESTRIAN:
                    persons.append(det)

        if not motorcycles or not persons:
            return []

        violations = []

        for moto in motorcycles:
            moto_bbox = moto['bbox']
            track_id = moto.get('track_id', -1)

            # Expand motorcycle bbox to catch nearby/overlapping persons
            expanded_bbox = expand_bbox(moto_bbox, self.bbox_expansion)

            # Count persons overlapping with this motorcycle
            overlapping_persons = []

            for person in persons:
                person_bbox = person['bbox']

                # Check IoU with expanded bbox
                iou = calculate_iou(expanded_bbox, person_bbox)

                # Also check if person center is inside expanded motorcycle bbox
                person_cx = (person_bbox[0] + person_bbox[2]) / 2
                person_cy = (person_bbox[1] + person_bbox[3]) / 2
                center_inside = bbox_contains_point(expanded_bbox, (person_cx, person_cy))

                if iou >= self.iou_threshold or center_inside:
                    overlapping_persons.append({
                        'bbox': person_bbox,
                        'score': person.get('score', 0),
                        'iou': iou
                    })

            person_count = len(overlapping_persons)

            # Check for triple riding
            if person_count >= self.min_persons:
                if self._should_report(track_id, timestamp):
                    ev = {
                        'event_id': str(uuid.uuid4()),
                        'event_type': 'triple_riding_violation',
                        'track_id': track_id,
                        'class_id': CLASS_MOTORCYCLE,
                        'score': moto.get('score', 0),
                        'frame_id': frame_idx,
                        'timestamp': timestamp.isoformat(),
                        'bbox': moto_bbox,
                        'meta': {
                            'person_count': person_count,
                            'min_required': self.min_persons,
                            'persons': overlapping_persons,
                            'expanded_bbox': expanded_bbox
                        }
                    }

                    self.reported_tracks[track_id] = timestamp
                    self._emit_event(ev)
                    violations.append(ev)

        return violations

    def process_track(self, track_id, bbox, timestamp, frame_id, all_detections=None, **kwargs):
        """
        Alternative interface matching other rules.
        Delegates to check_frame if all_detections provided.
        """
        if all_detections is not None:
            return self.check_frame(frame_id, timestamp, all_detections)
        return None

    def _emit_event(self, ev):
        """Write event to JSONL file."""
        runid = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = EVENTS_DIR / f'{runid}.jsonl'
        with open(out, 'a') as f:
            f.write(json.dumps(ev) + '\n')

    def get_rider_count_overlay(self, moto_bbox, person_count):
        """
        Return info for drawing rider count on frame.
        To be used by visualization code.
        """
        x1, y1, x2, y2 = moto_bbox
        return {
            'position': (int(x1), int(y1) - 25),
            'text': f'Riders: {person_count}',
            'color': (0, 0, 255) if person_count >= self.min_persons else (0, 255, 0),
            'is_violation': person_count >= self.min_persons
        }
