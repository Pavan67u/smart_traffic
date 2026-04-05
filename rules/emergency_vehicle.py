"""
Emergency Vehicle Detection.
Detects ambulances, fire trucks, and police vehicles using color and flicker analysis.
"""
import cv2
import numpy as np
from collections import deque
from datetime import datetime, timezone


class EmergencyVehicleDetector:
    """
    Detects emergency vehicles using visual cues:
    1. Red/Blue emergency light colors in roof area
    2. Flashing light patterns (brightness oscillation)
    """

    def __init__(self, flicker_threshold=0.25, history_frames=15, color_ratio_threshold=0.03):
        """
        Args:
            flicker_threshold: Brightness change ratio to detect flicker
            history_frames: Number of frames to analyze for flicker pattern
            color_ratio_threshold: Minimum ratio of emergency colors in roof area
        """
        self.flicker_threshold = flicker_threshold
        self.history_frames = history_frames
        self.color_ratio_threshold = color_ratio_threshold
        self.track_history = {}  # track_id -> deque of brightness values

        # HSV color ranges for emergency lights
        # Red (wraps around hue 0/180)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])

        # Blue
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([130, 255, 255])

        # White (bright lights)
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])

    def _get_roof_roi(self, bbox, frame_shape):
        """
        Extract top portion of bbox (roof area where lights would be).
        Returns (x1, y1, x2, y2) for ROI, or None if invalid.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1

        if h < 10:  # Too small
            return None

        # Top 30% of the vehicle (roof area)
        roof_height = int(h * 0.3)
        roof_y2 = y1 + roof_height

        # Bounds check
        frame_h, frame_w = frame_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        roof_y2 = min(frame_h, roof_y2)

        if x2 <= x1 or roof_y2 <= y1:
            return None

        return (x1, y1, x2, roof_y2)

    def _detect_emergency_colors(self, roi_img):
        """
        Check for red/blue/white emergency light colors.

        Returns:
            (has_emergency_colors, color_details)
        """
        if roi_img is None or roi_img.size == 0:
            return False, {}

        try:
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return False, {}

        total_pixels = roi_img.shape[0] * roi_img.shape[1]
        if total_pixels == 0:
            return False, {}

        # Red detection (two ranges for hue wrap-around)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Blue detection
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # White/bright detection
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        red_ratio = np.sum(red_mask > 0) / total_pixels
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        white_ratio = np.sum(white_mask > 0) / total_pixels

        # Emergency lights typically have red OR blue, often with white
        has_emergency_colors = (
            red_ratio > self.color_ratio_threshold or
            blue_ratio > self.color_ratio_threshold or
            (red_ratio > 0.01 and blue_ratio > 0.01)  # Both colors present
        )

        return has_emergency_colors, {
            'red_ratio': round(red_ratio, 4),
            'blue_ratio': round(blue_ratio, 4),
            'white_ratio': round(white_ratio, 4)
        }

    def _detect_flicker(self, track_id, roi_brightness):
        """
        Detect light flickering pattern from brightness history.
        Emergency lights typically flash at 1-3 Hz.

        Returns:
            (is_flickering, flicker_rate)
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.history_frames)

        self.track_history[track_id].append(roi_brightness)
        history = list(self.track_history[track_id])

        if len(history) < 5:
            return False, 0

        # Count significant brightness changes
        changes = 0
        for i in range(1, len(history)):
            prev_br = max(history[i - 1], 1)  # Avoid division by zero
            diff_ratio = abs(history[i] - history[i - 1]) / prev_br

            if diff_ratio > self.flicker_threshold:
                changes += 1

        flicker_rate = changes / (len(history) - 1)

        # Emergency lights show regular oscillation
        # At least 20% of frames should show brightness change
        is_flickering = flicker_rate > 0.2

        return is_flickering, round(flicker_rate, 3)

    def check_vehicle(self, track_id, bbox, frame):
        """
        Check if a vehicle appears to be an emergency vehicle.

        Args:
            track_id: Vehicle track ID
            bbox: Vehicle bounding box [x1, y1, x2, y2]
            frame: Current video frame (BGR)

        Returns:
            (is_emergency, confidence, details)
        """
        roi_coords = self._get_roof_roi(bbox, frame.shape)
        if roi_coords is None:
            return False, 0, {}

        x1, y1, x2, y2 = roi_coords
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False, 0, {}

        # Check emergency colors
        has_colors, color_details = self._detect_emergency_colors(roi)

        # Check flicker pattern
        brightness = np.mean(roi)
        is_flickering, flicker_rate = self._detect_flicker(track_id, brightness)

        # Calculate confidence score
        score = 0.0

        if has_colors:
            # Weight by color intensity
            max_color = max(color_details.get('red_ratio', 0), color_details.get('blue_ratio', 0))
            score += min(0.5, max_color * 10)  # Cap at 0.5

        if is_flickering:
            score += 0.3 + (flicker_rate * 0.2)  # Up to 0.5 for strong flicker

        # Threshold for emergency classification
        is_emergency = score >= 0.4

        return is_emergency, round(score, 3), {
            'has_emergency_colors': has_colors,
            'is_flickering': is_flickering,
            'flicker_rate': flicker_rate,
            'brightness': round(brightness, 2),
            'roi_coords': roi_coords,
            **color_details
        }

    def cleanup_old_tracks(self, active_track_ids):
        """Remove old tracks not in active set."""
        to_remove = [tid for tid in self.track_history if tid not in active_track_ids]
        for tid in to_remove:
            del self.track_history[tid]


class EmergencyVehiclePriority:
    """
    Manages signal priority for detected emergency vehicles.
    """

    def __init__(self, signal_manager=None, priority_duration_seconds=30):
        """
        Args:
            signal_manager: SignalManager instance for signal control
            priority_duration_seconds: How long to maintain green for emergency vehicle
        """
        self.signal_manager = signal_manager
        self.priority_duration = priority_duration_seconds
        self.active_priority = None  # (track_id, start_time)

    def trigger_priority(self, track_id, direction='incoming'):
        """
        Trigger signal priority for emergency vehicle.

        Args:
            track_id: Emergency vehicle track ID
            direction: 'incoming' or 'outgoing' to determine signal phase
        """
        if self.signal_manager is None:
            return {'status': 'no_signal_manager'}

        now = datetime.now(timezone.utc)

        # Check if priority already active
        if self.active_priority:
            existing_id, start_time = self.active_priority
            elapsed = (now - start_time).total_seconds()
            if elapsed < self.priority_duration:
                return {'status': 'priority_active', 'track_id': existing_id}

        # Activate green signal for emergency vehicle
        try:
            self.signal_manager.set_state('GREEN')
            self.active_priority = (track_id, now)
            return {
                'status': 'priority_activated',
                'track_id': track_id,
                'duration': self.priority_duration
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def check_priority_expiry(self):
        """Check if active priority has expired and reset signal if needed."""
        if self.active_priority is None:
            return

        track_id, start_time = self.active_priority
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        if elapsed >= self.priority_duration:
            self.active_priority = None
            # Signal manager will resume normal operation
            return {'status': 'priority_expired', 'track_id': track_id}

        return None
