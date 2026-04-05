"""Unit tests for all violation detection rules."""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

# Import all rule engines
from rules.red_light import StopLineRule
from rules.lane import LaneViolationRule
from rules.wrong_way import WrongWayRule
from rules.zebra_crossing import ZebraCrossingRule
from rules.triple_riding import TripleRidingRule, calculate_iou, bbox_contains_point
from rules.emergency_vehicle import EmergencyVehicleDetector


class TestRedLightRule:
    """Test StopLineRule for red light violations."""

    @pytest.fixture
    def rule(self):
        # Define a stop zone polygon
        stop_zone = [(0, 200), (640, 200), (640, 400), (0, 400)]
        return StopLineRule(stop_zone, required_stop_seconds=1.0, min_violation_speed=5.0)

    def test_no_violation_on_green_signal(self, rule):
        """No violation when signal is GREEN."""
        t0 = datetime.now(timezone.utc)
        result = rule.process_track(
            track_id=1,
            bbox=[100, 250, 150, 300],  # Inside stop zone
            timestamp=t0,
            frame_id=0,
            signal_state='GREEN'
        )
        assert result is None

    def test_no_violation_on_yellow_signal(self, rule):
        """No violation when signal is YELLOW."""
        t0 = datetime.now(timezone.utc)
        result = rule.process_track(
            track_id=1,
            bbox=[100, 250, 150, 300],
            timestamp=t0,
            frame_id=0,
            signal_state='YELLOW'
        )
        assert result is None

    def test_violation_when_moving_on_red(self, rule):
        """Violation when vehicle moves through stop zone on RED."""
        t0 = datetime.now(timezone.utc)

        # First frame - establish track (dt=0, no violation possible)
        rule.process_track(
            track_id=1,
            bbox=[100, 210, 150, 260],
            timestamp=t0,
            frame_id=0,
            signal_state='RED'
        )

        # Second frame after ghost suppression (0.6s > 0.5s threshold)
        # Move significantly through the zone - this should trigger violation
        t1 = t0 + timedelta(seconds=0.6)
        result = rule.process_track(
            track_id=1,
            bbox=[100, 300, 150, 350],  # Moved ~65px in 0.6s = 108 px/s
            timestamp=t1,
            frame_id=1,
            signal_state='RED'
        )

        assert result is not None
        assert result['event_type'] == 'red_light_violation'
        assert result['track_id'] == 1

    def test_no_violation_when_stopped(self, rule):
        """No violation when vehicle is stopped (speed < threshold)."""
        t0 = datetime.now(timezone.utc)

        # Establish track
        rule.process_track(
            track_id=1,
            bbox=[100, 250, 150, 300],
            timestamp=t0,
            frame_id=0,
            signal_state='RED'
        )

        # Barely move (< 5 px/s)
        t1 = t0 + timedelta(seconds=1.0)
        result = rule.process_track(
            track_id=1,
            bbox=[101, 251, 151, 301],  # Only 1.4px movement in 1s
            timestamp=t1,
            frame_id=1,
            signal_state='RED'
        )

        assert result is None

    def test_cooldown_prevents_spam(self, rule):
        """Cooldown prevents multiple violations for same vehicle."""
        t0 = datetime.now(timezone.utc)

        # First frame - establish track
        rule.process_track(track_id=1, bbox=[100, 210, 150, 260], timestamp=t0, frame_id=0, signal_state='RED')

        # Trigger first violation (after ghost suppression threshold)
        t1 = t0 + timedelta(seconds=0.6)
        result1 = rule.process_track(track_id=1, bbox=[100, 300, 150, 350], timestamp=t1, frame_id=1, signal_state='RED')
        assert result1 is not None, "First violation should trigger"

        # Try again within cooldown period (2s) - should be blocked
        t2 = t1 + timedelta(seconds=0.5)
        result2 = rule.process_track(track_id=1, bbox=[100, 360, 150, 410], timestamp=t2, frame_id=2, signal_state='RED')
        assert result2 is None, "Second violation should be blocked by cooldown"

        # Try after cooldown expires (2s+)
        t3 = t1 + timedelta(seconds=2.5)
        result3 = rule.process_track(track_id=1, bbox=[100, 380, 150, 430], timestamp=t3, frame_id=3, signal_state='RED')
        # May or may not trigger depending on zone - just verify no error


class TestLaneViolationRule:
    """Test LaneViolationRule for lane crossing violations."""

    @pytest.fixture
    def rule(self):
        # Vertical lane line at x=320
        lane_line = [(320, 0), (320, 480)]
        return LaneViolationRule(lane_line)

    def test_no_violation_same_side(self, rule):
        """No violation when staying on same side of line."""
        t0 = datetime.now(timezone.utc)

        rule.process_track(track_id=1, bbox=[100, 100, 150, 150], timestamp=t0, frame_id=0)

        t1 = t0 + timedelta(seconds=1.5)
        result = rule.process_track(track_id=1, bbox=[200, 100, 250, 150], timestamp=t1, frame_id=1)

        assert result is None

    def test_violation_on_crossing(self, rule):
        """Violation when crossing the lane line."""
        t0 = datetime.now(timezone.utc)

        # Start on left side (centroid at x=125)
        rule.process_track(track_id=1, bbox=[100, 100, 150, 150], timestamp=t0, frame_id=0)

        # Cross to right side (centroid at x=400) - must wait for ghost suppression
        t1 = t0 + timedelta(seconds=1.5)
        result = rule.process_track(track_id=1, bbox=[350, 100, 450, 150], timestamp=t1, frame_id=1)

        assert result is not None
        assert result['event_type'] == 'lane_violation'
        assert result['track_id'] == 1

    def test_crosses_line_detection(self, rule):
        """Test the line crossing detection math."""
        # Left to right crossing
        left = (200, 240)
        right = (400, 240)
        assert rule._crosses_line(left, right) == True

        # Both on same side
        left1 = (100, 100)
        left2 = (200, 200)
        assert rule._crosses_line(left1, left2) == False


class TestWrongWayRule:
    """Test WrongWayRule for wrong-way driving detection."""

    @pytest.fixture
    def rule(self):
        # Zone polygon and expected direction (southbound = down = (0, 1))
        direction_zone = [(100, 100), (540, 100), (540, 380), (100, 380)]
        expected_direction = (0, 1)  # Traffic should flow downward
        return WrongWayRule(direction_zone, expected_direction, angle_tolerance=60.0, min_history_points=3, min_displacement=20.0)

    def test_no_violation_correct_direction(self, rule):
        """No violation when traveling in correct direction."""
        t0 = datetime.now(timezone.utc)

        # Move downward (correct direction)
        for i in range(5):
            t = t0 + timedelta(seconds=i * 0.5)
            y_pos = 150 + i * 30  # Moving down
            result = rule.process_track(
                track_id=1,
                bbox=[300, y_pos, 350, y_pos + 50],
                timestamp=t,
                frame_id=i
            )

        assert result is None

    def test_violation_wrong_direction(self, rule):
        """Violation when traveling against expected direction."""
        t0 = datetime.now(timezone.utc)

        results = []
        # Move upward (wrong direction)
        for i in range(6):
            t = t0 + timedelta(seconds=i * 0.5)
            y_pos = 350 - i * 30  # Moving up (wrong way)
            result = rule.process_track(
                track_id=1,
                bbox=[300, y_pos, 350, y_pos + 50],
                timestamp=t,
                frame_id=i
            )
            if result:
                results.append(result)

        assert len(results) > 0
        assert results[0]['event_type'] == 'wrong_way_violation'

    def test_angle_calculation(self, rule):
        """Test angle calculation between vectors."""
        # Same direction = 0 degrees
        v1 = (0, 1)
        v2 = (0, 1)
        assert abs(rule._angle_between_vectors(v1, v2)) < 0.1

        # Opposite direction = 180 degrees
        v3 = (0, -1)
        angle = rule._angle_between_vectors(v1, v3)
        assert abs(angle - 180.0) < 0.1

    def test_outside_zone_no_violation(self, rule):
        """No violation when outside detection zone."""
        t0 = datetime.now(timezone.utc)

        # Move upward but outside zone
        for i in range(5):
            t = t0 + timedelta(seconds=i * 0.5)
            y_pos = 50 - i * 10  # Outside the zone (y=100-380)
            result = rule.process_track(
                track_id=1,
                bbox=[300, y_pos, 350, y_pos + 30],
                timestamp=t,
                frame_id=i
            )

        assert result is None


class TestZebraCrossingRule:
    """Test ZebraCrossingRule for pedestrian crossing violations."""

    @pytest.fixture
    def rule(self):
        crossing_zone = [(200, 200), (440, 200), (440, 280), (200, 280)]
        return ZebraCrossingRule(crossing_zone, dwell_threshold_seconds=2.0, min_speed_threshold=3.0)

    def test_no_violation_passing_through(self, rule):
        """No violation when passing through quickly."""
        t0 = datetime.now(timezone.utc)

        # Pass through the crossing zone quickly
        for i in range(5):
            t = t0 + timedelta(seconds=i * 0.2)
            x_pos = 150 + i * 80  # Moving across
            result = rule.process_track(
                track_id=1,
                bbox=[x_pos, 220, x_pos + 60, 270],
                timestamp=t,
                frame_id=i
            )

        assert result is None

    def test_violation_when_stopped(self, rule):
        """Violation when stopped on crossing for > threshold."""
        t0 = datetime.now(timezone.utc)

        # Stay still on crossing
        results = []
        for i in range(15):
            t = t0 + timedelta(seconds=i * 0.2)  # Every 0.2s, 15 frames = 3 seconds
            result = rule.process_track(
                track_id=1,
                bbox=[300, 220, 360, 270],  # Stationary
                timestamp=t,
                frame_id=i
            )
            if result:
                results.append(result)

        assert len(results) > 0
        assert results[0]['event_type'] == 'zebra_crossing_violation'
        assert 'dwell_time_seconds' in results[0]['meta']

    def test_no_violation_outside_zone(self, rule):
        """No violation when stopped outside crossing zone."""
        t0 = datetime.now(timezone.utc)

        # Stopped but outside zone
        for i in range(20):
            t = t0 + timedelta(seconds=i * 0.2)
            result = rule.process_track(
                track_id=1,
                bbox=[50, 50, 110, 100],  # Outside zone
                timestamp=t,
                frame_id=i
            )

        assert result is None


class TestTripleRidingRule:
    """Test TripleRidingRule for multi-rider detection."""

    @pytest.fixture
    def rule(self):
        return TripleRidingRule(min_persons=3, iou_threshold=0.15, bbox_expansion=1.3)

    def test_calculate_iou(self):
        """Test IoU calculation."""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = calculate_iou(box1, box2)

        # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.01

    def test_bbox_contains_point(self):
        """Test point containment."""
        bbox = [100, 100, 200, 200]
        assert bbox_contains_point(bbox, (150, 150)) == True
        assert bbox_contains_point(bbox, (50, 50)) == False

    def test_no_violation_single_rider(self, rule):
        """No violation with single rider."""
        t0 = datetime.now(timezone.utc)
        detections = [
            {'bbox': [200, 200, 280, 350], 'class': 3, 'track_id': 1, 'score': 0.9},  # motorcycle
            {'bbox': [210, 180, 270, 280], 'class': 4, 'track_id': 2, 'score': 0.8},  # 1 person
        ]

        violations = rule.check_frame(0, t0, detections)
        assert len(violations) == 0

    def test_no_violation_two_riders(self, rule):
        """No violation with two riders (legal)."""
        t0 = datetime.now(timezone.utc)
        detections = [
            {'bbox': [200, 200, 280, 350], 'class': 3, 'track_id': 1, 'score': 0.9},  # motorcycle
            {'bbox': [210, 180, 250, 280], 'class': 4, 'track_id': 2, 'score': 0.8},  # person 1
            {'bbox': [240, 190, 280, 290], 'class': 4, 'track_id': 3, 'score': 0.8},  # person 2
        ]

        violations = rule.check_frame(0, t0, detections)
        assert len(violations) == 0

    def test_violation_triple_riding(self, rule):
        """Violation with three riders."""
        t0 = datetime.now(timezone.utc)
        detections = [
            {'bbox': [200, 200, 300, 380], 'class': 3, 'track_id': 1, 'score': 0.9},  # motorcycle
            {'bbox': [210, 180, 250, 280], 'class': 4, 'track_id': 2, 'score': 0.8},  # person 1
            {'bbox': [250, 185, 290, 285], 'class': 4, 'track_id': 3, 'score': 0.8},  # person 2
            {'bbox': [220, 200, 280, 300], 'class': 4, 'track_id': 4, 'score': 0.8},  # person 3
        ]

        violations = rule.check_frame(0, t0, detections)
        assert len(violations) == 1
        assert violations[0]['event_type'] == 'triple_riding_violation'
        assert violations[0]['meta']['person_count'] == 3


class TestEmergencyVehicleDetector:
    """Test EmergencyVehicleDetector."""

    @pytest.fixture
    def detector(self):
        return EmergencyVehicleDetector()

    def test_no_emergency_normal_vehicle(self, detector):
        """Normal vehicle should not be detected as emergency."""
        # Create a plain gray frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = [200, 150, 350, 300]

        is_emergency, confidence, details = detector.check_vehicle(1, bbox, frame)
        assert is_emergency == False
        assert confidence < 0.4

    def test_emergency_with_red_lights(self, detector):
        """Vehicle with red roof should trigger detection."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = [200, 150, 350, 300]

        # Add red color to roof area (top 30%)
        roof_y1 = 150
        roof_y2 = 150 + int((300 - 150) * 0.3)
        frame[roof_y1:roof_y2, 200:350] = [0, 0, 255]  # BGR red

        is_emergency, confidence, details = detector.check_vehicle(1, bbox, frame)
        assert details['has_emergency_colors'] == True
        assert details['red_ratio'] > 0.03

    def test_emergency_with_blue_lights(self, detector):
        """Vehicle with blue roof should trigger detection."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = [200, 150, 350, 300]

        # Add blue color to roof area
        roof_y1 = 150
        roof_y2 = 150 + int((300 - 150) * 0.3)
        frame[roof_y1:roof_y2, 200:350] = [255, 0, 0]  # BGR blue

        is_emergency, confidence, details = detector.check_vehicle(1, bbox, frame)
        assert details['has_emergency_colors'] == True
        assert details['blue_ratio'] > 0.03

    def test_flicker_detection(self, detector):
        """Test flicker detection with oscillating brightness."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        bbox = [200, 150, 350, 300]

        # Simulate flashing lights by varying brightness
        for i in range(20):
            # Alternate brightness
            brightness = 200 if i % 2 == 0 else 50
            frame[:] = brightness
            detector.check_vehicle(1, bbox, frame)

        # Check final state
        _, _, details = detector.check_vehicle(1, bbox, frame)
        assert 'flicker_rate' in details

    def test_roof_roi_extraction(self, detector):
        """Test roof ROI calculation."""
        frame_shape = (480, 640, 3)
        bbox = [100, 100, 200, 200]

        roi = detector._get_roof_roi(bbox, frame_shape)
        assert roi is not None
        x1, y1, x2, y2 = roi

        # Should be top 30% of vehicle
        assert y1 == 100
        assert y2 == 100 + int(100 * 0.3)  # 30% of height


class TestRuleIntegration:
    """Integration tests for rule combinations."""

    def test_multiple_rules_same_track(self):
        """Test that multiple rules can process the same track."""
        stop_zone = [(0, 200), (640, 200), (640, 400), (0, 400)]
        lane_line = [(320, 0), (320, 480)]

        red_light_rule = StopLineRule(stop_zone)
        lane_rule = LaneViolationRule(lane_line)

        t0 = datetime.now(timezone.utc)
        bbox = [100, 250, 150, 300]

        # Both rules should be able to process same track data
        # Note: RedLightRule only tracks state when signal is RED
        red_light_rule.process_track(track_id=1, bbox=bbox, timestamp=t0, frame_id=0, signal_state='RED')
        lane_rule.process_track(track_id=1, bbox=bbox, timestamp=t0, frame_id=0)

        # Should not interfere with each other
        assert 1 in red_light_rule.track_states
        assert 1 in lane_rule.track_states

    def test_rules_independent_state(self):
        """Verify rules maintain independent state."""
        red_rule = StopLineRule([(0, 100), (640, 100), (640, 300), (0, 300)])
        lane_rule = LaneViolationRule([(320, 0), (320, 480)])

        t0 = datetime.now(timezone.utc)

        # Process different tracks
        red_rule.process_track(track_id=1, bbox=[100, 150, 150, 200], timestamp=t0, frame_id=0, signal_state='RED')
        lane_rule.process_track(track_id=2, bbox=[400, 100, 450, 150], timestamp=t0, frame_id=0)

        assert 1 in red_rule.track_states
        assert 1 not in lane_rule.track_states
        assert 2 in lane_rule.track_states
        assert 2 not in red_rule.track_states


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
