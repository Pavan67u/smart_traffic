"""
Traffic Density Heatmap Generator.
Real-time visualization of traffic density on video frames.
"""
import cv2
import numpy as np
from collections import deque


class DensityHeatmap:
    """
    Generates a real-time traffic density heatmap overlay.

    Divides the frame into a grid and tracks vehicle density per cell,
    using exponential moving average for smooth transitions.
    """

    def __init__(self, grid_size=(8, 12), decay_factor=0.85, history_frames=30):
        """
        Args:
            grid_size: (rows, cols) for grid division
            decay_factor: EMA decay (0.85 = moderate smoothing)
            history_frames: Number of frames for density averaging
        """
        self.grid_rows, self.grid_cols = grid_size
        self.decay_factor = decay_factor
        self.history_frames = history_frames

        self.density_grid = None  # Will be initialized on first frame
        self.frame_size = None

        # Color gradient: green -> yellow -> orange -> red
        self.color_stops = [
            (0, (0, 255, 0)),      # Green (low density)
            (2, (0, 255, 255)),    # Yellow
            (4, (0, 165, 255)),    # Orange
            (6, (0, 0, 255)),      # Red (high density)
        ]

    def _get_cell(self, centroid, frame_w, frame_h):
        """Get grid cell (row, col) for a centroid."""
        x, y = centroid
        col = min(int(x / frame_w * self.grid_cols), self.grid_cols - 1)
        row = min(int(y / frame_h * self.grid_rows), self.grid_rows - 1)
        return max(0, row), max(0, col)

    def _interpolate_color(self, density):
        """Interpolate color based on density value."""
        # Find the two color stops to interpolate between
        prev_stop = self.color_stops[0]
        next_stop = self.color_stops[-1]

        for i, (threshold, color) in enumerate(self.color_stops):
            if density <= threshold:
                next_stop = (threshold, color)
                if i > 0:
                    prev_stop = self.color_stops[i - 1]
                break
            prev_stop = (threshold, color)

        # Interpolate
        prev_thresh, prev_color = prev_stop
        next_thresh, next_color = next_stop

        if next_thresh == prev_thresh:
            return prev_color

        t = (density - prev_thresh) / (next_thresh - prev_thresh)
        t = max(0, min(1, t))

        color = tuple(int(prev_color[i] + t * (next_color[i] - prev_color[i])) for i in range(3))
        return color

    def update(self, detections, frame_shape):
        """
        Update density grid with current frame detections.

        Args:
            detections: List of detection dicts with 'bbox' key
            frame_shape: Shape of the video frame (h, w, c)

        Returns:
            numpy array: Current density grid
        """
        h, w = frame_shape[:2]
        self.frame_size = (w, h)

        if self.density_grid is None:
            self.density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # Count detections per cell
        current_counts = np.zeros_like(self.density_grid)

        for det in detections:
            bbox = det.get('bbox', det.get('box', []))
            if len(bbox) < 4:
                continue

            # Get centroid
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            row, col = self._get_cell((cx, cy), w, h)
            current_counts[row, col] += 1

        # Exponential moving average for smooth transitions
        self.density_grid = (
            self.decay_factor * self.density_grid +
            (1 - self.decay_factor) * current_counts
        )

        return self.density_grid

    def render_overlay(self, frame, alpha=0.35, show_grid=True, show_counts=False):
        """
        Render semi-transparent heatmap overlay on frame.

        Args:
            frame: Video frame to overlay on (will be modified)
            alpha: Transparency of overlay (0-1)
            show_grid: Whether to draw grid lines
            show_counts: Whether to show count numbers in cells

        Returns:
            Frame with heatmap overlay
        """
        if self.density_grid is None or self.frame_size is None:
            return frame

        h, w = frame.shape[:2]
        cell_w = w // self.grid_cols
        cell_h = h // self.grid_rows

        overlay = frame.copy()

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                density = self.density_grid[row, col]

                if density < 0.1:  # Skip empty cells
                    continue

                # Get color based on density
                color = self._interpolate_color(density)

                # Calculate cell position
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                # Draw filled rectangle
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                # Show count text
                if show_counts and density >= 0.5:
                    text = f'{density:.1f}'
                    text_x = x1 + cell_w // 2 - 15
                    text_y = y1 + cell_h // 2 + 5
                    cv2.putText(overlay, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Blend overlay with original
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw grid lines
        if show_grid:
            for i in range(1, self.grid_cols):
                x = i * cell_w
                cv2.line(result, (x, 0), (x, h), (128, 128, 128), 1)
            for i in range(1, self.grid_rows):
                y = i * cell_h
                cv2.line(result, (0, y), (w, y), (128, 128, 128), 1)

        return result

    def get_stats(self):
        """Return current density statistics."""
        if self.density_grid is None:
            return {
                'max_density': 0,
                'mean_density': 0,
                'total_count': 0,
                'hotspots': [],
                'congestion_level': 'low'
            }

        max_density = float(np.max(self.density_grid))
        mean_density = float(np.mean(self.density_grid))
        total_count = float(np.sum(self.density_grid))

        # Find hotspots (cells with high density)
        hotspots = []
        threshold = max(2.0, mean_density + np.std(self.density_grid))
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self.density_grid[row, col] >= threshold:
                    hotspots.append({
                        'row': row,
                        'col': col,
                        'density': float(self.density_grid[row, col])
                    })

        # Determine congestion level
        if max_density >= 6:
            congestion_level = 'critical'
        elif max_density >= 4:
            congestion_level = 'high'
        elif max_density >= 2:
            congestion_level = 'medium'
        else:
            congestion_level = 'low'

        return {
            'max_density': round(max_density, 2),
            'mean_density': round(mean_density, 2),
            'total_count': round(total_count, 1),
            'hotspots': hotspots,
            'congestion_level': congestion_level,
            'grid': self.density_grid.tolist()
        }

    def reset(self):
        """Reset the density grid."""
        if self.density_grid is not None:
            self.density_grid.fill(0)


class VehicleCounter:
    """
    Simple vehicle counter by class.
    Tracks unique vehicles using track IDs.
    """

    def __init__(self):
        self.counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'pedestrian': 0,
            'total': 0
        }
        self.counted_tracks = set()

        # Class name mappings
        self.class_names = {
            0: 'car',
            1: 'truck',
            2: 'bus',
            3: 'motorcycle',
            4: 'pedestrian'
        }

    def count(self, track_id, class_id):
        """
        Count a vehicle if not already counted.

        Args:
            track_id: Unique track identifier
            class_id: Class ID or class name

        Returns:
            bool: True if counted (new vehicle), False if already counted
        """
        if track_id in self.counted_tracks:
            return False

        self.counted_tracks.add(track_id)

        # Get class name
        if isinstance(class_id, int):
            class_name = self.class_names.get(class_id, 'unknown')
        else:
            class_name = str(class_id).lower()

        if class_name in self.counts:
            self.counts[class_name] += 1

        self.counts['total'] += 1
        return True

    def get_counts(self):
        """Return current counts."""
        return dict(self.counts)

    def reset(self):
        """Reset all counts."""
        for key in self.counts:
            self.counts[key] = 0
        self.counted_tracks.clear()

    def render_overlay(self, frame, position=(10, 30)):
        """
        Render count overlay on frame.

        Args:
            frame: Video frame
            position: (x, y) position for text

        Returns:
            Frame with count overlay
        """
        x, y = position
        line_height = 25

        # Draw background
        bg_height = line_height * (len(self.counts) + 1)
        cv2.rectangle(frame, (x - 5, y - 20), (x + 180, y + bg_height), (0, 0, 0), -1)

        # Draw title
        cv2.putText(frame, 'Vehicle Counts', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw counts
        y_offset = y + line_height
        for name, count in self.counts.items():
            if name == 'total':
                continue
            text = f'{name.capitalize()}: {count}'
            cv2.putText(frame, text, (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height

        # Draw total
        cv2.putText(frame, f'Total: {self.counts["total"]}', (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
