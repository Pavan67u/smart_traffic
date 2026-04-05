"""
Verify zone geometry is correct and properly separated.
"""
import cv2
import numpy as np
import json
from pathlib import Path

def visualize_zones(camera_id='default', output_path='zone_visualization.jpg'):
    """Draw all zones on a sample frame for visual verification."""

    # Load camera config
    config_path = Path(__file__).parent / 'config' / 'cameras.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    cam_cfg = config.get(camera_id, config.get('default', {}))
    ref_res = cam_cfg.get('reference_resolution', [1280, 720])
    w, h = int(ref_res[0]), int(ref_res[1])

    # Create blank frame
    frame = np.ones((h, w, 3), dtype=np.uint8) * 240

    # Draw grid and labels
    cv2.putText(frame, f'Camera: {camera_id} ({w}x{h})', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Draw zones
    colors = {
        'stop_zone': (0, 0, 255),      # Red - stop line zone (bottom)
        'direction_zone': (0, 255, 255), # Yellow - direction zone (main area)
        'zebra_crossing_zone': (0, 255, 0),  # Green - zebra crossing
        'lane_line': (255, 0, 0),      # Blue - lane divider
    }

    # Stop Zone
    stop_zone = cam_cfg.get('stop_zone')
    if stop_zone:
        pts = np.array(stop_zone, dtype=np.int32)
        cv2.polylines(frame, [pts], True, colors['stop_zone'], 3)
        # Fill with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], colors['stop_zone'])
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.putText(frame, 'STOP ZONE', tuple(pts[0] + np.array([5, 25])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['stop_zone'], 2)

    # Direction Zone
    direction_zone = cam_cfg.get('direction_zone')
    if direction_zone:
        pts = np.array(direction_zone, dtype=np.int32)
        cv2.polylines(frame, [pts], True, colors['direction_zone'], 3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], colors['direction_zone'])
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        cv2.putText(frame, 'DIRECTION ZONE', tuple(pts[0] + np.array([5, 25])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['direction_zone'], 2)

        # Draw expected direction arrow
        expected_dir = cam_cfg.get('expected_direction', [0, 1])
        cx = int(sum(p[0] for p in direction_zone) / len(direction_zone))
        cy = int(sum(p[1] for p in direction_zone) / len(direction_zone))
        arrow_len = 80
        dx, dy = expected_dir
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx, dy = dx/norm, dy/norm
        end_x = int(cx + dx * arrow_len)
        end_y = int(cy + dy * arrow_len)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 0, 0), 3, tipLength=0.3)

    # Zebra Crossing Zone
    zebra_zone = cam_cfg.get('zebra_crossing_zone')
    if zebra_zone:
        pts = np.array(zebra_zone, dtype=np.int32)
        cv2.polylines(frame, [pts], True, colors['zebra_crossing_zone'], 3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], colors['zebra_crossing_zone'])
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.putText(frame, 'ZEBRA CROSSING', tuple(pts[0] + np.array([5, 25])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['zebra_crossing_zone'], 2)

    # Lane Line
    lane_line = cam_cfg.get('lane_line')
    if lane_line and len(lane_line) == 2:
        p1 = tuple(map(int, lane_line[0]))
        p2 = tuple(map(int, lane_line[1]))
        cv2.line(frame, p1, p2, colors['lane_line'], 3)
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(frame, 'LANE', (p1[0] - 50, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['lane_line'], 2)

    # Add legend
    legend_y = h - 100
    cv2.putText(frame, 'Legend:', (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.rectangle(frame, (20, legend_y + 20), (35, legend_y + 35), colors['stop_zone'], -1)
    cv2.putText(frame, 'Stop/Red Light Zone', (45, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(frame, (20, legend_y + 40), (35, legend_y + 55), colors['direction_zone'], -1)
    cv2.putText(frame, 'Direction Zone', (45, legend_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(frame, (20, legend_y + 60), (35, legend_y + 75), colors['zebra_crossing_zone'], -1)
    cv2.putText(frame, 'Zebra Crossing Zone', (45, legend_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save
    cv2.imwrite(output_path, frame)
    print(f"✓ Saved visualization: {output_path}")
    return frame

def verify_zone_separation(camera_id='default'):
    """Verify zones don't overlap unexpectedly."""
    config_path = Path(__file__).parent / 'config' / 'cameras.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    cam_cfg = config.get(camera_id, config.get('default', {}))
    ref_res = cam_cfg.get('reference_resolution', [1280, 720])
    w, h = int(ref_res[0]), int(ref_res[1])

    print(f"\n{'='*60}")
    print(f"Camera: {camera_id} ({w}x{h})")
    print(f"{'='*60}")

    issues = []

    # Check stop zone
    stop_zone = cam_cfg.get('stop_zone')
    if stop_zone:
        min_y = min(p[1] for p in stop_zone)
        max_y = max(p[1] for p in stop_zone)
        print(f"✓ Stop Zone Y range: {min_y}-{max_y}")
        if max_y <= min_y:
            issues.append(f"Stop zone has invalid Y range: {max_y} <= {min_y}")

    # Check direction zone
    direction_zone = cam_cfg.get('direction_zone')
    if direction_zone:
        min_y = min(p[1] for p in direction_zone)
        max_y = max(p[1] for p in direction_zone)
        print(f"✓ Direction Zone Y range: {min_y}-{max_y}")

    # Check zebra crossing zone
    zebra_zone = cam_cfg.get('zebra_crossing_zone')
    if zebra_zone:
        min_y = min(p[1] for p in zebra_zone)
        max_y = max(p[1] for p in zebra_zone)
        print(f"✓ Zebra Crossing Zone Y range: {min_y}-{max_y}")

    # Verify no overlap between zebra and stop zones
    if zebra_zone and stop_zone:
        z_min = min(p[1] for p in zebra_zone)
        z_max = max(p[1] for p in zebra_zone)
        s_min = min(p[1] for p in stop_zone)

        if z_max >= s_min:
            issues.append(f"⚠ Zones may overlap: zebra_max({z_max}) >= stop_min({s_min})")
        else:
            gap = s_min - z_max
            print(f"✓ Zebra-Stop gap: {gap}px (separation OK)")

    # Check lane line is between 20-80% of width
    lane_line = cam_cfg.get('lane_line')
    if lane_line:
        x1, x2 = lane_line[0][0], lane_line[1][0]
        x_pct = (x1 / w) * 100
        if 20 <= x_pct <= 80:
            print(f"✓ Lane line at {x_pct:.1f}% of width (good center positioning)")
        else:
            issues.append(f"Lane line at {x_pct:.1f}% (should be 20-80%)")

    # Check expected direction
    exp_dir = cam_cfg.get('expected_direction')
    if exp_dir:
        print(f"✓ Expected direction: {exp_dir}")

    print(f"\n{'─'*60}")
    if issues:
        print("⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✓ All zones properly configured!")
        return True

if __name__ == '__main__':
    print("Verifying zone geometry for all cameras...\n")

    # Load config
    config_path = Path(__file__).parent / 'config' / 'cameras.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    all_good = True
    for camera_id in ['default', 'intersection_a', 'intersection_b', 'lab_cam_01']:
        if camera_id in config:
            result = verify_zone_separation(camera_id)
            all_good = all_good and result
            visualize_zones(camera_id, f'zone_viz_{camera_id}.jpg')

    print(f"\n{'='*60}")
    if all_good:
        print("✓ ALL CAMERAS HAVE VALID ZONE CONFIGURATIONS")
    else:
        print("✗ SOME ISSUES FOUND - CHECK ABOVE")
    print(f"{'='*60}\n")
