"""Advanced ML features: ensemble models, active learning, video streaming."""
import json
import subprocess
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

# ============================================================================
# Model Ensemble & Confidence Aggregation
# ============================================================================

class ModelEnsemble:
    """Run multiple YOLO models in parallel and aggregate predictions."""
    
    def __init__(self, model_paths: list):
        """Initialize ensemble with multiple model paths."""
        self.models = []
        self.model_paths = model_paths
        
        try:
            from ultralytics import YOLO
            for path in model_paths:
                try:
                    model = YOLO(str(path))
                    self.models.append((path, model))
                    print(f"✓ Loaded ensemble model: {path}")
                except Exception as e:
                    print(f"✗ Failed to load model {path}: {e}")
        except ImportError:
            print("YOLOv8 not available")
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Run all models and aggregate predictions using voting."""
        if not self.models:
            return []
        
        all_predictions = []
        
        # Run each model
        for path, model in self.models:
            try:
                results = model.predict(str(image_path), conf=confidence_threshold, verbose=False)
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None:
                        for box in r.boxes:
                            all_predictions.append({
                                'model_path': path,
                                'bbox': box.xyxy[0].tolist(),
                                'confidence': float(box.conf),
                                'class_id': int(box.cls),
                                'class_name': r.names[int(box.cls)]
                            })
            except Exception as e:
                print(f"Ensemble prediction error with {path}: {e}")
        
        # Aggregate: group nearby boxes and average confidence
        aggregated = self._aggregate_boxes(all_predictions)
        return aggregated
    
    def _aggregate_boxes(self, predictions, iou_threshold=0.5):
        """NMS-style aggregation: cluster nearby boxes and average confidence."""
        if not predictions:
            return []
        
        # Sort by confidence descending
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        keeps = []
        used = set()
        
        for i, pred in enumerate(predictions):
            if i in used:
                continue
            
            cluster = [pred]
            used.add(i)
            
            # Find similar boxes
            for j in range(i + 1, len(predictions)):
                if j in used:
                    continue
                
                # Check IOU with current cluster's centroid
                iou = self._iou(pred['bbox'], predictions[j]['bbox'])
                if iou > iou_threshold:
                    cluster.append(predictions[j])
                    used.add(j)
            
            # Average the cluster
            aggregated = self._average_cluster(cluster)
            keeps.append(aggregated)
        
        return keeps
    
    @staticmethod
    def _iou(box1, box2):
        """Calculate IOU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        intersect_min_x = max(x1_min, x2_min)
        intersect_min_y = max(y1_min, y2_min)
        intersect_max_x = min(x1_max, x2_max)
        intersect_max_y = min(y1_max, y2_max)
        
        if intersect_max_x < intersect_min_x or intersect_max_y < intersect_min_y:
            return 0
        
        intersect_area = (intersect_max_x - intersect_min_x) * (intersect_max_y - intersect_min_y)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area
        
        return intersect_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def _average_cluster(cluster):
        """Average predictions in a cluster."""
        n = len(cluster)
        
        avg_bbox = [
            sum(p['bbox'][i] for p in cluster) / n
            for i in range(4)
        ]
        avg_confidence = sum(p['confidence'] for p in cluster) / n
        
        # Use most common class
        class_votes = {}
        for p in cluster:
            class_name = p['class_name']
            class_votes[class_name] = class_votes.get(class_name, 0) + 1
        dominant_class = max(class_votes, key=class_votes.get)
        
        return {
            'bbox': avg_bbox,
            'confidence': avg_confidence,
            'confidence_sources': n,  # How many models agreed
            'class_name': dominant_class,
            'class_id': cluster[0]['class_id']  # Assuming same class
        }


# ============================================================================
# Active Learning
# ============================================================================

def flag_uncertain_detections(detections, confidence_range=(0.50, 0.70)):
    """Flag detections with uncertain confidence for human labeling."""
    uncertain = []
    for det in detections:
        conf = det.get('confidence', 1.0)
        if confidence_range[0] <= conf <= confidence_range[1]:
            uncertain.append({
                **det,
                'flagged_for_labeling': True,
                'reason': f'Confidence {conf:.2f} in uncertain range'
            })
    return uncertain


# ============================================================================
# Video Streaming Integration (RTSP/MJPEG)
# ============================================================================

class VideoStreamWriter:
    """Handle real-time video stream input (RTSP, MJPEG, Webcam)."""
    
    STREAM_TYPES = {
        'rtsp': 'RTSP stream',
        'mjpeg': 'MJPEG stream',
        'webcam': 'Webcam (device)',
        'http': 'HTTP M-JPEG'
    }
    
    @staticmethod
    def validate_stream_url(url: str):
        """Validate stream URL format."""
        url_lower = url.lower()
        
        if url_lower.startswith('rtsp://'):
            return 'rtsp'
        elif url_lower.startswith('http://') and 'mjpeg' in url_lower:
            return 'mjpeg'
        elif url_lower.startswith('http://'):
            return 'http'
        elif url.isdigit():  # Device index like '0'
            return 'webcam'
        else:
            return None
    
    @staticmethod
    def get_stream_info(url: str):
        """Get metadata about a stream (resolution, fps, codec)."""
        import cv2
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return None
        
        info = {
            'url': url,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info
    
    @staticmethod
    def record_stream_snippet(url: str, output_path: str, duration_seconds=30):
        """Record a short snippet from a stream for testing."""
        import cv2
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return False
        
        fps = max(cap.get(cv2.CAP_PROP_FPS), 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_limit = int(fps * duration_seconds)
        frame_count = 0
        
        while frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        return True
