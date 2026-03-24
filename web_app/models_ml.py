"""Active learning pipeline for continuous model improvement."""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json

db = SQLAlchemy()

class LabelingTask(db.Model):
    """Uncertain detections flagged for human labeling."""
    id = db.Column(db.Integer, primary_key=True)
    violation_id = db.Column(db.Integer, db.ForeignKey('violation.id'), nullable=True)
    image_path = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)  # Detection confidence (0.5-0.7 "uncertain" range)
    predicted_label = db.Column(db.String(50), nullable=False)  # Model's prediction
    human_label = db.Column(db.String(50), nullable=True)  # Officer's correction (if labeling done)
    is_correct = db.Column(db.Boolean, nullable=True)  # True if human agrees with model
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    labeled_at = db.Column(db.DateTime, nullable=True)
    labeled_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, labeled, imported_to_training
    batch_id = db.Column(db.String(50), nullable=True, index=True)  # Group labels by retraining batch
    
    def to_dict(self):
        return {
            'id': self.id,
            'violation_id': self.violation_id,
            'image_path': self.image_path,
            'confidence': self.confidence,
            'predicted_label': self.predicted_label,
            'human_label': self.human_label,
            'is_correct': self.is_correct,
            'created_at': self.created_at.isoformat(),
            'labeled_at': self.labeled_at.isoformat() if self.labeled_at else None,
            'status': self.status,
            'batch_id': self.batch_id
        }


class TrainingBatch(db.Model):
    """Tracks retraining events with labeled data."""
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    status = db.Column(db.String(20), default='collecting')  # collecting, ready, training, completed
    labeled_count = db.Column(db.Integer, default=0)
    accuracy_threshold = db.Column(db.Float, default=0.8)  # Min accuracy to trigger retraining
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime, nullable=True)
    model_version = db.Column(db.String(50), nullable=True)  # e.g., 'v1.2.3_retrained_april_2026'
    metrics = db.Column(db.Text, nullable=True)  # JSON with precision/recall/f1
    
    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'status': self.status,
            'labeled_count': self.labeled_count,
            'accuracy_threshold': self.accuracy_threshold,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'model_version': self.model_version,
            'metrics': json.loads(self.metrics) if self.metrics else None
        }
