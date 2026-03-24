from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class Violation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    violation_type = db.Column(db.String(50), nullable=False)  # e.g., 'stop_line'
    image_path = db.Column(db.String(200), nullable=True)      # Path to evidence crop
    vehicle_type = db.Column(db.String(50), nullable=True)     # e.g., 'car', 'bus'
    track_id = db.Column(db.Integer, nullable=True)            # Track ID for consistency check
    confidence = db.Column(db.Float, nullable=True)            # Detection/Rule confidence
    status = db.Column(db.String(20), default='New')           # New, Reviewed, Sent

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'violation_type': self.violation_type,
            'image_path': self.image_path,
            'vehicle_type': self.vehicle_type,
            'status': self.status,
            'track_id': self.track_id,
            'confidence': self.confidence
        }
