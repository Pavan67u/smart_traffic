from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    """User model with role-based access control."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='viewer')  # admin, officer, viewer
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime, nullable=True)
    
    def set_password(self, password):
        # Use pbkdf2:sha256 for Python 3.9 compatibility (scrypt not available)
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def has_permission(self, action):
        """Check if user has permission for action."""
        # Admin has all permissions
        if self.role == 'admin':
            return True
        # Officer can review/update violations, manage profiles
        if self.role == 'officer':
            return action in ['view_violations', 'update_status', 'export_data', 'manage_profiles']
        # Viewer can only view
        if self.role == 'viewer':
            return action == 'view_violations'
        return False
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class AuditLog(db.Model):
    """Audit trail for all user actions."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    action = db.Column(db.String(100), nullable=False, index=True)  # e.g., 'update_violation_status'
    resource_type = db.Column(db.String(50), index=True)  # e.g., 'violation', 'camera_profile'
    resource_id = db.Column(db.Integer, index=True)
    details = db.Column(db.Text)  # JSON with old/new values
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    ip_address = db.Column(db.String(45), nullable=True)  # IPv4 or IPv6
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'ip_address': self.ip_address
        }


class Violation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    violation_type = db.Column(db.String(50), nullable=False, index=True)  # e.g., 'stop_line'
    image_path = db.Column(db.String(200), nullable=True)      # Path to evidence crop
    vehicle_type = db.Column(db.String(50), nullable=True, index=True)     # e.g., 'car', 'bus'
    track_id = db.Column(db.Integer, nullable=True)            # Track ID for consistency check
    confidence = db.Column(db.Float, nullable=True)            # Detection/Rule confidence
    status = db.Column(db.String(20), default='New', index=True)           # New, Reviewed, Sent
    priority_score = db.Column(db.Integer, default=50)         # 0-100 priority
    priority_level = db.Column(db.String(10), default='medium') # high, medium, low
    camera_id = db.Column(db.String(50), nullable=True, index=True)        # Which camera

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'violation_type': self.violation_type,
            'image_path': self.image_path,
            'vehicle_type': self.vehicle_type,
            'status': self.status,
            'track_id': self.track_id,
            'confidence': self.confidence,
            'priority_score': self.priority_score,
            'priority_level': self.priority_level,
            'camera_id': self.camera_id
        }
