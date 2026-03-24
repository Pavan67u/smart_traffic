"""Authentication and authorization utilities."""
from functools import wraps
from flask import request, jsonify, session
from web_app.models import User, AuditLog, db
from datetime import datetime, timezone
import json

def audit_action(action, resource_type=None, resource_id=None, details=None):
    """Record an audit log entry."""
    try:
        user_id = session.get('user_id')
        ip_address = request.remote_addr
        
        audit= AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=json.dumps(details) if details else None,
            ip_address=ip_address
        )
        db.session.add(audit)
        db.session.commit()
    except Exception as e:
        print(f"Audit log error: {e}")


def require_login(f):
    """Decorator to require user to be logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({'error': 'Not authenticated'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function


def require_permission(permission):
    """Decorator to check user permission."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            user = User.query.get(session['user_id'])
            if not user:
                session.clear()
                return jsonify({'error': 'User not found'}), 401
            
            if not user.has_permission(permission):
                audit_action('permission_denied', resource_type='action', details={'permission': permission})
                return jsonify({'error': 'Permission denied'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_role(role):
    """Decorator to check user role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401
            
            user = User.query.get(session['user_id'])
            if not user or user.role != role:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def init_default_users(app):
    """Initialize default admin user if none exists."""
    with app.app_context():
        # Check if any admin user exists
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            # Create default admin
            default_admin = User(
                username='admin',
                email='admin@smarttraffic.local',
                role='admin',
                is_active=True
            )
            default_admin.set_password('admin123')  # Change in production!
            db.session.add(default_admin)
            db.session.commit()
            print("✓ Created default admin user (admin/admin123)")
