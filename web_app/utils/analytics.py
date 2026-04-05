"""Analytics, metrics, and monitoring utilities."""
import json
from datetime import datetime, timezone, timedelta
from flask import request
from web_app.models import Violation, User, AuditLog, db
from sqlalchemy import func, and_, or_

# ============================================================================
# Real-time Analytics
# ============================================================================

def get_analytics_summary(days=7):
    """Get comprehensive analytics for dashboard."""
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)
    
    violations = Violation.query.filter(Violation.timestamp >= start_date).all()
    total_violations = len(violations)
    
    # Group by violation type
    by_type = {}
    by_vehicle = {}
    by_priority = {'high': 0, 'medium': 0, 'low': 0}
    by_status = {}
    
    for v in violations:
        by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        by_vehicle[v.vehicle_type or 'unknown'] = by_vehicle.get(v.vehicle_type, 0) + 1
        by_priority[v.priority_level] += 1
        by_status[v.status] = by_status.get(v.status, 0) + 1
    
    # Hourly distribution
    hourly = {}
    for v in violations:
        hour = v.timestamp.strftime('%Y-%m-%d %H:00')
        hourly[hour] = hourly.get(hour, 0) + 1
    
    return {
        'total': total_violations,
        'by_type': by_type,
        'by_vehicle': by_vehicle,
        'by_priority': by_priority,
        'by_status': by_status,
        'hourly': dict(sorted(hourly.items())),
        'days': days
    }


def get_camera_health():
    """Get health metrics for each camera."""
    violations_by_camera = db.session.query(
        Violation.camera_id,
        func.count(Violation.id).label('count'),
        func.avg(Violation.confidence).label('avg_confidence'),
        func.max(Violation.timestamp).label('last_violation')
    ).group_by(Violation.camera_id).all()
    
    cameras = {}
    for cam_id, count, avg_conf, last_ts in violations_by_camera:
        time_since_last = None
        if last_ts:
            # SQLite can return naive datetimes; normalize to UTC-aware before subtracting.
            if getattr(last_ts, 'tzinfo', None) is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_ts
            time_since_last = delta.total_seconds()
        
        cameras[cam_id or 'default'] = {
            'violation_count': count,
            'avg_confidence': float(avg_conf or 0.0),
            'last_violation': last_ts.isoformat() if last_ts else None,
            'seconds_ago': time_since_last,
            'status': 'online' if (time_since_last is not None and time_since_last < 300) else 'offline'
        }
    
    return cameras


def get_heatmap_data(limit_days=7):
    """Get violation heatmap by time of day and day of week."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=limit_days)
    
    violations = Violation.query.filter(Violation.timestamp >= start).all()
    
    # Create heatmap: hour vs day_of_week
    heatmap = {}
    for v in violations:
        hour = v.timestamp.hour
        dow = v.timestamp.weekday()  # 0=Mon, 6=Sun
        key = f"{dow}_{hour:02d}"  # e.g., "1_14" = Tuesday 14:00
        heatmap[key] = heatmap.get(key, 0) + 1
    
    return heatmap


# ============================================================================
# Advanced Filtering
# ============================================================================

def build_filter_query(filters):
    """Build SQLAlchemy query from filter parameters."""
    query = Violation.query
    
    # Date range
    if filters.get('start_date'):
        try:
            start = datetime.fromisoformat(filters['start_date']).replace(tzinfo=timezone.utc)
            query = query.filter(Violation.timestamp >= start)
        except ValueError:
            pass
    
    if filters.get('end_date'):
        try:
            end = datetime.fromisoformat(filters['end_date']).replace(tzinfo=timezone.utc)
            # Add 24 hours to include entire day
            end = end.replace(hour=23, minute=59, second=59) + timedelta(seconds=1)
            query = query.filter(Violation.timestamp <= end)
        except ValueError:
            pass
    
    # Violation type
    if filters.get('violation_type'):
        query = query.filter(Violation.violation_type == filters['violation_type'])
    
    # Vehicle type
    if filters.get('vehicle_type'):
        query = query.filter(Violation.vehicle_type == filters['vehicle_type'])
    
    # Status
    if filters.get('status'):
        query = query.filter(Violation.status == filters['status'])
    
    # Priority
    if filters.get('priority_level'):
        query = query.filter(Violation.priority_level == filters['priority_level'])
    
    # Camera
    if filters.get('camera_id'):
        query = query.filter(Violation.camera_id == filters['camera_id'])
    
    # Priority score range
    if filters.get('min_priority_score'):
        query = query.filter(Violation.priority_score >= int(filters['min_priority_score']))
    
    if filters.get('max_priority_score'):
        query = query.filter(Violation.priority_score <= int(filters['max_priority_score']))
    
    # Confidence range
    if filters.get('min_confidence'):
        query = query.filter(Violation.confidence >= float(filters['min_confidence']))
    
    if filters.get('max_confidence'):
        query = query.filter(Violation.confidence <= float(filters['max_confidence']))
    
    # Text search
    if filters.get('search_text'):
        search = f"%{filters['search_text']}%"
        query = query.filter(
            or_(
                Violation.vehicle_type.ilike(search),
                Violation.violation_type.ilike(search)
            )
        )
    
    return query


# ============================================================================
# Data Retention & Privacy
# ============================================================================

def cleanup_old_violations(days=90):
    """Delete violations older than specified days. Run as scheduled task."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted = Violation.query.filter(Violation.timestamp < cutoff).delete()
    db.session.commit()
    return deleted


def blur_license_plates_in_image(image_path):
    """GDPR: Blur license plates in violation image."""
    # TODO: Implement YOLO-based license plate detection and blur
    # For now, this is a placeholder
    pass


def export_user_data(user_id):
    """GDPR: Export all user-related data as JSON."""
    user = User.query.get(user_id)
    if not user:
        return None
    
    # Get all audit logs for this user
    logs = AuditLog.query.filter_by(user_id=user_id).all()
    
    # Get violations created/updated by this user (via audit logs)
    violation_ids = set()
    for log in logs:
        if log.resource_type == 'violation':
            violation_ids.add(log.resource_id)
    
    violations = Violation.query.filter(Violation.id.in_(violation_ids)).all() if violation_ids else []
    
    return {
        'user': user.to_dict(),
        'audit_logs': [log.to_dict() for log in logs],
        'violations_modified': [v.to_dict() for v in violations],
        'export_timestamp': datetime.now(timezone.utc).isoformat()
    }


def delete_user_data(user_id):
    """GDPR: Delete all user data (except audit trail for compliance)."""
    user = User.query.get(user_id)
    if not user:
        return None
    
    # Soft delete: mark as inactive
    user.is_active = False
    db.session.commit()
    
    return True


# ============================================================================
# Predictive Analytics
# ============================================================================

def get_violation_trends(days=30):
    """Predict high-violation times based on historical data."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    
    violations = Violation.query.filter(
        Violation.timestamp >= start
    ).all()
    
    # Group by hour of day across all days
    hourly_trends = {}
    daily_trends = {}
    
    for v in violations:
        hour = v.timestamp.hour
        day_of_week = v.timestamp.strftime('%A')
        
        hourly_trends[hour] = hourly_trends.get(hour, 0) + 1
        daily_trends[day_of_week] = daily_trends.get(day_of_week, 0) + 1
    
    # Find peak hours
    peak_hours = sorted(hourly_trends.items(), key=lambda x: x[1], reverse=True)[:3]
    peak_days = sorted(daily_trends.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        'peak_hours': [{'hour': h, 'count': c} for h, c in peak_hours],
        'peak_days': [{'day': d, 'count': c} for d, c in peak_days],
        'hourly_distribution': hourly_trends,
        'daily_distribution': daily_trends
    }


def predict_resource_allocation(days=7):
    """Suggest where to deploy enforcement (high-violation intersections)."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    
    camera_stats = db.session.query(
        Violation.camera_id,
        func.count(Violation.id).label('count'),
        func.avg(Violation.priority_score).label('avg_priority')
    ).filter(
        Violation.timestamp >= start
    ).group_by(Violation.camera_id).order_by(
        func.count(Violation.id).desc()
    ).all()
    
    recommendations = []
    for cam_id, count, avg_priority in camera_stats:
        if count >= 5:  # At least 5 violations to recommend
            recommendations.append({
                'camera': cam_id or 'default',
                'violation_count': count,
                'avg_priority_score': float(avg_priority or 0.0),
                'recommendation': 'HIGH' if count >= 20 else 'MEDIUM'
            })
    
    return sorted(recommendations, key=lambda x: x['violation_count'], reverse=True)
