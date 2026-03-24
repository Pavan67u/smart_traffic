"""Celery configuration for distributed inference processing."""
import os
from celery import Celery
from kombu import Exchange, Queue

# Define the Celery app
app = Celery('smart_traffic')

# Redis broker configuration
BROKER_URL = os.environ.get('CELERY_BROKER', 'redis://localhost:6379/0')
RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

app.conf.update(
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 min hard limit
    task_soft_time_limit=25 * 60,  # 25 min soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Task queues by priority
default_exchange = Exchange('smart_traffic', type='direct')
app.conf.task_queues = (
    Queue('high_priority', default_exchange, routing_key='high_priority'),
    Queue('inference', default_exchange, routing_key='inference'),
    Queue('postprocessing', default_exchange, routing_key='postprocessing'),
    Queue('default', default_exchange, routing_key='default'),
)

app.conf.task_default_queue = 'default'
app.conf.task_default_routing_key = 'default'


# ============================================================================
# Async Inference Tasks
# ============================================================================

@app.task(queue='inference', bind=True)
def async_inference_task(self, video_path, camera_id='default', model_path=None):
    """
    Async inference task - processes video asynchronously.
    
    Returns:
        dict with run_id, violation_count, output_path
    """
    try:
        # Import here to avoid circular imports
        import json
        from pathlib import Path
        import uuid
        
        # Run inference (simplified)
        run_id = str(uuid.uuid4())[:8]
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100})
        
        # Call main inference function
        from web_app.app import run_inference_on_video
        violations = run_inference_on_video(video_path, camera_id, model_path)
        
        self.update_state(state='PROGRESS', meta={'current': 100, 'total': 100})
        
        return {
            'ok': True,
            'run_id': run_id,
            'violation_count': len(violations),
            'status': 'completed'
        }
    except Exception as exc:
        self.update_state(state='FAILURE', meta={'exc_type': type(exc).__name__, 'exc_message': str(exc)})
        raise


@app.task(queue='postprocessing')
def async_postprocess_violations(run_id, violations):
    """Post-process violations: aggregate stats, send alerts, etc."""
    try:
        from datetime import datetime, timezone
        
        # Generate statistics
        stats = {
            'total': len(violations),
            'by_type': {},
            'high_priority': 0,
            'processed_at': datetime.now(timezone.utc).isoformat()
        }
        
        for v in violations:
            v_type = v.get('violation_type', 'unknown')
            stats['by_type'][v_type] = stats['by_type'].get(v_type, 0) + 1
            if v.get('priority_level') == 'high':
                stats['high_priority'] += 1
        
        return {'ok': True, 'run_id': run_id, 'stats': stats}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}


@app.task(queue='default')
def async_send_alerts(violation_id, alert_type='webhook'):
    """Send alerts for high-priority violations."""
    try:
        from web_app.app import _emit_alerts
        
        # Get violation from DB
        from web_app.models import Violation
        v = Violation.query.get(violation_id)
        
        if v:
            _emit_alerts(f"camera_{v.camera_id or 'default'}", {
                'violation_id': v.id,
                'type': v.violation_type,
                'priority': v.priority_level
            })
        
        return {'ok': True, 'violation_id': violation_id}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}


# ============================================================================
# Scheduled Tasks (Celery Beat)
# ============================================================================

from celery.schedules import crontab

app.conf.beat_schedule = {
    'cleanup-old-violations-daily': {
        'task': 'worker_tasks.cleanup_old_violations',
        'schedule': crontab(hour=2, minute=0),  # 2 AM UTC daily
    },
    'generate-analytics-hourly': {
        'task': 'worker_tasks.generate_analytics',
        'schedule': 3600,  # Every hour
    },
    'retrain-model-weekly': {
        'task': 'worker_tasks.retrain_model_if_ready',
        'schedule': crontab(day_of_week=0, hour=3, minute=0),  # Sundays at 3 AM
    },
}
