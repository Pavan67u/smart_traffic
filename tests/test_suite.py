"""Comprehensive test suite for Smart Traffic system."""
import pytest
import json
from datetime import datetime, timezone, timedelta
from web_app.app import app, db
from web_app.models import Violation, User, AuditLog


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


@pytest.fixture
def admin_user():
    """Create admin user for testing."""
    user = User(
        username='admin_test',
        email='admin@test.local',
        role='admin',
        is_active=True
    )
    user.set_password('testpass')
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def officer_user():
    """Create officer user for testing."""
    user = User(
        username='officer_test',
        email='officer@test.local',
        role='officer',
        is_active=True
    )
    user.set_password('testpass')
    db.session.add(user)
    db.session.commit()
    return user


class TestAuthentication:
    """Test RBAC and authentication."""
    
    def test_login_success(self, client, admin_user):
        """Test successful login."""
        response = client.post('/login', data={
            'username': 'admin_test',
            'password': 'testpass'
        }, follow_redirects=True)
        assert response.status_code == 200
    
    def test_login_failure_wrong_password(self, client, admin_user):
        """Test login with wrong password."""
        response = client.post('/login', data={
            'username': 'admin_test',
            'password': 'wrongpass'
        })
        assert b'Invalid credentials' in response.data
    
    def test_login_failure_user_not_found(self, client):
        """Test login with non-existent user."""
        response = client.post('/login', data={
            'username': 'nonexistent',
            'password': 'password'
        })
        assert b'Invalid credentials' in response.data
    
    def test_logout(self, client, admin_user):
        """Test user logout."""
        # Login first
        client.post('/login', data={
            'username': 'admin_test',
            'password': 'testpass'
        })
        
        # Logout
        response = client.get('/logout', follow_redirects=True)
        assert response.status_code == 200
        assert b'login' in response.data.lower()
    
    def test_password_hash(self):
        """Test password hashing."""
        user = User(username='test', email='test@test.local')
        user.set_password('password123')
        
        assert user.password_hash != 'password123'
        assert user.check_password('password123')
        assert not user.check_password('wrongpassword')


class TestRBAC:
    """Test role-based access control."""
    
    def test_admin_permissions(self, admin_user):
        """Test admin has all permissions."""
        assert admin_user.has_permission('view_violations')
        assert admin_user.has_permission('update_status')
        assert admin_user.has_permission('export_data')
        assert admin_user.has_permission('manage_profiles')
        assert admin_user.has_permission('any_permission')
    
    def test_officer_permissions(self, officer_user):
        """Test officer limited permissions."""
        assert officer_user.has_permission('view_violations')
        assert officer_user.has_permission('update_status')
        assert officer_user.has_permission('export_data')
        assert officer_user.has_permission('manage_profiles')
        assert not officer_user.has_permission('delete_users')
    
    def test_viewer_permissions(self):
        """Test viewer read-only permissions."""
        viewer = User(username='viewer', email='viewer@test.local', role='viewer')
        viewer.set_password('pass')
        db.session.add(viewer)
        db.session.commit()
        
        assert viewer.has_permission('view_violations')
        assert not viewer.has_permission('update_status')
        assert not viewer.has_permission('export_data')


class TestViolationModel:
    """Test Violation model and database queries."""
    
    def test_create_violation(self):
        """Test creating a violation."""
        v = Violation(
            violation_type='red_light',
            vehicle_type='car',
            confidence=0.95,
            priority_score=75,
            priority_level='high',
            status='New'
        )
        db.session.add(v)
        db.session.commit()
        
        assert v.id is not None
        assert v.violation_type == 'red_light'
        assert v.priority_level == 'high'
    
    def test_violation_indexing(self):
        """Test indexed columns for query performance."""
        # Create multiple violations
        for i in range(10):
            v = Violation(
                violation_type='red_light' if i % 2 == 0 else 'lane_cross',
                vehicle_type='car' if i % 2 == 0 else 'bus',
                confidence=0.5 + (i * 0.05),
                priority_score=50 + i,
                priority_level='high' if i > 6 else 'medium'
            )
            db.session.add(v)
        db.session.commit()
        
        # Query by indexed column (should be fast)
        result = Violation.query.filter_by(violation_type='red_light').all()
        assert len(result) == 5
        
        # Query by priority_level
        high_priority = Violation.query.filter_by(priority_level='high').all()
        assert len(high_priority) == 3
    
    def test_violation_to_dict(self):
        """Test violation serialization."""
        v = Violation(
            violation_type='red_light',
            vehicle_type='car',
            confidence=0.95,
            priority_score=80,
            priority_level='high'
        )
        db.session.add(v)
        db.session.commit()
        
        d = v.to_dict()
        assert d['violation_type'] == 'red_light'
        assert d['vehicle_type'] == 'car'
        assert d['priority_score'] == 80


class TestAuditLog:
    """Test audit logging."""
    
    def test_create_audit_log(self, admin_user):
        """Test creating audit log."""
        log = AuditLog(
            user_id=admin_user.id,
            action='update_violation_status',
            resource_type='violation',
            resource_id=1,
            details=json.dumps({'old_status': 'New', 'new_status': 'Reviewed'}),
            ip_address='127.0.0.1'
        )
        db.session.add(log)
        db.session.commit()
        
        assert log.id is not None
        assert log.action == 'update_violation_status'
        assert log.user_id == admin_user.id
    
    def test_audit_log_serialization(self, admin_user):
        """Test audit log to_dict."""
        log = AuditLog(
            user_id=admin_user.id,
            action='test_action',
            resource_type='test_resource',
            resource_id=999
        )
        db.session.add(log)
        db.session.commit()
        
        d = log.to_dict()
        assert d['action'] == 'test_action'
        assert d['resource_type'] == 'test_resource'
        assert d['user_id'] == admin_user.id


class TestAnalytics:
    """Test analytics functions."""
    
    def test_analytics_summary(self):
        """Test analytics summary generation."""
        from web_app.utils.analytics import get_analytics_summary
        
        # Create test data
        now = datetime.now(timezone.utc)
        for i in range(10):
            v = Violation(
                violation_type='red_light' if i % 2 == 0 else 'lane_cross',
                vehicle_type='car',
                priority_level='high' if i > 5 else 'medium',
                status='New' if i < 3 else 'Reviewed',
                timestamp=now - timedelta(hours=i)
            )
            db.session.add(v)
        db.session.commit()
        
        summary = get_analytics_summary(days=1)
        assert summary['total'] == 10
        assert len(summary['by_type']) > 0
        assert 'high' in summary['by_priority']
    
    def test_camera_health(self):
        """Test camera health metrics."""
        from web_app.utils.analytics import get_camera_health
        
        # Create violations for different cameras
        for cam in ['cam1', 'cam2', 'cam1']:
            v = Violation(
                camera_id=cam,
                violation_type='red_light',
                confidence=0.9 if cam == 'cam1' else 0.7
            )
            db.session.add(v)
        db.session.commit()
        
        health = get_camera_health()
        assert 'cam1' in health or 'cam2' in health
    
    def test_advanced_filter_query(self):
        """Test advanced filtering."""
        from web_app.utils.analytics import build_filter_query
        
        # Create test violations
        now = datetime.now(timezone.utc)
        for i in range(5):
            v = Violation(
                violation_type='red_light',
                vehicle_type='car' if i < 3 else 'bus',
                priority_score=60 + i * 5,
                timestamp=now - timedelta(days=i)
            )
            db.session.add(v)
        db.session.commit()
        
        # Test filter by vehicle type
        filters = {'vehicle_type': 'car'}
        query = build_filter_query(filters)
        result = query.all()
        assert len(result) == 3
        assert all(v.vehicle_type == 'car' for v in result)
        
        # Test filter by priority score
        filters = {'min_priority_score': 70}
        query = build_filter_query(filters)
        result = query.all()
        assert all(v.priority_score >= 70 for v in result)


class TestAPI:
    """Test API endpoints."""
    
    def test_api_requires_login(self, client):
        """Test that API requires authentication."""
        response = client.get('/api/analytics/summary')
        assert response.status_code == 401
    
    def test_analytics_api_with_auth(self, client, admin_user):
        """Test analytics API with proper auth."""
        # Create session
        client.post('/login', data={
            'username': 'admin_test',
            'password': 'testpass'
        })
        
        # Create some test data
        v = Violation(
            violation_type='red_light',
            vehicle_type='car',
            priority_level='high',
            priority_score=80
        )
        db.session.add(v)
        db.session.commit()
        
        # Test analytics endpoint
        response = client.get('/api/analytics/summary')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
