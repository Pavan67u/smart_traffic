"""
Multi-Channel Alert System.
Extends the existing webhook/email alerts with SMS and WhatsApp via Twilio.
"""
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Twilio configuration from environment variables
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '').strip()
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '').strip()
TWILIO_PHONE_FROM = os.environ.get('TWILIO_PHONE_FROM', '').strip()  # e.g., +1234567890
TWILIO_PHONE_TO = os.environ.get('TWILIO_PHONE_TO', '').strip()      # e.g., +0987654321
TWILIO_WHATSAPP_FROM = os.environ.get('TWILIO_WHATSAPP_FROM', '').strip()  # e.g., whatsapp:+14155238886
TWILIO_WHATSAPP_TO = os.environ.get('TWILIO_WHATSAPP_TO', '').strip()      # e.g., whatsapp:+1234567890

# Lazy-loaded Twilio client
_twilio_client = None


def _get_twilio_client():
    """Get or create Twilio client (lazy initialization)."""
    global _twilio_client

    if _twilio_client is not None:
        return _twilio_client

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.debug("Twilio credentials not configured")
        return None

    try:
        from twilio.rest import Client
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialized successfully")
        return _twilio_client
    except ImportError:
        logger.warning("Twilio package not installed. Run: pip install twilio")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Twilio client: {e}")
        return None


def send_sms_alert(message, to_phone=None):
    """
    Send SMS alert via Twilio.

    Args:
        message: Alert message text (max 1600 chars)
        to_phone: Recipient phone number (optional, uses env default)

    Returns:
        dict: Result with 'sent' boolean and details
    """
    client = _get_twilio_client()
    if not client:
        return {'sent': False, 'reason': 'twilio_not_configured'}

    to_phone = to_phone or TWILIO_PHONE_TO
    if not to_phone:
        return {'sent': False, 'reason': 'recipient_phone_not_configured'}

    if not TWILIO_PHONE_FROM:
        return {'sent': False, 'reason': 'sender_phone_not_configured'}

    try:
        # Truncate message to SMS limit
        message = message[:1600]

        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_FROM,
            to=to_phone
        )

        logger.info(f"SMS sent successfully. SID: {msg.sid}")
        return {
            'sent': True,
            'sid': msg.sid,
            'to': to_phone,
            'status': msg.status
        }

    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        return {'sent': False, 'reason': str(e)}


def send_whatsapp_alert(message, to_number=None, media_url=None):
    """
    Send WhatsApp alert via Twilio.

    Args:
        message: Alert message text
        to_number: Recipient WhatsApp number (e.g., whatsapp:+1234567890)
        media_url: Optional URL to image attachment (must be publicly accessible)

    Returns:
        dict: Result with 'sent' boolean and details
    """
    client = _get_twilio_client()
    if not client:
        return {'sent': False, 'reason': 'twilio_not_configured'}

    to_number = to_number or TWILIO_WHATSAPP_TO
    if not to_number:
        return {'sent': False, 'reason': 'recipient_whatsapp_not_configured'}

    # Ensure whatsapp: prefix
    if not to_number.startswith('whatsapp:'):
        to_number = f'whatsapp:{to_number}'

    from_number = TWILIO_WHATSAPP_FROM
    if not from_number:
        return {'sent': False, 'reason': 'sender_whatsapp_not_configured'}

    if not from_number.startswith('whatsapp:'):
        from_number = f'whatsapp:{from_number}'

    try:
        kwargs = {
            'body': message,
            'from_': from_number,
            'to': to_number
        }

        # Add media if provided (must be publicly accessible URL)
        if media_url:
            kwargs['media_url'] = [media_url]

        msg = client.messages.create(**kwargs)

        logger.info(f"WhatsApp message sent. SID: {msg.sid}")
        return {
            'sent': True,
            'sid': msg.sid,
            'to': to_number,
            'status': msg.status
        }

    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")
        return {'sent': False, 'reason': str(e)}


def format_violation_alert(violation):
    """
    Format violation data into a human-readable alert message.

    Args:
        violation: Violation dict with event_type, track_id, timestamp, etc.

    Returns:
        str: Formatted alert message
    """
    event_type = violation.get('event_type', 'unknown')
    event_type_display = event_type.replace('_', ' ').title()

    track_id = violation.get('track_id', 'N/A')
    timestamp = violation.get('timestamp', datetime.now(timezone.utc).isoformat())

    # Parse timestamp for display
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            time_str = timestamp
    else:
        time_str = str(timestamp)

    # Build message
    message = f"""
🚨 TRAFFIC VIOLATION ALERT

Type: {event_type_display}
Vehicle ID: {track_id}
Time: {time_str}
""".strip()

    # Add meta information
    meta = violation.get('meta', {})
    if meta:
        if 'person_count' in meta:
            message += f"\nRiders: {meta['person_count']}"
        if 'speed_px_s' in meta:
            message += f"\nSpeed: {meta['speed_px_s']:.1f} px/s"
        if 'dwell_time_seconds' in meta:
            message += f"\nDwell Time: {meta['dwell_time_seconds']:.1f}s"
        if 'angle_deviation' in meta:
            message += f"\nAngle: {meta['angle_deviation']:.1f}°"

    message += "\n\n📍 View details in Smart Traffic Dashboard"

    return message


def emit_multi_channel_alert(violation, channels=None):
    """
    Send alerts through multiple channels.

    Args:
        violation: Violation event dict
        channels: List of channels to use ['sms', 'whatsapp'] or None for all

    Returns:
        dict: Results from each channel
    """
    if channels is None:
        channels = ['sms', 'whatsapp']

    message = format_violation_alert(violation)
    results = {}

    if 'sms' in channels:
        results['sms'] = send_sms_alert(message)

    if 'whatsapp' in channels:
        # Check for evidence image URL (would need to be publicly accessible)
        media_url = violation.get('_public_image_url')
        results['whatsapp'] = send_whatsapp_alert(message, media_url=media_url)

    return results


def check_twilio_config():
    """
    Check if Twilio is properly configured.

    Returns:
        dict: Configuration status with details
    """
    return {
        'configured': bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN),
        'sms_ready': bool(TWILIO_PHONE_FROM and TWILIO_PHONE_TO),
        'whatsapp_ready': bool(TWILIO_WHATSAPP_FROM and TWILIO_WHATSAPP_TO),
        'account_sid_set': bool(TWILIO_ACCOUNT_SID),
        'auth_token_set': bool(TWILIO_AUTH_TOKEN),
        'phone_from': TWILIO_PHONE_FROM[:4] + '***' if TWILIO_PHONE_FROM else None,
        'phone_to': TWILIO_PHONE_TO[:4] + '***' if TWILIO_PHONE_TO else None,
        'whatsapp_from': TWILIO_WHATSAPP_FROM[:12] + '***' if TWILIO_WHATSAPP_FROM else None,
        'whatsapp_to': TWILIO_WHATSAPP_TO[:12] + '***' if TWILIO_WHATSAPP_TO else None
    }


# Convenience function for testing
def test_sms_alert():
    """Send a test SMS alert."""
    test_violation = {
        'event_type': 'test_alert',
        'track_id': 'TEST-001',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'meta': {}
    }
    return send_sms_alert(format_violation_alert(test_violation))


def test_whatsapp_alert():
    """Send a test WhatsApp alert."""
    test_violation = {
        'event_type': 'test_alert',
        'track_id': 'TEST-001',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'meta': {}
    }
    return send_whatsapp_alert(format_violation_alert(test_violation))
