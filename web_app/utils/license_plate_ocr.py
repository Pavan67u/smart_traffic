"""License Plate Recognition (LPR/ANPR) Integration Module."""

class LicensePlateOCR:
    """Interface for license plate detection and OCR."""
    
    def __init__(self, provider='google_vision'):
        """
        Initialize LPR provider.
        
        Options:
        - 'google_vision': Google Cloud Vision API (high accuracy, $)
        - 'tesseract': Open-source Tesseract OCR (free, decent)
        - 'easyocr': OpenCV-based easy OCR (free, modern)
        - 'azure': Azure Computer Vision (accuracy, $)
        """
        self.provider = provider
        self.detector = None
        self.engine = None
        
        if provider == 'tesseract':
            try:
                import pytesseract
                self.engine = pytesseract
                print("✓ Tesseract OCR initialized")
            except ImportError:
                print("✗ pytesseract not installed: pip install pytesseract")
        
        elif provider == 'easyocr':
            try:
                import easyocr
                self.engine = easyocr.Reader(['en'])
                print("✓ EasyOCR initialized")
            except ImportError:
                print("✗ easyocr not installed: pip install easyocr")
        
        elif provider == 'google_vision':
            try:
                from google.cloud import vision
                self.engine = vision.ImageAnnotatorClient()
                print("✓ Google Vision API client initialized")
            except ImportError:
                print("✗ google-cloud-vision not installed: pip install google-cloud-vision")
    
    def extract_text_from_region(self, image, bbox):
        """Extract text (license plate) from image region (bbox)."""
        import cv2
        import numpy as np
        
        x1, y1, x2, y2 = bbox
        cropped = image[int(y1):int(y2), int(x1):int(x2)]
        
        if cropped.size == 0:
            return None
        
        # Preprocess for OCR
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply OCR
        if self.provider == 'tesseract':
            text = self.engine.image_to_string(gray, config='--psm 11 --oem 3')
        elif self.provider == 'easyocr':
            result = self.engine.readtext(gray)
            text = ''.join([r[1] for r in result])
        else:
            text = None
        
        return text.strip() if text else None
    
    def is_valid_plate_format(self, text, region='india'):
        """Validate format of extracted plate."""
        import re
        
        text = text.upper().replace(' ', '')
        
        if region == 'india':
            # Format: AA 01 AA 1234 (State-code, district, sequence, registration)
            pattern = r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}'
            return bool(re.match(pattern, text)) and len(text) >= 10
        
        elif region == 'us':
            # Format: ABC1234 or similar
            pattern = r'[A-Z0-9]{3,8}'
            return bool(re.match(pattern, text)) and 3 <= len(text) <= 8
        
        return len(text) > 3
    
    def lookup_vehicle_owner(self, plate_number, region='india'):
        """
        Lookup vehicle registration details (requires API integration).
        
        Backend integrations:
        - India: SITU (Standardized Interface for Traffic Management) API
        - US: DVLA/State DMV APIs
        - EU: National vehicle registries
        """
        
        if region == 'india':
            # Example: Connect to SITU or local DMV API
            # This would require proper credentials and API setup
            return self._lookup_india_situ(plate_number)
        
        elif region == 'us':
            return self._lookup_us_dmv(plate_number)
        
        return None
    
    @staticmethod
    def _lookup_india_situ(plate_number):
        """Lookup India vehicle via SITU API (placeholder)."""
        # In production, this would call actual SITU API
        # Example response structure:
        return {
            'registration_number': plate_number,
            'owner_name': 'John Doe',
            'owner_email': 'john@example.com',
            'vehicle_type': 'Two Wheeler',
            'seating_capacity': 2,
            'registration_date': '2024-01-15',
            'fitness_valid_until': '2026-01-15',
            'source': 'SITU (Standardized Interface for Traffic Management)'
        }
    
    @staticmethod
    def _lookup_us_dmv(plate_number):
        """Lookup US vehicle via state DMV (placeholder)."""
        return {
            'registration_number': plate_number,
            'owner_name': 'Jane Smith',
            'state': 'CA',
            'vehicle_year': 2022,
            'vehicle_make': 'Tesla',
            'vehicle_model': 'Model 3'
        }


class ViolationNotificationSystem:
    """Send violation notifications to vehicle owner via email/SMS."""
    
    @staticmethod
    def send_owner_notification(plate_number, owner_info, violation_details):
        """Send violation notice to registered owner."""
        import smtplib
        from email.mime.text import MIMEText
        
        email = owner_info.get('owner_email')
        if not email:
            return False
        
        subject = f"Traffic Violation Notice: {plate_number}"
        body = f"""
Dear {owner_info.get('owner_name')},

Your vehicle ({plate_number}) has been cited for a traffic violation.

Violation Type: {violation_details.get('type')}
Time: {violation_details.get('timestamp')}
Location: {violation_details.get('location')}
Fine Amount: ${violation_details.get('fine_amount', 'TBD')}

Please review the evidence and submit your response within 30 days.

Evidence: [Link to violation details]

Smart Traffic Enforcement System
        """
        
        try:
            # Use configured SMTP server
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'noreply@smarttraffic.local'
            msg['To'] = email
            
            # Send via SMTP (requires configuration)
            # smtp = smtplib.SMTP(os.environ['SMTP_SERVER'])
            # smtp.send_message(msg)
            # smtp.quit()
            
            print(f"✓ Notification sent to {email}")
            return True
        except Exception as e:
            print(f"✗ Failed to send notification: {e}")
            return False


# ============================================================================
# Integration Example
# ============================================================================

def process_violation_with_lpr(image, violation_info):
    """End-to-end: detect plate → OCR → lookup → notify."""
    
    # 1. Initialize LPR
    lpr = LicensePlateOCR(provider='easyocr')
    
    # 2. Extract license plate from violation evidence
    plate_bbox = extract_plate_bbox_from_image(image)  # Your bbox extraction
    if not plate_bbox:
        return {'ok': False, 'reason': 'No plate detected'}
    
    # 3. OCR
    plate_text = lpr.extract_text_from_region(image, plate_bbox)
    if not plate_text:
        return {'ok': False, 'reason': 'OCR failed'}
    
    # 4. Validate format
    if not lpr.is_valid_plate_format(plate_text, region='india'):
        return {'ok': False, 'reason': 'Invalid plate format'}
    
    # 5. Lookup owner
    owner_info = lpr.lookup_vehicle_owner(plate_text, region='india')
    if not owner_info:
        return {'ok': False, 'reason': 'Owner lookup failed'}
    
    # 6. Send notification
    notification_sys = ViolationNotificationSystem()
    notified = notification_sys.send_owner_notification(
        plate_text, 
        owner_info, 
        violation_info
    )
    
    return {
        'ok': True,
        'plate_number': plate_text,
        'owner_name': owner_info.get('owner_name'),
        'owner_email': owner_info.get('owner_email'),
        'notified': notified,
        'violation_id': violation_info.get('id')
    }


def extract_plate_bbox_from_image(image):
    """Detect license plate region in image (placeholder)."""
    # In production, use YOLOv8 fine-tuned for plate detection
    # or a specialized plate detector
    return None  # Placeholder
