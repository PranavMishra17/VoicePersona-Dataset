"""
Parse model responses to extract structured metadata
"""
import re
import logging

logger = logging.getLogger(__name__)

def parse_voice_response(response: str) -> dict:
    """Parse voice analysis response to extract metadata"""
    result = {
        'gender': 'unknown',
        'age': 'unknown', 
        'accent': 'unknown',
        'voice_profile': response
    }
    
    try:
        # Extract GENDER
        gender_match = re.search(r'GENDER:\s*([^\n]+)', response, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(1).strip().lower()
            if any(g in gender for g in ['male', 'female', 'man', 'woman']):
                if any(g in gender for g in ['female', 'woman']):
                    result['gender'] = 'female'
                elif any(g in gender for g in ['male', 'man']):
                    result['gender'] = 'male'
                else:
                    result['gender'] = gender
        
        # Extract AGE
        age_match = re.search(r'AGE:\s*([^\n]+)', response, re.IGNORECASE)
        if age_match:
            age = age_match.group(1).strip().lower()
            # Standardize age ranges
            if any(a in age for a in ['teen', '10', '15', '16', '17', '18', '19']):
                result['age'] = 'teens'
            elif any(a in age for a in ['twenties', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']):
                result['age'] = 'twenties'
            elif any(a in age for a in ['thirties', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']):
                result['age'] = 'thirties'
            elif any(a in age for a in ['forties', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49']):
                result['age'] = 'forties'
            elif any(a in age for a in ['fifties', '50', 'older', 'elder']):
                result['age'] = 'fifties+'
            else:
                result['age'] = age
        
        # Extract ACCENT
        accent_match = re.search(r'ACCENT:\s*([^\n]+)', response, re.IGNORECASE)
        if accent_match:
            accent = accent_match.group(1).strip()
            result['accent'] = accent
        
        # Extract VOICE_PROFILE
        profile_match = re.search(r'VOICE_PROFILE:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if profile_match:
            result['voice_profile'] = profile_match.group(1).strip()
        
        logger.debug(f"Parsed metadata: {result}")
        
    except Exception as e:
        logger.warning(f"Failed to parse response metadata: {e}")
    
    return result