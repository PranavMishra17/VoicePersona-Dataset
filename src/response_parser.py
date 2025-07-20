"""
Parse model responses to extract structured metadata
"""
import re
import logging

logger = logging.getLogger(__name__)

def parse_voice_response(response: str, dataset_name: str) -> dict:
    """Parse voice analysis response to extract metadata"""
    result = {
        'gender': 'unknown',
        'age': 'unknown', 
        'accent': 'unknown',
        'voice_profile': response
    }
    
    # For GLOBE_V2, we already have metadata, just return the response as voice_profile
    if dataset_name == 'globe_v2':
        result['voice_profile'] = response.strip()
        return result
    
    # For other datasets, parse structured response
    try:
        # Extract GENDER
        gender_match = re.search(r'GENDER:\s*([^\n]+)', response, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(1).strip().lower()
            if 'female' in gender or 'woman' in gender:
                result['gender'] = 'female'
            elif 'male' in gender or 'man' in gender:
                result['gender'] = 'male'
            else:
                result['gender'] = gender.strip()
        
        # Extract AGE
        age_match = re.search(r'AGE:\s*([^\n]+)', response, re.IGNORECASE)
        if age_match:
            age = age_match.group(1).strip().lower()
            # Standardize age ranges
            if any(a in age for a in ['teen', '10', '15', '16', '17', '18', '19']):
                result['age'] = 'teens'
            elif any(a in age for a in ['twenties', '20']):
                result['age'] = 'twenties'
            elif any(a in age for a in ['thirties', '30']):
                result['age'] = 'thirties'
            elif any(a in age for a in ['forties', '40']):
                result['age'] = 'forties'
            elif any(a in age for a in ['fifties', '50', 'older', 'elder']):
                result['age'] = 'fifties+'
            else:
                result['age'] = age.strip()
        
        # Extract ACCENT
        accent_match = re.search(r'ACCENT:\s*([^\n]+)', response, re.IGNORECASE)
        if accent_match:
            accent = accent_match.group(1).strip()
            # Clean up common variations
            if 'neutral' in accent.lower() or 'unknown' in accent.lower():
                result['accent'] = 'General American'  # Default assumption
            else:
                result['accent'] = accent
        
        # Extract VOICE_PROFILE
        profile_match = re.search(r'VOICE_PROFILE:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if profile_match:
            result['voice_profile'] = profile_match.group(1).strip()
        else:
            # If no structured format found, clean the response
            cleaned = re.sub(r'(GENDER|AGE|ACCENT):\s*[^\n]*\n?', '', response, flags=re.IGNORECASE)
            result['voice_profile'] = cleaned.strip()
        
        logger.debug(f"Parsed metadata for {dataset_name}: {result}")
        
    except Exception as e:
        logger.warning(f"Failed to parse response metadata for {dataset_name}: {e}")
        # Fallback: use entire response as voice_profile
        result['voice_profile'] = response.strip()
    
    return result