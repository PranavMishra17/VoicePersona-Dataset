"""
Specific dataset processors for different voice datasets
"""
import logging
import numpy as np
from typing import Dict, Any
from datasets import load_dataset
from base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)

class GlobeV2Processor(BaseDatasetProcessor):
    """GLOBE_V2 dataset processor with diversity sampling"""
    
    def load_dataset(self, split: str = "train"):
        """Load GLOBE_V2 dataset"""
        try:
            if self.config.USE_STREAMING:
                logger.info("Loading GLOBE_V2 dataset in streaming mode...")
                self.dataset = load_dataset("MushanW/GLOBE_V2", split=split, streaming=True)
            else:
                split_str = f"train[:{self.config.SUBSET_SIZE}]" if self.config.USE_SUBSET else split
                logger.info(f"Loading GLOBE_V2 dataset (split: {split_str})...")
                self.dataset = load_dataset("MushanW/GLOBE_V2", split=split_str)
            return True
        except Exception as e:
            logger.error(f"Failed to load GLOBE_V2: {str(e)}")
            raise
    
    def extract_sample_data(self, sample, idx: int) -> Dict[str, Any]:
        """Extract data from GLOBE_V2 sample"""
        # Handle audio data
        audio_data = sample['audio']
        if hasattr(audio_data, 'decode'):
            decoded = audio_data.decode()
            audio_array = decoded['array']
            sample_rate = decoded['sampling_rate']
        elif isinstance(audio_data, dict):
            audio_array = audio_data['array']
            sample_rate = audio_data['sampling_rate']
        else:
            audio_array = audio_data
            sample_rate = 16000
        
        return {
            'speaker_id': sample.get('speaker_id', f'globe_speaker_{idx}'),
            'transcript': sample.get('transcript', ''),
            'audio_array': audio_array,
            'sample_rate': sample_rate,
            'duration': sample.get('duration', len(audio_array) / sample_rate),
            'gender': sample.get('gender', 'unknown'),
            'age': sample.get('age', 'unknown'),
            'accent': sample.get('accent', 'unknown')
        }
    
    def get_analysis_prompt(self, sample_data: Dict[str, Any]) -> str:
        """Get analysis prompt for GLOBE_V2"""
        metadata = f"Speaker: {sample_data['gender']} in {sample_data['age']}, {sample_data['accent']} accent"
        
        return f"""Analyze this voice recording for character consistency purposes. {metadata}.

Provide a detailed voice profile covering:

**Vocal Qualities:** Pitch (high/low/medium), tone (warm/cold/bright/dark), timbre (rich/thin/raspy/smooth), resonance (nasal/throaty/chest/head voice), breathiness, clarity

**Speaking Style:** Pace (fast/slow/measured), rhythm, articulation (crisp/relaxed/mumbled), emphasis patterns, pauses, vocal fry, uptalk

**Emotional Undertones:** Confidence level, warmth, authority, friendliness, energy level, mood, approachability

**Character Impression:** What personality traits does this voice convey? What type of character would this voice suit? Professional roles? Social settings?

**Distinctive Features:** Unique vocal quirks, speech patterns, memorable qualities that define this voice

Focus on HOW they speak rather than what they say. Describe the voice in vivid, specific terms that would help someone recreate or recognize this vocal character."""

class LaionsProcessor(BaseDatasetProcessor):
    """Laions Got Talent dataset processor"""
    
    def load_dataset(self, split: str = "train"):
        """Load Laions dataset"""
        try:
            logger.info("Loading Laions Got Talent dataset...")
            self.dataset = load_dataset("laion/laions_got_talent", split=split, streaming=True)
            return True
        except Exception as e:
            logger.error(f"Failed to load Laions: {str(e)}")
            raise
    
    def extract_sample_data(self, sample, idx: int) -> Dict[str, Any]:
        """Extract data from Laions sample"""
        # Audio handling
        audio_data = sample['wav']
        if hasattr(audio_data, 'decode'):
            decoded = audio_data.decode()
            audio_array = decoded['array']
            sample_rate = decoded['sampling_rate']
        elif isinstance(audio_data, dict):
            audio_array = audio_data['array']
            sample_rate = audio_data['sampling_rate']
        else:
            audio_array = audio_data
            sample_rate = 24000  # Default from dataset info
        
        # Extract metadata from json field
        metadata = sample.get('json', {})
        emotion = metadata.get('emotion', 'unknown')
        sample_id = metadata.get('sample_id', f'laions_{idx}')
        
        return {
            'speaker_id': sample_id,
            'transcript': metadata.get('text', ''),
            'audio_array': audio_array,
            'sample_rate': sample_rate,
            'duration': metadata.get('duration', len(audio_array) / sample_rate),
            'emotion': emotion
        }
    
    def get_analysis_prompt(self, sample_data: Dict[str, Any]) -> str:
        """Get analysis prompt for Laions dataset"""
        emotion_context = f"Emotional context: {sample_data.get('emotion', 'unknown')}"
        
        return f"""Analyze this voice recording for character consistency. {emotion_context}.

Provide a comprehensive voice profile including:

**Vocal Qualities:** Pitch, tone, timbre, resonance, breathiness, clarity
**Speaking Style:** Pace, rhythm, articulation, emphasis patterns
**Emotional Expression:** How the emotion is conveyed through vocal qualities
**Character Traits:** Personality conveyed through voice, suitable character types
**Demographics:** Estimate gender (male/female/other), age range (teens/twenties/thirties/forties/fifties+), accent if detectable

Format your response to include:
GENDER: [estimated gender]
AGE: [estimated age range] 
ACCENT: [detected accent or 'neutral/unknown']
VOICE_PROFILE: [detailed voice description focusing on characteristics for character consistency]

Focus on vocal delivery rather than content."""

class AnimeVoxProcessor(BaseDatasetProcessor):
    """AnimeVox dataset processor"""
    
    def load_dataset(self, split: str = "train"):
        """Load AnimeVox dataset"""
        try:
            logger.info("Loading AnimeVox dataset...")
            self.dataset = load_dataset("taresh18/AnimeVox", split=split, streaming=True)
            return True
        except Exception as e:
            logger.error(f"Failed to load AnimeVox: {str(e)}")
            raise
    
    def extract_sample_data(self, sample, idx: int) -> Dict[str, Any]:
        """Extract data from AnimeVox sample"""
        # Audio handling
        audio_data = sample['audio']
        if hasattr(audio_data, 'decode'):
            decoded = audio_data.decode()
            audio_array = decoded['array']
            sample_rate = decoded['sampling_rate']
        elif isinstance(audio_data, dict):
            audio_array = audio_data['array']
            sample_rate = audio_data['sampling_rate']
        else:
            audio_array = audio_data
            sample_rate = 16000
        
        character_name = sample.get('character_name', f'anime_char_{idx}')
        anime_name = sample.get('anime', 'unknown')
        
        return {
            'speaker_id': f"{anime_name}_{character_name}",
            'transcript': sample.get('transcription', ''),
            'audio_array': audio_array,
            'sample_rate': sample_rate,
            'duration': len(audio_array) / sample_rate,
            'character_name': character_name,
            'anime': anime_name
        }
    
    def get_analysis_prompt(self, sample_data: Dict[str, Any]) -> str:
        """Get analysis prompt for AnimeVox"""
        character_context = f"Character: {sample_data.get('character_name', 'unknown')} from {sample_data.get('anime', 'unknown')}"
        
        return f"""Analyze this anime voice recording for character consistency. {character_context}.

Provide a comprehensive voice profile including:

**Vocal Qualities:** Pitch, tone, timbre, resonance, vocal style typical of anime characters
**Speaking Style:** Pace, rhythm, articulation, anime-specific vocal patterns
**Character Expression:** How personality is conveyed through vocal delivery
**Voice Acting Style:** Performance characteristics, emotional range
**Demographics:** Estimate gender (male/female/other), age range (teens/twenties/thirties/forties+), accent if detectable

Format your response to include:
GENDER: [estimated gender]
AGE: [estimated age range]
ACCENT: [detected accent or 'Japanese/anime style']
VOICE_PROFILE: [detailed voice description focusing on anime character voice consistency]

Focus on vocal performance and character voice traits."""

class AniSpeechProcessor(BaseDatasetProcessor):
    """AniSpeech dataset processor"""
    
    def load_dataset(self, split: str = "train"):
        """Load AniSpeech dataset"""
        try:
            logger.info("Loading AniSpeech dataset...")
            self.dataset = load_dataset("ShoukanLabs/AniSpeech", split=split, streaming=True)
            return True
        except Exception as e:
            logger.error(f"Failed to load AniSpeech: {str(e)}")
            raise
    
    def extract_sample_data(self, sample, idx: int) -> Dict[str, Any]:
        """Extract data from AniSpeech sample"""
        # Audio handling
        audio_data = sample['audio']
        if hasattr(audio_data, 'decode'):
            decoded = audio_data.decode()
            audio_array = decoded['array']
            sample_rate = decoded['sampling_rate']
        elif isinstance(audio_data, dict):
            audio_array = audio_data['array']
            sample_rate = audio_data['sampling_rate']
        else:
            audio_array = audio_data
            sample_rate = 16000
        
        voice_class = sample.get('voice', 'unknown')
        
        return {
            'speaker_id': f"anispeech_{voice_class}_{idx}",
            'transcript': sample.get('caption', ''),
            'audio_array': audio_array,
            'sample_rate': sample_rate,
            'duration': len(audio_array) / sample_rate,
            'voice_class': voice_class
        }
    
    def get_analysis_prompt(self, sample_data: Dict[str, Any]) -> str:
        """Get analysis prompt for AniSpeech"""
        voice_context = f"Voice class: {sample_data.get('voice_class', 'unknown')}"
        
        return f"""Analyze this anime speech recording for character consistency. {voice_context}.

Provide a comprehensive voice profile including:

**Vocal Qualities:** Pitch, tone, timbre, resonance, vocal characteristics
**Speaking Style:** Pace, rhythm, articulation, speech patterns
**Voice Classification:** How the voice fits its designated class/category
**Character Suitability:** What type of anime characters this voice would suit
**Demographics:** Estimate gender (male/female/other), age range (teens/twenties/thirties/forties+), accent characteristics

Format your response to include:
GENDER: [estimated gender]
AGE: [estimated age range]
ACCENT: [detected accent or 'anime/Japanese style']
VOICE_PROFILE: [detailed voice description for character consistency across anime productions]

Focus on voice characteristics that define this speaker's unique vocal identity."""

# Dataset registry
DATASET_PROCESSORS = {
    'globe_v2': GlobeV2Processor,
    'laions': LaionsProcessor,
    'animevox': AnimeVoxProcessor,
    'anispeech': AniSpeechProcessor
}

def get_processor(dataset_name: str, config):
    """Factory function to get appropriate processor"""
    if dataset_name not in DATASET_PROCESSORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_PROCESSORS.keys())}")
    
    return DATASET_PROCESSORS[dataset_name](config, dataset_name)