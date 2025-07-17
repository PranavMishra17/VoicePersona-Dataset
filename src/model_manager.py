"""
Lightweight model manager with multiple backend options and quantization fallback
"""
import torch
import librosa
import logging
import gc
import json
from typing import Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AudioAnalyzerBase(ABC):
    """Base class for audio analyzers"""
    
    @abstractmethod
    def analyze(self, audio_path: str, prompt: str) -> str:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

class Qwen2AudioManager(AudioAnalyzerBase):
    """Qwen2-Audio manager with quantization fallback"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.quantization_used = False
        
    def load_model(self):
        """Load model with fallback to quantization if needed"""
        try:
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            
            # Load processor first (always works)
            logger.info("Loading Qwen2-Audio processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_NAME,
                cache_dir=self.config.CACHE_DIR
            )
            logger.info("Processor loaded successfully!")
            
            # Try to load model without quantization first
            if not self.config.USE_QUANTIZATION:
                logger.info("Attempting to load model without quantization...")
                try:
                    self.model = self._load_model_normal()
                    logger.info("Model loaded successfully without quantization!")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load without quantization: {str(e)}")
                    if self.config.ENABLE_QUANTIZATION_FALLBACK:
                        logger.info("Falling back to quantization...")
                    else:
                        raise e
            
            # If quantization is enabled or fallback is needed
            if self.config.USE_QUANTIZATION or self.config.ENABLE_QUANTIZATION_FALLBACK:
                logger.info("Loading model with quantization...")
                self.model = self._load_model_quantized()
                self.quantization_used = True
                logger.info("Model loaded successfully with quantization!")
                
        except ImportError as e:
            if "bitsandbytes" in str(e):
                logger.error("bitsandbytes not installed. For quantization support, run: pip install bitsandbytes")
                # Try loading without quantization
                if not self.config.USE_QUANTIZATION:
                    logger.info("Attempting to load without quantization...")
                    self.model = self._load_model_normal()
                else:
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_model_normal(self):
        """Load model without quantization"""
        from transformers import Qwen2AudioForConditionalGeneration
        
        return Qwen2AudioForConditionalGeneration.from_pretrained(
            self.config.MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
            cache_dir=self.config.CACHE_DIR,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    def _load_model_quantized(self):
        """Load model with quantization (fallback method)"""
        from transformers import Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
        
        # Determine quantization type
        use_4bit = self.config.LOAD_IN_4BIT or (
            self.config.ENABLE_QUANTIZATION_FALLBACK and self.config.FALLBACK_4BIT
        )
        use_8bit = self.config.LOAD_IN_8BIT or (
            not use_4bit and self.config.ENABLE_QUANTIZATION_FALLBACK and self.config.FALLBACK_8BIT
        )
        
        if not (use_4bit or use_8bit):
            use_4bit = True  # Default fallback
            
        logger.info(f"Using {'4-bit' if use_4bit else '8-bit'} quantization")
        
        # Quantization config
        quant_kwargs = {
            "load_in_4bit": use_4bit,
            "load_in_8bit": use_8bit and not use_4bit,
        }
        
        if use_4bit:
            quant_kwargs.update({
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "llm_int8_enable_fp32_cpu_offload": False  # Keep on GPU only
            })
        
        quantization_config = BitsAndBytesConfig(**quant_kwargs)
        
        return Qwen2AudioForConditionalGeneration.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": 0},  # Keep everything on GPU 0
            torch_dtype=torch.float16,
            cache_dir=self.config.CACHE_DIR,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    def analyze(self, audio_path: str, prompt: str) -> str:
        """Analyze audio with the model"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Clip to max length
            max_samples = int(self.config.MAX_AUDIO_LENGTH * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Process with explicit sampling rate
            inputs = self.processor(
                text=prompt, 
                audios=audio, 
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Move to device
            if self.config.DEVICE == "cuda":
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                if self.config.MIXED_PRECISION and self.config.DEVICE == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.MAX_NEW_TOKENS,
                            temperature=self.config.TEMPERATURE,
                            do_sample=self.config.DO_SAMPLE,
                            top_p=self.config.TOP_P,
                            pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        temperature=self.config.TEMPERATURE,
                        do_sample=self.config.DO_SAMPLE,
                        top_p=self.config.TOP_P,
                        pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                    )
            
            # Decode - fix the input_ids access
            input_length = inputs['input_ids'].size(1) if 'input_ids' in inputs else 0
            response = self.processor.batch_decode(
                outputs[:, input_length:],
                skip_special_tokens=True
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_description(self, audio_path: str, prompt: str) -> str:
        """Generate description (alias for analyze method)"""
        return self.analyze(audio_path, prompt)
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        gc.collect()
        if self.config.DEVICE == "cuda":
            torch.cuda.empty_cache()

class WhisperAnalyzer(AudioAnalyzerBase):
    """Use Whisper for transcription + LLM for analysis"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load Whisper model"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            logger.info("Loading Whisper model...")
            
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-base",  # Use base model (74M params)
                cache_dir=self.config.CACHE_DIR
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-base",
                cache_dir=self.config.CACHE_DIR,
                torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.config.DEVICE == "cuda":
                self.model = self.model.cuda()
            
            logger.info("Whisper model loaded!")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper: {str(e)}")
            raise
    
    def analyze(self, audio_path: str, prompt: str) -> str:
        """Transcribe and analyze"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Transcribe
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            if self.config.DEVICE == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs["input_features"])
            
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Create basic analysis based on audio features
            analysis = self._analyze_audio_features(audio, sr, transcription)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Whisper analysis failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_description(self, audio_path: str, prompt: str) -> str:
        """Generate description (alias for analyze method)"""
        return self.analyze(audio_path, prompt)
    
    def _analyze_audio_features(self, audio: np.ndarray, sr: int, 
                               transcription: str) -> str:
        """Basic audio feature analysis"""
        # Calculate basic features
        duration = len(audio) / sr
        
        # Estimate pitch (very basic)
        try:
            f0 = np.mean(librosa.yin(audio, fmin=50, fmax=400))
        except:
            f0 = 150  # Default
        
        # Energy
        energy = np.mean(librosa.feature.rms(y=audio))
        
        # Speaking rate (words per minute)
        word_count = len(transcription.split())
        speaking_rate = (word_count / duration) * 60 if duration > 0 else 0
        
        # Create description
        pitch_desc = "high-pitched" if f0 > 180 else "low-pitched" if f0 < 120 else "medium-pitched"
        energy_desc = "energetic" if energy > 0.1 else "calm" if energy < 0.05 else "moderate"
        rate_desc = "fast" if speaking_rate > 180 else "slow" if speaking_rate < 120 else "moderate"
        
        analysis = f"""Voice Analysis:
The speaker has a {pitch_desc} voice with {energy_desc} delivery. 
Speaking rate is {rate_desc} at approximately {speaking_rate:.0f} words per minute.
Duration: {duration:.1f} seconds.

Transcript: {transcription}

Character/Persona:
Based on the vocal characteristics, this voice would suit a {self._suggest_character(pitch_desc, energy_desc, rate_desc)}.
"""
        return analysis
    
    def _suggest_character(self, pitch: str, energy: str, rate: str) -> str:
        """Suggest character based on features"""
        suggestions = {
            ("high-pitched", "energetic", "fast"): "young, enthusiastic character or excited narrator",
            ("low-pitched", "calm", "slow"): "wise elder, meditation guide, or authoritative figure",
            ("medium-pitched", "moderate", "moderate"): "professional presenter, teacher, or everyday character",
        }
        
        key = (pitch, energy, rate)
        return suggestions.get(key, "versatile character suitable for various roles")
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        gc.collect()
        if self.config.DEVICE == "cuda":
            torch.cuda.empty_cache()

class APIAnalyzer(AudioAnalyzerBase):
    """Use API services for analysis"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.API_KEY
        
    def load_model(self):
        """Initialize API client"""
        if not self.api_key:
            raise ValueError("API_KEY not set in environment variables")
        logger.info(f"Using {self.config.API_PROVIDER} API for analysis")
    
    def analyze(self, audio_path: str, prompt: str) -> str:
        """Analyze using API"""
        if self.config.API_PROVIDER == "openai":
            return self._analyze_openai(audio_path, prompt)
        else:
            return "API analysis not implemented"
    
    def generate_description(self, audio_path: str, prompt: str) -> str:
        """Generate description (alias for analyze method)"""
        return self.analyze(audio_path, prompt)
    
    def _analyze_openai(self, audio_path: str, prompt: str) -> str:
        """Use OpenAI API"""
        try:
            import openai
            openai.api_key = self.api_key
            
            # Transcribe with Whisper API
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            
            # Analyze with GPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert voice analyst."},
                    {"role": "user", "content": f"Based on this transcript, provide a voice analysis: {transcript.text}"}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API analysis failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """No cleanup needed for API"""
        pass

def create_analyzer(config) -> AudioAnalyzerBase:
    """Factory function to create appropriate analyzer"""
    if config.USE_API:
        return APIAnalyzer(config)
    elif config.USE_ALTERNATIVE_MODEL:
        return WhisperAnalyzer(config)
    else:
        return Qwen2AudioManager(config)