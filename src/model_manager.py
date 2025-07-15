"""
Model manager for Qwen2-Audio with GPU optimization and error handling
"""
import torch
import librosa
import logging
import gc
from typing import Optional, Tuple
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from contextlib import contextmanager
import numpy as np

from config import (
    MODEL_NAME, DEVICE, MAX_AUDIO_LENGTH, MIXED_PRECISION,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, DO_SAMPLE
)

logger = logging.getLogger(__name__)

class Qwen2AudioManager:
    """Manages Qwen2-Audio model with GPU optimization"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.dtype = torch.float16 if device == "cuda" and MIXED_PRECISION else torch.float32
        
    def load_model(self):
        """Load model with error handling and GPU optimization"""
        try:
            logger.info(f"Loading Qwen2-Audio model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with GPU optimizations
            model_kwargs = {
                "torch_dtype": self.dtype,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            if self.device == "cuda":
                # Enable flash attention if available
                try:
                    model_kwargs["use_flash_attention_2"] = True
                except:
                    logger.warning("Flash Attention 2 not available")
            
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    @contextmanager
    def cuda_amp_context(self):
        """Context manager for automatic mixed precision"""
        if self.device == "cuda" and MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio with error handling"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(
                audio_path, 
                sr=self.processor.feature_extractor.sampling_rate
            )
            
            # Clip to maximum length
            max_samples = int(MAX_AUDIO_LENGTH * sr)
            if len(audio) > max_samples:
                logger.warning(f"Audio clipped from {len(audio)/sr:.2f}s to {MAX_AUDIO_LENGTH}s")
                audio = audio[:max_samples]
            
            # Normalize audio
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def generate_description(self, audio_path: str, prompt: str, 
                           max_retries: int = 3) -> str:
        """Generate description with error handling and retries"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        for attempt in range(max_retries):
            try:
                # Load and preprocess audio
                audio, sr = self.preprocess_audio(audio_path)
                
                # Process inputs
                inputs = self.processor(
                    text=prompt, 
                    audios=audio, 
                    return_tensors="pt"
                )
                
                # Move to device
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate with mixed precision
                with torch.no_grad():
                    with self.cuda_amp_context():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            do_sample=DO_SAMPLE,
                            top_p=TOP_P,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id
                        )
                
                # Decode response
                generated_ids = generated_ids[:, inputs.input_ids.size(1):]
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                return response.strip()
                
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU OOM error. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                if attempt == max_retries - 1:
                    raise
                    
            except Exception as e:
                logger.error(f"Generation error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
    
    def batch_generate(self, audio_paths: list, prompt: str) -> list:
        """Generate descriptions for multiple audio files"""
        results = []
        for path in audio_paths:
            try:
                desc = self.generate_description(path, prompt)
                results.append(desc)
            except Exception as e:
                logger.error(f"Failed to process {path}: {str(e)}")
                results.append(f"Error: {str(e)}")
        return results
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage"""
        if self.device == "cuda":
            return {
                "allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB"
            }
        return {"device": "cpu"}
    
    def cleanup(self):
        """Clean up model and free memory"""
        logger.info("Cleaning up model...")
        
        if self.model:
            del self.model
            self.model = None
            
        if self.processor:
            del self.processor
            self.processor = None
            
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
        
        logger.info("Cleanup complete")