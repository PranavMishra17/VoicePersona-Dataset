"""
Dataset processor for GLOBE_V2 with checkpointing and error handling
"""
import os
import json
import pickle
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
import torch
import gc
import traceback
from datetime import datetime

import config
from config import (
    DATASET_NAME, CHECKPOINT_FILE, CHECKPOINT_INTERVAL,
    OUTPUT_JSON, OUTPUT_HF_DATASET, CLEAR_CACHE_INTERVAL,
    DATA_DIR, DEVICE
)
from model_manager import Qwen2AudioManager
from prompts import VoiceDescriptionPrompts

logger = logging.getLogger(__name__)

class GlobeV2Processor:
    """Processes GLOBE_V2 dataset with GPU optimization"""

    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.model_manager = None
        self.checkpoint_data = {
            'processed_indices': set(),
            'processed_data': [],
            'last_index': -1,
            'start_time': None,
            'total_time': 0
        }
        self.temp_audio_path = config.DATA_DIR / "temp_audio.wav"
        
    def load_dataset(self, split: str = "train"):
        """Load GLOBE_V2 dataset with streaming option"""
        try:
            if hasattr(self.config, 'USE_STREAMING') and self.config.USE_STREAMING:
                logger.info(f"Loading GLOBE_V2 dataset in streaming mode...")
                self.dataset = load_dataset(
                    self.config.DATASET_NAME,
                    split=split,
                    streaming=True
                )
            else:
                # For non-streaming, use slicing if subset is enabled
                if hasattr(self.config, 'USE_SUBSET') and self.config.USE_SUBSET:
                    split_str = f"train[:{self.config.SUBSET_SIZE}]"
                else:
                    split_str = split
                logger.info(f"Loading GLOBE_V2 dataset (split: {split_str})...")
                self.dataset = load_dataset(self.config.DATASET_NAME, split=split_str)
                logger.info(f"Dataset loaded! Total samples: {len(self.dataset)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        if CHECKPOINT_FILE.exists():
            try:
                logger.info("Loading checkpoint...")
                with open(CHECKPOINT_FILE, 'rb') as f:
                    self.checkpoint_data = pickle.load(f)
                
                processed_count = len(self.checkpoint_data['processed_indices'])
                logger.info(f"Checkpoint loaded. Processed {processed_count} samples")
                logger.info(f"Last index: {self.checkpoint_data['last_index']}")
                
                if self.checkpoint_data.get('total_time', 0) > 0:
                    avg_time = self.checkpoint_data['total_time'] / processed_count
                    logger.info(f"Average processing time: {avg_time:.2f}s per sample")
                
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                return False
        return False
    
    def save_checkpoint(self):
        """Save checkpoint with error handling"""
        try:
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Update timing info
            if hasattr(self, 'processing_start_time'):
                elapsed = (datetime.now() - self.processing_start_time).total_seconds()
                self.checkpoint_data['total_time'] = elapsed
            
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(self.checkpoint_data, f)
            
            processed_count = len(self.checkpoint_data['processed_indices'])
            logger.info(f"Checkpoint saved. Processed {processed_count} samples")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def save_audio_temp(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Save audio to temporary file"""
        try:
            self.temp_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.temp_audio_path), audio_array, sample_rate)
            return str(self.temp_audio_path)
        except Exception as e:
            logger.error(f"Failed to save temp audio: {str(e)}")
            raise
    
    def cleanup_temp_audio(self):
        """Remove temporary audio file"""
        if self.temp_audio_path.exists():
            try:
                os.remove(self.temp_audio_path)
            except:
                pass
    
    def process_single_sample(self, sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Process a single sample with comprehensive error handling"""
        try:
            logger.info(f"=== PROCESSING SAMPLE {idx} ===")
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Speaker ID: {sample.get('speaker_id', 'unknown')}")
            logger.info(f"Transcript: {sample.get('transcript', '')}")
            logger.info(f"Accent: {sample.get('accent', '')}")
            logger.info(f"Age: {sample.get('age', '')}")
            logger.info(f"Gender: {sample.get('gender', '')}")
            logger.info(f"Duration: {sample.get('duration', 0.0)}")
            
            # Extract audio data - handle multiple formats
            audio_data = sample['audio']
            logger.info(f"Audio data type: {type(audio_data)}")
            
            try:
                if hasattr(audio_data, 'decode'):
                    # AudioDecoder object (newer datasets)
                    logger.info("Decoding AudioDecoder object...")
                    decoded = audio_data.decode()
                    audio_array = decoded['array']
                    sample_rate = decoded['sampling_rate']
                elif isinstance(audio_data, dict):
                    # Dict format (older datasets)
                    logger.info(f"Audio dict keys: {list(audio_data.keys())}")
                    audio_array = audio_data['array']
                    sample_rate = audio_data['sampling_rate']
                else:
                    # Direct array
                    audio_array = audio_data
                    sample_rate = 16000
                
                logger.info(f"Final audio - type: {type(audio_array)}, sample_rate: {sample_rate}")
                if hasattr(audio_array, 'shape'):
                    logger.info(f"Audio shape: {audio_array.shape}")
                
            except Exception as e:
                logger.error(f"Audio extraction failed: {e}")
                return None
            
            # Save to temp file
            logger.info(f"Saving audio to temp file: {self.temp_audio_path}")
            temp_path = self.save_audio_temp(audio_array, sample_rate)
            logger.info(f"Audio saved successfully")
            
            # Generate comprehensive voice description using dataset metadata
            logger.info("Generating comprehensive voice description...")
            
            # Create comprehensive prompt with metadata
            metadata_context = f"Speaker: {sample.get('gender', 'unknown')} in {sample.get('age', 'unknown')}, {sample.get('accent', 'unknown')} accent"
            
            comprehensive_prompt = f"""Analyze this voice recording for character consistency purposes. {metadata_context}.

Provide a detailed voice profile covering:

**Vocal Qualities:** Pitch (high/low/medium), tone (warm/cold/bright/dark), timbre (rich/thin/raspy/smooth), resonance (nasal/throaty/chest/head voice), breathiness, clarity

**Speaking Style:** Pace (fast/slow/measured), rhythm, articulation (crisp/relaxed/mumbled), emphasis patterns, pauses, vocal fry, uptalk

**Emotional Undertones:** Confidence level, warmth, authority, friendliness, energy level, mood, approachability

**Character Impression:** What personality traits does this voice convey? What type of character would this voice suit? Professional roles? Social settings?

**Distinctive Features:** Unique vocal quirks, speech patterns, memorable qualities that define this voice

Focus on HOW they speak rather than what they say. Describe the voice in vivid, specific terms that would help someone recreate or recognize this vocal character."""
            
            voice_description = self.model_manager.generate_description(temp_path, comprehensive_prompt)
            logger.info(f"Voice description result length: {len(voice_description)}")
            
            # Compile result with single comprehensive description
            result = {
                'index': idx,
                'transcript': sample.get('transcript', ''),
                'speaker_id': sample.get('speaker_id', ''),
                'accent': sample.get('accent', ''),
                'age': sample.get('age', ''),
                'gender': sample.get('gender', ''),
                'duration': sample.get('duration', 0.0),
                'voice_description': voice_description,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"=== SAMPLE {idx} COMPLETE ===")
            
            # Cleanup
            self.cleanup_temp_audio()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            self.cleanup_temp_audio()
            return None
    
    def process_dataset(self, start_idx: int = 0, end_idx: Optional[int] = None,
                       resume: bool = True):
        """Process dataset with GPU optimization and checkpointing"""
        # Initialize model
        self.model_manager = Qwen2AudioManager(self.config)
        self.model_manager.load_model()
        
        # Load checkpoint if resuming
        if resume and self.load_checkpoint():
            start_idx = max(start_idx, self.checkpoint_data['last_index'] + 1)
            processed_data = self.checkpoint_data['processed_data']
        else:
            processed_data = []
            self.checkpoint_data['processed_data'] = processed_data
        
        logger.info(f"Processing samples from {start_idx} to {end_idx if end_idx else 'end'}")
        self.processing_start_time = datetime.now()
        
        # Handle streaming vs non-streaming datasets
        if hasattr(self.config, 'USE_STREAMING') and self.config.USE_STREAMING:
            processed_data = self._process_streaming_dataset(start_idx, end_idx, processed_data)
        else:
            processed_data = self._process_regular_dataset(start_idx, end_idx, processed_data)
        
        # Final save
        self.save_checkpoint()
        self.model_manager.cleanup()
        
        logger.info(f"Processing complete! Total processed: {len(processed_data)}")
        return processed_data
    
    def _process_streaming_dataset(self, start_idx: int, end_idx: Optional[int], processed_data: List):
        """Process streaming dataset with diversity sampling"""
        # Track accent+gender combinations for diversity
        combination_counts = {}
        target_samples = end_idx - start_idx if end_idx else 1000
        
        # Calculate target per combination (flexible)
        max_combinations = 104  # 52 accents * 2 genders
        samples_per_combination = max(1, target_samples // max_combinations)
        
        logger.info(f"Diversity sampling: targeting {samples_per_combination} samples per accent+gender combination")
        
        dataset_iter = iter(self.dataset)
        idx = 0
        samples_processed = 0
        samples_examined = 0
        
        with tqdm(desc="Processing (diversity sampling)") as pbar:
            for sample in dataset_iter:
                samples_examined += 1
                
                # Check if we've collected enough samples
                if samples_processed >= target_samples:
                    break
                
                # Skip if already processed (checkpoint recovery)
                if idx in self.checkpoint_data['processed_indices']:
                    idx += 1
                    continue
                
                # Get combination key
                gender = sample.get('gender', 'unknown')
                accent = sample.get('accent', 'unknown')
                
                # Focus on male/female for diversity
                if gender not in ['male', 'female']:
                    idx += 1
                    continue
                
                combination_key = f"{accent}+{gender}"
                current_count = combination_counts.get(combination_key, 0)
                
                # Decide whether to include this sample
                should_include = False
                
                if current_count < samples_per_combination:
                    # Always include if under target for this combination
                    should_include = True
                elif samples_processed < target_samples:
                    # If some combinations are full but we need more samples,
                    # include with lower probability
                    import random
                    should_include = random.random() < 0.3
                
                if should_include:
                    # Process sample
                    result = self.process_single_sample(sample, idx)
                    
                    if result:
                        processed_data.append(result)
                        self.checkpoint_data['processed_indices'].add(idx)
                        self.checkpoint_data['last_index'] = idx
                        samples_processed += 1
                        combination_counts[combination_key] = current_count + 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'processed': samples_processed,
                            'combinations': len(combination_counts),
                            'examined': samples_examined,
                            'speaker': result.get('speaker_id', 'unknown')[:10],
                        })
                        pbar.update(1)
                    
                    # Periodic operations
                    if samples_processed % CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint()
                    
                    if DEVICE == "cuda" and samples_processed % CLEAR_CACHE_INTERVAL == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                idx += 1
                
                # Safety break if we've examined too many samples
                if samples_examined > target_samples * 10:
                    logger.warning(f"Examined {samples_examined} samples, stopping to avoid infinite loop")
                    break
        
        # Log diversity statistics
        logger.info(f"\nDiversity Statistics:")
        logger.info(f"Total combinations found: {len(combination_counts)}")
        logger.info(f"Samples per combination range: {min(combination_counts.values()) if combination_counts else 0}-{max(combination_counts.values()) if combination_counts else 0}")
        
        # Show top combinations
        sorted_combinations = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for combo, count in sorted_combinations:
            logger.info(f"  {combo}: {count} samples")
                
        return processed_data
    
    def _process_regular_dataset(self, start_idx: int, end_idx: Optional[int], processed_data: List):
        """Process regular (non-streaming) dataset"""
        # Set end index
        if end_idx is None:
            end_idx = len(self.dataset)
        
        # Process with progress bar
        with tqdm(range(start_idx, end_idx), desc="Processing") as pbar:
            for idx in pbar:
                # Skip if already processed
                if idx in self.checkpoint_data['processed_indices']:
                    continue
                
                # Get sample
                sample = self.dataset[idx]
                
                # Process sample
                result = self.process_single_sample(sample, idx)
                
                if result:
                    processed_data.append(result)
                    self.checkpoint_data['processed_indices'].add(idx)
                    self.checkpoint_data['last_index'] = idx
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'speaker': result.get('speaker_id', 'unknown')[:10],
                        'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if DEVICE == "cuda" else "CPU"
                    })
                
                # Periodic operations
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()
                
                if DEVICE == "cuda" and (idx + 1) % CLEAR_CACHE_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return processed_data
    
    def save_results(self, processed_data: Optional[List[Dict]] = None):
        """Save processed dataset in multiple formats"""
        if processed_data is None:
            processed_data = self.checkpoint_data['processed_data']
        
        if not processed_data:
            logger.warning("No data to save")
            return
        
        try:
            # Save as JSON
            OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved JSON: {OUTPUT_JSON}")
            
            # Create HuggingFace dataset (without audio arrays)
            hf_data = []
            for item in processed_data:
                hf_item = {k: v for k, v in item.items() if k != 'audio'}
                hf_data.append(hf_item)
            
            hf_dataset = Dataset.from_list(hf_data)
            
            # Save HuggingFace dataset
            OUTPUT_HF_DATASET.parent.mkdir(parents=True, exist_ok=True)
            hf_dataset.save_to_disk(str(OUTPUT_HF_DATASET))
            logger.info(f"Saved HuggingFace dataset: {OUTPUT_HF_DATASET}")
            
            # Log statistics
            self._log_statistics(processed_data)
            
            return hf_dataset
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def _log_statistics(self, data: List[Dict]):
        """Log dataset statistics"""
        total = len(data)
        
        # Calculate average description length
        description_lengths = [len(item['voice_description']) for item in data]
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total samples: {total}")
        logger.info(f"Average voice description length: {np.mean(description_lengths):.0f} chars")
        
        # Gender distribution
        gender_counts = {}
        for item in data:
            gender = item.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        logger.info(f"Gender distribution: {gender_counts}")
        
        # Age distribution
        age_counts = {}
        for item in data:
            age = item.get('age', 'unknown')
            age_counts[age] = age_counts.get(age, 0) + 1
        logger.info(f"Age distribution: {age_counts}")

def test_processing(config, n_samples=2):
    """Run a small test processing loop for n_samples."""
    import logging
    logger = logging.getLogger(__name__)
    processor = GlobeV2Processor(config)
    
    try:
        # Load dataset
        processor.load_dataset(split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset for test: {str(e)}")
        return 0

    # Process only n_samples
    processed = processor.process_dataset(start_idx=0, end_idx=n_samples, resume=False)
    processor.save_results(processed)
    logger.info(f"Test processing complete. Processed {len(processed)} samples.")
    return len(processed)