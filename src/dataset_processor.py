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
            logger.debug(f"Processing sample {idx}: {sample.get('speaker_id', 'unknown')}")
            
            # Extract audio data
            if isinstance(sample['audio'], dict):
                audio_array = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
            else:
                audio_array = sample['audio']
                sample_rate = 16000  # Default
            
            # Save to temp file
            temp_path = self.save_audio_temp(audio_array, sample_rate)
            
            # Generate descriptions
            voice_analysis = self.model_manager.generate_description(
                temp_path, 
                VoiceDescriptionPrompts.get_analysis_prompt()
            )
            
            character_description = self.model_manager.generate_description(
                temp_path,
                VoiceDescriptionPrompts.get_character_prompt()
            )
            
            # Compile result
            result = {
                'index': idx,
                'transcript': sample.get('transcript', ''),
                'speaker_id': sample.get('speaker_id', ''),
                'accent': sample.get('accent', ''),
                'age': sample.get('age', ''),
                'gender': sample.get('gender', ''),
                'duration': sample.get('duration', 0.0),
                'voice_analysis': voice_analysis,
                'character_description': character_description,
                'combined_description': f"Voice Analysis:\n{voice_analysis}\n\nCharacter/Persona:\n{character_description}",
                'processing_timestamp': datetime.now().isoformat()
            }
            
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
        """Process streaming dataset"""
        # Skip to start_idx for streaming dataset
        dataset_iter = iter(self.dataset)
        
        # Skip samples before start_idx
        for _ in range(start_idx):
            try:
                next(dataset_iter)
            except StopIteration:
                break
        
        # Process samples
        idx = start_idx
        samples_processed = 0
        
        with tqdm(desc="Processing") as pbar:
            for sample in dataset_iter:
                # Check if we've reached end_idx
                if end_idx and idx >= end_idx:
                    break
                
                # Skip if already processed
                if idx in self.checkpoint_data['processed_indices']:
                    idx += 1
                    continue
                
                # Process sample
                result = self.process_single_sample(sample, idx)
                
                if result:
                    processed_data.append(result)
                    self.checkpoint_data['processed_indices'].add(idx)
                    self.checkpoint_data['last_index'] = idx
                    samples_processed += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'idx': idx,
                        'speaker': result.get('speaker_id', 'unknown')[:10],
                        'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if DEVICE == "cuda" else "CPU"
                    })
                    pbar.update(1)
                
                # Periodic operations
                if (samples_processed + 1) % CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()
                
                if DEVICE == "cuda" and (samples_processed + 1) % CLEAR_CACHE_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                idx += 1
                
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
        
        # Calculate average description lengths
        analysis_lengths = [len(item['voice_analysis']) for item in data]
        character_lengths = [len(item['character_description']) for item in data]
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total samples: {total}")
        logger.info(f"Average voice analysis length: {np.mean(analysis_lengths):.0f} chars")
        logger.info(f"Average character description length: {np.mean(character_lengths):.0f} chars")
        
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