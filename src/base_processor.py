"""
Base dataset processor with modular architecture
"""
import os
import json
import pickle
import logging
import soundfile as sf
import numpy as np
import shutil
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
import torch
import gc
import traceback
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

from model_manager import Qwen2AudioManager

logger = logging.getLogger(__name__)

class BaseDatasetProcessor(ABC):
    """Base class for dataset processors"""
    
    def __init__(self, config, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.dataset = None
        self.model_manager = None
        
        # Dataset-specific paths
        self.output_dir = config.OUTPUT_DIR / dataset_name
        self.audio_dir = self.output_dir / "audio"
        self.checkpoint_file = config.CHECKPOINT_DIR / f"checkpoint_{dataset_name}.pkl"
        self.output_json = self.output_dir / f"{dataset_name}_descriptions.json"
        self.output_hf = self.output_dir / f"{dataset_name}_hf_dataset"
        
        # Create directories
        for dir_path in [self.output_dir, self.audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_data = {
            'processed_indices': set(),
            'processed_data': [],
            'last_index': -1,
            'start_time': None,
            'total_time': 0
        }
        
    @abstractmethod
    def load_dataset(self, split: str = "train"):
        """Load dataset - implemented by each processor"""
        pass
    
    @abstractmethod
    def extract_sample_data(self, sample, idx: int) -> Dict[str, Any]:
        """Extract standardized data from sample - implemented by each processor"""
        pass
    
    @abstractmethod
    def get_analysis_prompt(self, sample_data: Dict[str, Any]) -> str:
        """Get analysis prompt for this dataset type"""
        pass
    
    def save_audio_file(self, audio_array: np.ndarray, sample_rate: int, 
                       sample_id: str) -> str:
        """Save audio file and return path"""
        try:
            audio_filename = f"{sample_id}.wav"
            audio_path = self.audio_dir / audio_filename
            sf.write(str(audio_path), audio_array, sample_rate)
            return str(audio_path)
        except Exception as e:
            logger.error(f"Failed to save audio {sample_id}: {str(e)}")
            raise
    
    def process_single_sample(self, sample, idx: int) -> Optional[Dict[str, Any]]:
        """Process a single sample"""
        try:
            logger.info(f"=== PROCESSING {self.dataset_name.upper()} SAMPLE {idx} ===")
            
            # Extract sample data using dataset-specific method
            sample_data = self.extract_sample_data(sample, idx)
            logger.info(f"Extracted data: {sample_data['speaker_id']}")
            
            # Save audio file
            audio_path = self.save_audio_file(
                sample_data['audio_array'], 
                sample_data['sample_rate'],
                f"{self.dataset_name}_{idx}_{sample_data['speaker_id']}"
            )
            
            # Get analysis prompt
            prompt = self.get_analysis_prompt(sample_data)
            
            # Generate description
            logger.info("Generating voice description...")
            voice_description = self.model_manager.generate_description(audio_path, prompt)
            logger.info(f"Description length: {len(voice_description)}")
            
            # Parse response for missing metadata (new datasets only)
            if self.dataset_name != 'globe_v2':
                from response_parser import parse_voice_response
                parsed = parse_voice_response(voice_description, self.dataset_name)
                
                # Use parsed metadata if not available in sample_data
                for field in ['gender', 'age', 'accent']:
                    if field not in sample_data or sample_data[field] == 'unknown':
                        sample_data[field] = parsed[field]
                
                # Use cleaned voice profile
                voice_description = parsed['voice_profile']
            
            # Compile result
            result = {
                'index': idx,
                'dataset': self.dataset_name,
                'speaker_id': sample_data['speaker_id'],
                'transcript': sample_data['transcript'],
                'audio_path': audio_path,
                'duration': sample_data['duration'],
                'voice_description': voice_description,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields if available
            for field in ['gender', 'age', 'accent']:
                if field in sample_data:
                    result[field] = sample_data[field]
            
            logger.info(f"=== SAMPLE {idx} COMPLETE ===")
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                logger.info("Loading checkpoint...")
                with open(self.checkpoint_file, 'rb') as f:
                    self.checkpoint_data = pickle.load(f)
                
                processed_count = len(self.checkpoint_data['processed_indices'])
                logger.info(f"Checkpoint loaded. Processed {processed_count} samples")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                return False
        return False
    
    def save_checkpoint(self):
        """Save checkpoint"""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            if hasattr(self, 'processing_start_time'):
                elapsed = (datetime.now() - self.processing_start_time).total_seconds()
                self.checkpoint_data['total_time'] = elapsed
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.checkpoint_data, f)
            
            processed_count = len(self.checkpoint_data['processed_indices'])
            logger.info(f"Checkpoint saved. Processed {processed_count} samples")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def process_dataset(self, max_samples: int = None, resume: bool = True):
        """Process dataset with checkpointing"""
        # Initialize model
        if not self.model_manager:
            self.model_manager = Qwen2AudioManager(self.config)
            self.model_manager.load_model()
        
        # Load checkpoint if resuming
        if resume and self.load_checkpoint():
            processed_data = self.checkpoint_data['processed_data']
        else:
            processed_data = []
            self.checkpoint_data['processed_data'] = processed_data
        
        logger.info(f"Processing {self.dataset_name} dataset (max: {max_samples or 'unlimited'})")
        self.processing_start_time = datetime.now()
        
        # Process samples
        processed_data = self._process_samples(max_samples, processed_data)
        
        # Final save
        self.save_checkpoint()
        self.model_manager.cleanup()
        
        logger.info(f"Processing complete! Total processed: {len(processed_data)}")
        return processed_data
    
    def _process_samples(self, max_samples: Optional[int], processed_data: List):
        """Process samples with random sampling"""
        dataset_iter = iter(self.dataset)
        samples_processed = len(processed_data)
        samples_examined = 0
        
        with tqdm(desc=f"Processing {self.dataset_name}") as pbar:
            for sample in dataset_iter:
                if max_samples and samples_processed >= max_samples:
                    break
                
                samples_examined += 1
                
                # Random sampling (skip some samples)
                if max_samples and samples_examined > 1:
                    import random
                    skip_probability = max(0, 1 - (max_samples / (samples_examined * 2)))
                    if random.random() < skip_probability:
                        continue
                
                # Check if already processed
                if samples_examined in self.checkpoint_data['processed_indices']:
                    continue
                
                # Process sample
                result = self.process_single_sample(sample, samples_examined)
                
                if result:
                    processed_data.append(result)
                    self.checkpoint_data['processed_indices'].add(samples_examined)
                    self.checkpoint_data['last_index'] = samples_examined
                    samples_processed += 1
                    
                    pbar.set_postfix({
                        'processed': samples_processed,
                        'examined': samples_examined,
                        'speaker': result.get('speaker_id', 'unknown')[:10],
                    })
                    pbar.update(1)
                
                # Periodic operations
                if samples_processed % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()
                
                if self.config.DEVICE == "cuda" and samples_processed % self.config.CLEAR_CACHE_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return processed_data
    
    def save_results(self, processed_data: Optional[List[Dict]] = None):
        """Save results in multiple formats"""
        if processed_data is None:
            processed_data = self.checkpoint_data['processed_data']
        
        if not processed_data:
            logger.warning("No data to save")
            return
        
        try:
            # Save as JSON
            with open(self.output_json, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved JSON: {self.output_json}")
            
            # Create HuggingFace dataset
            hf_data = []
            for item in processed_data:
                hf_item = {k: v for k, v in item.items() if k != 'audio_array'}
                hf_data.append(hf_item)
            
            hf_dataset = Dataset.from_list(hf_data)
            hf_dataset.save_to_disk(str(self.output_hf))
            logger.info(f"Saved HuggingFace dataset: {self.output_hf}")
            
            # Log statistics
            self._log_statistics(processed_data)
            
            return hf_dataset
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def _log_statistics(self, data: List[Dict]):
        """Log dataset statistics"""
        total = len(data)
        description_lengths = [len(item['voice_description']) for item in data]
        
        logger.info(f"\n{self.dataset_name.upper()} Statistics:")
        logger.info(f"Total samples: {total}")
        logger.info(f"Average description length: {np.mean(description_lengths):.0f} chars")
        
        # Show field distributions if available
        for field in ['gender', 'age', 'accent']:
            if any(field in item for item in data):
                counts = {}
                for item in data:
                    value = item.get(field, 'unknown')
                    counts[value] = counts.get(value, 0) + 1
                logger.info(f"{field.title()} distribution: {counts}")