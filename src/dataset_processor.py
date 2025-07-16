"""
Lightweight dataset processor with streaming support
"""
import os
import json
import jsonlines
import pickle
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Iterator
from datasets import load_dataset, Dataset
import torch
import gc
from datetime import datetime
import random

from model_manager import create_analyzer
from prompts import VoiceDescriptionPrompts

logger = logging.getLogger(__name__)

class GlobeV2StreamProcessor:
    """Process GLOBE_V2 with streaming to minimize disk usage"""
    
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.analyzer = None
        self.checkpoint_data = {
            'processed_count': 0,
            'last_index': -1,
            'start_time': None,
            'total_time': 0
        }
        self.temp_audio_path = config.DATA_DIR / "temp_audio.wav"
        
    def load_dataset_streaming(self) -> Iterator:
        """Load dataset in streaming mode"""
        try:
            logger.info("Loading GLOBE_V2 dataset in streaming mode...")
            
            if self.config.USE_STREAMING:
                # Stream the dataset
                dataset = load_dataset(
                    self.config.DATASET_NAME,
                    split=self.config.SUBSET_SPLIT if self.config.USE_SUBSET else "train",
                    streaming=True,
                    cache_dir=self.config.CACHE_DIR
                )
                
                # Apply sampling if needed
                if self.config.USE_SAMPLING:
                    dataset = dataset.filter(
                        lambda x: random.random() < self.config.SAMPLE_RATE
                    )
                
                return dataset
            else:
                # Load subset normally
                dataset = load_dataset(
                    self.config.DATASET_NAME,
                    split=self.config.SUBSET_SPLIT,
                    cache_dir=self.config.CACHE_DIR
                )
                return iter(dataset)
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def save_audio_temp(self, audio_dict: Dict) -> str:
        """Save audio to temporary file"""
        try:
            if isinstance(audio_dict, dict):
                audio_array = audio_dict['array']
                sample_rate = audio_dict['sampling_rate']
            else:
                audio_array = audio_dict
                sample_rate = 16000
            
            # Ensure directory exists
            self.temp_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio
            sf.write(str(self.temp_audio_path), audio_array, sample_rate)
            return str(self.temp_audio_path)
            
        except Exception as e:
            logger.error(f"Failed to save temp audio: {str(e)}")
            raise
    
    def process_single_sample(self, sample: Dict) -> Optional[Dict[str, Any]]:
        """Process a single sample"""
        try:
            # Save audio to temp file
            temp_path = self.save_audio_temp(sample['audio'])
            
            # Generate description
            prompt = VoiceDescriptionPrompts.get_combined_prompt()
            description = self.analyzer.analyze(temp_path, prompt)
            
            # Create result
            result = {
                'index': self.checkpoint_data['processed_count'],
                'transcript': sample.get('transcript', ''),
                'speaker_id': sample.get('speaker_id', ''),
                'accent': sample.get('accent', ''),
                'age': sample.get('age', ''),
                'gender': sample.get('gender', ''),
                'duration': sample.get('duration', 0.0),
                'description': description,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Cleanup temp file
            if self.temp_audio_path.exists():
                os.remove(self.temp_audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample: {str(e)}")
            return None
    
    def process_streaming(self, max_samples: Optional[int] = None,
                         output_file: Optional[str] = None):
        """Process dataset in streaming mode"""
        # Initialize analyzer
        self.analyzer = create_analyzer(self.config)
        self.analyzer.load_model()
        
        # Setup output file
        if output_file is None:
            output_file = self.config.OUTPUT_JSON
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if exists
        self.load_checkpoint()
        start_count = self.checkpoint_data['processed_count']
        
        # Get dataset iterator
        dataset_iter = self.load_dataset_streaming()
        
        # Process samples
        processed_count = 0
        self.processing_start_time = datetime.now()
        
        # Open output file in append mode
        with jsonlines.open(output_file, mode='a') as writer:
            # Create progress bar
            pbar = tqdm(
                dataset_iter, 
                desc="Processing samples",
                initial=start_count,
                total=max_samples
            )
            
            for idx, sample in enumerate(pbar):
                # Skip if already processed
                if idx < start_count:
                    continue
                
                # Check max samples
                if max_samples and processed_count >= max_samples:
                    break
                
                # Process sample
                result = self.process_single_sample(sample)
                
                if result:
                    # Write to file immediately
                    writer.write(result)
                    processed_count += 1
                    self.checkpoint_data['processed_count'] = start_count + processed_count
                    self.checkpoint_data['last_index'] = idx
                    
                    # Update progress
                    pbar.set_postfix({
                        'processed': processed_count,
                        'speaker': result.get('speaker_id', '')[:15]
                    })
                
                # Save checkpoint periodically
                if processed_count % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()
                    
                    # Clear cache
                    if self.config.DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Check memory usage
                if self.config.DEVICE == "cuda":
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    if mem_used > self.config.MAX_MEMORY_MB / 1000:
                        logger.warning(f"High memory usage: {mem_used:.1f}GB")
                        torch.cuda.empty_cache()
                        gc.collect()
        
        # Final cleanup
        self.save_checkpoint()
        self.analyzer.cleanup()
        
        logger.info(f"Processing complete! Processed {processed_count} samples")
        logger.info(f"Output saved to: {output_file}")
        
        return processed_count
    
    def process_batch(self, samples: List[Dict], start_idx: int = 0) -> List[Dict]:
        """Process a batch of samples"""
        results = []
        
        for idx, sample in enumerate(samples):
            logger.info(f"Processing sample {start_idx + idx}...")
            result = self.process_single_sample(sample)
            if result:
                results.append(result)
        
        return results
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        checkpoint_path = self.config.CHECKPOINT_FILE
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    self.checkpoint_data = pickle.load(f)
                logger.info(f"Checkpoint loaded. Processed: {self.checkpoint_data['processed_count']}")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
        return False
    
    def save_checkpoint(self):
        """Save checkpoint"""
        try:
            # Update timing
            if hasattr(self, 'processing_start_time'):
                elapsed = (datetime.now() - self.processing_start_time).total_seconds()
                self.checkpoint_data['total_time'] += elapsed
            
            # Save checkpoint
            self.config.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(self.checkpoint_data, f)
            
            logger.info(f"Checkpoint saved. Processed: {self.checkpoint_data['processed_count']}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def convert_to_hf_dataset(self, jsonl_file: str) -> Dataset:
        """Convert JSONL output to HuggingFace dataset"""
        try:
            # Read JSONL file
            data = []
            with jsonlines.open(jsonl_file) as reader:
                for obj in reader:
                    data.append(obj)
            
            # Create dataset
            dataset = Dataset.from_list(data)
            
            # Save to disk
            output_path = self.config.OUTPUT_HF_DATASET
            output_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"HuggingFace dataset saved to: {output_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to convert to HF dataset: {str(e)}")
            raise

def test_processing(config, n_samples: int = 2):
    """Test processing on a few samples"""
    logger.info("Starting test processing...")
    
    # Check resources first
    if not config.check_resources():
        logger.warning("Resource check failed. Proceeding anyway...")
    
    processor = GlobeV2StreamProcessor(config)
    
    # Process test samples
    test_output = config.OUTPUT_DIR / "test_output.jsonl"
    processed = processor.process_streaming(
        max_samples=n_samples,
        output_file=test_output
    )
    
    # Display results
    if test_output.exists():
        logger.info("\nTest results:")
        with jsonlines.open(test_output) as reader:
            for obj in reader:
                logger.info