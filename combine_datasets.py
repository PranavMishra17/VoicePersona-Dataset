#!/usr/bin/env python3
"""
Combine all VoicePersona datasets with audio recovery and memory management
"""
import json
import os
import shutil
from pathlib import Path
from datasets import Dataset, Audio, load_dataset, concatenate_datasets
from huggingface_hub import HfApi
import soundfile as sf
import numpy as np
from tqdm import tqdm
import gc
import tempfile
import psutil

class DatasetCombiner:
    def __init__(self, base_path="dataset"):
        self.base_path = Path(base_path)
        self.temp_dir = Path("temp_batches")
        self.stats = {
            'total_processed': 0,
            'missing_audio': 0,
            'recovered_audio': 0,
            'audio_errors': 0,
            'successful': 0
        }
        self.original_datasets = {}
        
    def cleanup_previous_runs(self):
        """Clean up previous runs"""
        paths_to_clean = [
            Path("voicepersona_combined"),
            self.temp_dir
        ]
        
        for path in paths_to_clean:
            if path.exists():
                print(f"🧹 Cleaning up {path}")
                shutil.rmtree(path)
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
    
    def load_original_datasets(self):
        """Load original datasets for audio recovery"""
        print("📡 Loading original datasets for audio recovery...")
        
        dataset_configs = {
            'globe_v2': ("MushanW/GLOBE_V2", "train"),
            'laions': ("laion/laions_got_talent", "train"),
            'animevox': ("taresh18/AnimeVox", "train"),
            'anispeech': ("ShoukanLabs/AniSpeech", "ENGLISH")
        }
        
        for name, (repo, split) in dataset_configs.items():
            try:
                print(f"  Loading {name}...")
                dataset = load_dataset(repo, split=split, streaming=True)
                self.original_datasets[name] = dataset
                print(f"  ✅ {name} loaded")
            except Exception as e:
                print(f"  ⚠️ Failed to load {name}: {e}")
                self.original_datasets[name] = None
    
    def find_missing_audio(self, item, dataset_name):
        """Attempt to recover missing audio from original dataset"""
        if dataset_name not in self.original_datasets or self.original_datasets[dataset_name] is None:
            return None
        
        try:
            original_dataset = self.original_datasets[dataset_name]
            speaker_id = item['speaker_id']
            transcript = item['transcript']
            
            # Search through original dataset
            for orig_sample in original_dataset:
                match_found = False
                
                if dataset_name == 'globe_v2':
                    match_found = (orig_sample.get('speaker_id') == speaker_id and 
                                 orig_sample.get('transcript', '').strip() == transcript.strip())
                elif dataset_name == 'laions':
                    orig_metadata = orig_sample.get('json', {})
                    match_found = (orig_metadata.get('text', '').strip() == transcript.strip())
                elif dataset_name == 'animevox':
                    match_found = (orig_sample.get('transcription', '').strip() == transcript.strip())
                elif dataset_name == 'anispeech':
                    match_found = (orig_sample.get('caption', '').strip() == transcript.strip())
                
                if match_found:
                    print(f"    📁 Found match for {speaker_id}")
                    audio_data = orig_sample['audio']
                    
                    # Handle different audio formats
                    if hasattr(audio_data, 'decode'):
                        decoded = audio_data.decode()
                        return decoded['array'], decoded['sampling_rate']
                    elif isinstance(audio_data, dict):
                        return audio_data['array'], audio_data['sampling_rate']
                    else:
                        return audio_data, 16000
            
            return None
            
        except Exception as e:
            print(f"    ⚠️ Error recovering audio for {speaker_id}: {e}")
            return None
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def load_dataset_data(self, dataset_name):
        """Load dataset with audio recovery"""
        dataset_path = self.base_path / dataset_name
        json_file = dataset_path / f"{dataset_name}_descriptions.json"
        audio_dir = dataset_path / "audio"
        
        print(f"Loading {dataset_name}...")
        
        if not json_file.exists():
            print(f"❌ JSON file not found: {json_file}")
            return []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        dataset_missing = 0
        dataset_recovered = 0
        dataset_errors = 0
        
        for item in tqdm(data, desc=f"Processing {dataset_name}"):
            self.stats['total_processed'] += 1
            
            audio_filename = Path(item['audio_path']).name
            audio_file_path = audio_dir / audio_filename
            
            audio_array = None
            sample_rate = None
            
            # Try to load existing audio file
            if audio_file_path.exists():
                try:
                    audio_array, sample_rate = sf.read(str(audio_file_path))
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                except Exception as e:
                    print(f"    ⚠️ Error reading {audio_file_path}: {e}")
            
            # If no audio found, try to recover from original dataset
            if audio_array is None:
                dataset_missing += 1
                self.stats['missing_audio'] += 1
                
                print(f"  🔍 Attempting to recover: {item['speaker_id']}")
                recovery_result = self.find_missing_audio(item, dataset_name)
                
                if recovery_result:
                    audio_array, sample_rate = recovery_result
                    
                    # Save recovered audio
                    audio_dir.mkdir(exist_ok=True)
                    try:
                        if len(audio_array.shape) > 1:
                            audio_array = np.mean(audio_array, axis=1)
                        sf.write(str(audio_file_path), audio_array, sample_rate)
                        dataset_recovered += 1
                        self.stats['recovered_audio'] += 1
                        print(f"    ✅ Recovered and saved: {audio_filename}")
                    except Exception as e:
                        print(f"    ⚠️ Failed to save recovered audio: {e}")
                        audio_array = None
            
            if audio_array is None or len(audio_array) == 0:
                dataset_errors += 1
                self.stats['audio_errors'] += 1
                continue
            
            # Create processed item
            try:
                processed_item = {
                    'index': item['index'],
                    'dataset': item['dataset'], 
                    'speaker_id': item['speaker_id'],
                    'transcript': item['transcript'],
                    'audio': {
                        'array': audio_array.astype(np.float32),
                        'sampling_rate': int(sample_rate)
                    },
                    'duration': float(item['duration']),
                    'voice_description': item['voice_description'],
                    'gender': item.get('gender', 'unknown'),
                    'age': item.get('age', 'unknown'), 
                    'accent': item.get('accent', 'unknown')
                }
                
                processed_data.append(processed_item)
                self.stats['successful'] += 1
                
            except Exception as e:
                dataset_errors += 1
                self.stats['audio_errors'] += 1
                continue
        
        print(f"✅ {dataset_name}: {len(processed_data)} successful, {dataset_missing} missing, {dataset_recovered} recovered, {dataset_errors} errors")
        return processed_data
    
    def save_batch_to_disk(self, batch_data, batch_num):
        """Save individual batch to disk"""
        batch_path = self.temp_dir / f"batch_{batch_num}"
        
        try:
            batch_dataset = Dataset.from_list(batch_data)
            batch_dataset = batch_dataset.cast_column("audio", Audio(sampling_rate=16000))
            batch_dataset.save_to_disk(str(batch_path))
            return batch_path
        except Exception as e:
            print(f"    ❌ Failed to save batch {batch_num}: {e}")
            return None
    
    def create_dataset_with_disk_batching(self, combined_data, batch_size=50):
        """Create dataset using disk-based batching to avoid memory issues"""
        print(f"Creating dataset with disk batching (batch size: {batch_size})...")
        
        if not combined_data:
            raise ValueError("No data to create dataset from")
        
        batch_paths = []
        total_batches = (len(combined_data) + batch_size - 1) // batch_size
        
        # Process and save each batch to disk
        for i in range(0, len(combined_data), batch_size):
            batch_data = combined_data[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_data)} samples)")
            print(f"    Memory usage: {self.get_memory_usage_mb():.1f} MB")
            
            batch_path = self.save_batch_to_disk(batch_data, batch_num)
            if batch_path:
                batch_paths.append(batch_path)
            
            # Clear memory
            del batch_data
            gc.collect()
        
        if not batch_paths:
            raise ValueError("No batches were successfully saved")
        
        # Load and concatenate batches
        print(f"  Loading and concatenating {len(batch_paths)} batches...")
        datasets_to_concat = []
        
        for i, batch_path in enumerate(batch_paths):
            try:
                print(f"    Loading batch {i+1}/{len(batch_paths)}")
                batch_dataset = Dataset.load_from_disk(str(batch_path))
                datasets_to_concat.append(batch_dataset)
            except Exception as e:
                print(f"    ⚠️ Failed to load batch {i+1}: {e}")
                continue
        
        if not datasets_to_concat:
            raise ValueError("No batches could be loaded")
        
        # Concatenate all datasets
        final_dataset = concatenate_datasets(datasets_to_concat)
        
        # Cleanup temp files
        print("  🧹 Cleaning up temporary files...")
        for batch_path in batch_paths:
            if batch_path.exists():
                shutil.rmtree(batch_path)
        
        print(f"✅ Dataset created with {len(final_dataset)} samples")
        return final_dataset
    
    def combine_all_datasets(self):
        """Main combination process"""
        datasets = ['animevox', 'anispeech', 'globe_v2', 'laions']
        all_data = []
        
        # Load original datasets for recovery
        self.load_original_datasets()
        
        global_index = 0
        for dataset_name in datasets:
            try:
                dataset_data = self.load_dataset_data(dataset_name)
                
                # Update global index
                for item in dataset_data:
                    item['index'] = global_index
                    global_index += 1
                    all_data.append(item)
                    
            except Exception as e:
                print(f"❌ Error processing {dataset_name}: {e}")
                continue
        
        print(f"\n📊 Final Statistics:")
        print(f"Total items processed: {self.stats['total_processed']}")
        print(f"Missing audio files: {self.stats['missing_audio']}")
        print(f"Recovered audio files: {self.stats['recovered_audio']}")
        print(f"Audio processing errors: {self.stats['audio_errors']}")
        print(f"Successfully combined: {self.stats['successful']}")
        
        return all_data
    
    def save_dataset_locally(self, dataset, output_path="voicepersona_combined"):
        """Save final dataset locally"""
        try:
            print(f"💾 Saving dataset to {output_path}...")
            output_path = Path(output_path)
            
            if output_path.exists():
                shutil.rmtree(output_path)
            
            dataset.save_to_disk(str(output_path))
            print("✅ Dataset saved locally!")
            return True
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False
    
    def push_to_huggingface(self, dataset, repo_id="Paranoiid/VoicePersona"):
        """Push to HuggingFace with error handling"""
        try:
            print(f"🚀 Pushing to HuggingFace: {repo_id}")
            dataset.push_to_hub(repo_id, private=False)
            print("✅ Successfully pushed to HuggingFace!")
            return True
        except Exception as e:
            print(f"❌ Error pushing to HuggingFace: {e}")
            return False

def main():
    print("🎯 VoicePersona Dataset Combiner v3.0")
    print("=" * 50)
    
    combiner = DatasetCombiner()
    
    try:
        # Cleanup previous runs
        combiner.cleanup_previous_runs()
        
        # Combine datasets with recovery
        combined_data = combiner.combine_all_datasets()
        
        if not combined_data:
            print("❌ No data to process!")
            return
        
        # Create dataset with disk batching
        dataset = combiner.create_dataset_with_disk_batching(combined_data, batch_size=50)
        
        # Save locally
        local_success = combiner.save_dataset_locally(dataset)
        
        if not local_success:
            print("❌ Failed to save locally. Aborting.")
            return
        
        # Show sample and stats
        print(f"\n📊 Sample Data Preview:")
        sample = dataset[0]
        for key, value in sample.items():
            if key == 'audio':
                print(f"{key}: array shape {value['array'].shape}, sr {value['sampling_rate']}")
            elif key == 'voice_description':
                print(f"{key}: {str(value)[:100]}...")
            else:
                print(f"{key}: {value}")
        
        print(f"\n📈 Final Dataset: {len(dataset)} samples")
        
        # Count by dataset
        dataset_counts = {}
        for item in combined_data:
            ds = item['dataset']
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        for ds, count in dataset_counts.items():
            print(f"  {ds}: {count} samples")
        
        # Cleanup temp directory
        if combiner.temp_dir.exists():
            shutil.rmtree(combiner.temp_dir)
        
        # Ask about HuggingFace upload
        push_choice = input(f"\n🚀 Push to HuggingFace? (y/n): ").strip().lower()
        
        if push_choice == 'y':
            combiner.push_to_huggingface(dataset)
        else:
            print("\n💡 To upload manually:")
            print("from datasets import load_from_disk")
            print("dataset = load_from_disk('voicepersona_combined')")
            print("dataset.push_to_hub('Paranoiid/VoicePersona')")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        # Cleanup on error
        if combiner.temp_dir.exists():
            shutil.rmtree(combiner.temp_dir)

if __name__ == "__main__":
    main()

"""
python combine_datasets.py


What the Script Does
    Loads JSON files and audio from each dataset folder
    Combines ~80K samples with proper global indexing
    Creates HuggingFace dataset with Audio feature
    Saves locally first as backup
    Shows preview and statistics
    Asks whether to push to HuggingFace



💡 To upload manually:
from datasets import load_from_disk
dataset = load_from_disk('voicepersona_combined')
dataset.push_to_hub('Paranoiid/VoicePersona')

"""