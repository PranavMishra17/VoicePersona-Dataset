#!/usr/bin/env python3
"""
Combine all VoicePersona datasets into one HuggingFace dataset with error handling
"""
import json
import os
from pathlib import Path
from datasets import Dataset, Audio
from huggingface_hub import HfApi
import soundfile as sf
import numpy as np
from tqdm import tqdm
import gc

class DatasetCombiner:
    def __init__(self, base_path="dataset"):
        self.base_path = Path(base_path)
        self.stats = {
            'total_processed': 0,
            'missing_audio': 0,
            'audio_errors': 0,
            'successful': 0
        }
    
    def load_dataset_data(self, dataset_name):
        """Load dataset JSON and audio files with error handling"""
        dataset_path = self.base_path / dataset_name
        json_file = dataset_path / f"{dataset_name}_descriptions.json"
        audio_dir = dataset_path / "audio"
        
        print(f"Loading {dataset_name}...")
        
        if not json_file.exists():
            print(f"❌ JSON file not found: {json_file}")
            return []
        
        # Load JSON data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON for {dataset_name}: {e}")
            return []
        
        # Process each item
        processed_data = []
        dataset_missing = 0
        dataset_errors = 0
        
        for item in tqdm(data, desc=f"Processing {dataset_name}"):
            self.stats['total_processed'] += 1
            
            # Extract audio filename from path
            audio_filename = Path(item['audio_path']).name
            audio_file_path = audio_dir / audio_filename
            
            # Skip if audio file doesn't exist
            if not audio_file_path.exists():
                dataset_missing += 1
                self.stats['missing_audio'] += 1
                continue
            
            # Load audio data with error handling
            try:
                audio_array, sample_rate = sf.read(str(audio_file_path))
                
                # Ensure mono audio
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                
                # Validate audio data
                if len(audio_array) == 0:
                    dataset_errors += 1
                    self.stats['audio_errors'] += 1
                    continue
                
                # Create item with desired field order
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
        
        print(f"✅ {dataset_name}: {len(processed_data)} successful, {dataset_missing} missing audio, {dataset_errors} audio errors")
        return processed_data
    
    def combine_all_datasets(self):
        """Combine all four datasets"""
        datasets = ['animevox', 'anispeech', 'globe_v2', 'laions']
        all_data = []
        
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
        print(f"Audio processing errors: {self.stats['audio_errors']}")
        print(f"Successfully combined: {self.stats['successful']}")
        
        return all_data
    
    def create_huggingface_dataset_batch(self, combined_data, batch_size=1000):
        """Create HuggingFace dataset in batches to avoid memory issues"""
        print("Creating HuggingFace dataset in batches...")
        
        if not combined_data:
            raise ValueError("No data to create dataset from")
        
        try:
            # Process in smaller batches to avoid memory issues
            all_batches = []
            total_batches = (len(combined_data) + batch_size - 1) // batch_size
            
            for i in range(0, len(combined_data), batch_size):
                batch_data = combined_data[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_data)} samples)")
                
                try:
                    batch_dataset = Dataset.from_list(batch_data)
                    batch_dataset = batch_dataset.cast_column("audio", Audio(sampling_rate=16000))
                    all_batches.append(batch_dataset)
                    
                    # Clear memory
                    gc.collect()
                    
                except Exception as e:
                    print(f"⚠️ Error processing batch {i//batch_size + 1}: {e}")
                    continue
            
            if not all_batches:
                raise ValueError("No batches were successfully processed")
            
            # Concatenate all batches
            print("Concatenating batches...")
            from datasets import concatenate_datasets
            final_dataset = concatenate_datasets(all_batches)
            
            print(f"✅ Dataset created with {len(final_dataset)} samples")
            print(f"Dataset features: {list(final_dataset.features.keys())}")
            
            return final_dataset
            
        except Exception as e:
            print(f"❌ Error creating HuggingFace dataset: {e}")
            raise
    
    def save_dataset_locally(self, dataset, output_path="voicepersona_combined"):
        """Save dataset locally with error handling"""
        try:
            print(f"Saving dataset to {output_path}...")
            output_path = Path(output_path)
            
            # Remove existing directory if it exists
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)
            
            dataset.save_to_disk(str(output_path))
            print("✅ Dataset saved locally!")
            return True
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False
    
    def push_to_huggingface(self, dataset, repo_id="Paranoiid/VoicePersona"):
        """Push dataset to HuggingFace Hub with error handling"""
        try:
            print(f"Pushing to HuggingFace: {repo_id}")
            dataset.push_to_hub(repo_id, private=False)
            print("✅ Successfully pushed to HuggingFace!")
            return True
        except Exception as e:
            print(f"❌ Error pushing to HuggingFace: {e}")
            print("This might be due to:")
            print("- Authentication issues (run: huggingface-cli login)")
            print("- Large dataset size (try manual upload)")
            print("- Network connectivity issues")
            return False

def main():
    print("🎯 VoicePersona Dataset Combiner v2.0")
    print("=" * 50)
    
    combiner = DatasetCombiner()
    
    try:
        # Combine datasets
        combined_data = combiner.combine_all_datasets()
        
        if not combined_data:
            print("❌ No data to process!")
            return
        
        # Create HuggingFace dataset in batches
        dataset = combiner.create_huggingface_dataset_batch(combined_data, batch_size=500)
        
        # Save locally first
        local_success = combiner.save_dataset_locally(dataset)
        
        if not local_success:
            print("❌ Failed to save locally. Aborting.")
            return
        
        # Show sample data
        print("\n📊 Sample Data Preview:")
        print("-" * 30)
        sample = dataset[0]
        for key, value in sample.items():
            if key == 'audio':
                print(f"{key}: array shape {value['array'].shape}, sr {value['sampling_rate']}")
            elif key == 'voice_description':
                print(f"{key}: {str(value)[:100]}...")
            else:
                print(f"{key}: {value}")
        
        print(f"\n📈 Dataset Statistics:")
        print(f"Final dataset size: {len(dataset)}")
        
        # Count by dataset
        dataset_counts = {}
        for item in combined_data:
            ds = item['dataset']
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        for ds, count in dataset_counts.items():
            print(f"  {ds}: {count} samples")
        
        # Ask about pushing to HuggingFace
        print("\n" + "=" * 50)
        push_choice = input("🚀 Push to HuggingFace (https://huggingface.co/datasets/Paranoiid/VoicePersona)? (y/n): ").strip().lower()
        
        if push_choice == 'y':
            success = combiner.push_to_huggingface(dataset)
            if not success:
                print_manual_upload_instructions()
        else:
            print_manual_upload_instructions()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Check the error above and ensure:")
        print("- Dataset folders exist with proper structure")
        print("- Sufficient disk space and memory available")
        print("- Audio files are not corrupted")

def print_manual_upload_instructions():
    """Print manual upload instructions"""
    print("\n📋 Manual Upload Instructions:")
    print("=" * 40)
    print("1. Files created:")
    print("   - voicepersona_combined/ (local dataset folder)")
    print("\n2. Manual upload options:")
    print("   A) Using huggingface_hub:")
    print("      from datasets import load_from_disk")
    print("      dataset = load_from_disk('voicepersona_combined')")
    print("      dataset.push_to_hub('Paranoiid/VoicePersona')")
    print("\n   B) Using git LFS:")
    print("      git clone https://huggingface.co/datasets/Paranoiid/VoicePersona")
    print("      cp -r voicepersona_combined/* VoicePersona/")
    print("      cd VoicePersona && git lfs track '*.arrow' && git add . && git commit -m 'Upload dataset' && git push")
    print("\n⚠️  Note: This dataset is large - git LFS or programmatic upload recommended")

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


"""