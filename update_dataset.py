#!/usr/bin/env python3
"""
Update VoicePersona Dataset with checkpointing for memory efficiency
"""
import random
import pickle
import gc
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Audio
from pathlib import Path
import pandas as pd

def save_checkpoint(data, filename):
    """Save checkpoint to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    """Load checkpoint from disk"""
    if Path(filename).exists():
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def update_voicepersona_dataset(source="local", output_path="voicepersona_updated"):
    """Update dataset with checkpointing"""
    print("🔄 Loading dataset...")
    
    if source == "local" and Path("voicepersona_combined").exists():
        dataset = load_from_disk("voicepersona_combined")
    else:
        dataset = load_dataset("Paranoiid/VoicePersona")['train']
    
    print(f"📊 Original dataset: {len(dataset):,} samples")
    
    # Step 1: Filter indices only (no audio)
    checkpoint_file = "filter_checkpoint.pkl"
    filtered_indices = load_checkpoint(checkpoint_file)
    
    if filtered_indices is None:
        print("🔍 Filtering samples (metadata only)...")
        filtered_indices = []
        
        for i in range(len(dataset)):
            if i % 1000 == 0:
                print(f"  Checking sample {i:,}/{len(dataset):,}")
            
            # Get only text data (no audio)
            sample = dataset.select([i]).remove_columns(['audio'])[0]
            voice_desc = sample.get('voice_description', '')
            
            if voice_desc and len(voice_desc.strip()) >= 60:
                filtered_indices.append(i)
        
        save_checkpoint(filtered_indices, checkpoint_file)
        print(f"✅ Filtered indices saved: {len(filtered_indices):,} samples")
    else:
        print(f"📁 Loaded filtered indices: {len(filtered_indices):,} samples")
    
    # Step 2: Create splits
    random.seed(42)
    random.shuffle(filtered_indices)
    
    train_size = min(10000, len(filtered_indices) * 2 // 3)
    val_size = min(3000, len(filtered_indices) // 6)
    test_size = min(2000, len(filtered_indices) - train_size - val_size)
    
    splits = {
        'train': filtered_indices[:train_size],
        'validation': filtered_indices[train_size:train_size + val_size],
        'test': filtered_indices[train_size + val_size:train_size + val_size + test_size]
    }
    
    print(f"📊 Splits: Train={len(splits['train'])}, Val={len(splits['validation'])}, Test={len(splits['test'])}")
    
    # Step 3: Process each split with checkpointing
    dataset_dict = {}
    
    for split_name, indices in splits.items():
        print(f"🔧 Processing {split_name} split...")
        
        split_checkpoint = f"{split_name}_checkpoint.pkl"
        processed_data = load_checkpoint(split_checkpoint)
        
        if processed_data is None:
            processed_data = []
            batch_size = 10  # Very small batches
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                print(f"  Batch {i//batch_size + 1}/{(len(indices) + batch_size - 1)//batch_size}")
                
                for j, orig_idx in enumerate(batch_indices):
                    try:
                        # Get single item with minimal memory
                        item = dataset.select([orig_idx])[0]
                        
                        processed_item = {
                            'speaker_id': len(processed_data),
                            'transcript': item['transcript'],
                            'audio': item['audio'],
                            'voice_description': item['voice_description'],
                            'gender': item.get('gender', 'unknown'),
                            'age': item.get('age', 'unknown'),
                            'accent': item.get('accent', 'unknown'),
                            'duration': item['duration'],
                            'dataset': item['dataset']
                        }
                        processed_data.append(processed_item)
                        
                        # Force memory cleanup
                        del item
                        gc.collect()
                        
                    except Exception as e:
                        print(f"    ⚠️ Skipped sample {orig_idx}: {e}")
                        continue
                
                # Save checkpoint every batch
                if i % (batch_size * 10) == 0:
                    save_checkpoint(processed_data, split_checkpoint)
            
            save_checkpoint(processed_data, split_checkpoint)
        
        print(f"✅ {split_name}: {len(processed_data)} samples")
        
        # Create dataset
        split_dataset = Dataset.from_list(processed_data)
        split_dataset = split_dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset_dict[split_name] = split_dataset
        
        # Cleanup
        del processed_data
        gc.collect()
    
    # Create final dataset
    final_dataset = DatasetDict(dataset_dict)
    
    # Save locally
    print(f"💾 Saving to {output_path}...")
    final_dataset.save_to_disk(output_path)
    
    # Cleanup checkpoints
    for f in ['filter_checkpoint.pkl', 'train_checkpoint.pkl', 'validation_checkpoint.pkl', 'test_checkpoint.pkl']:
        if Path(f).exists():
            Path(f).unlink()
    
    print("✅ Dataset updated successfully!")
    return final_dataset

def upload_updated_dataset(dataset_dict, repo_id="Paranoiid/VoicePersona"):
    """Upload updated dataset to HuggingFace Hub"""
    print(f"🚀 Uploading to {repo_id}...")
    
    try:
        dataset_dict.push_to_hub(repo_id, private=False)
        print("✅ Upload successful!")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    print("🎯 VoicePersona Dataset Updater (Memory-Safe)")
    print("=" * 50)
    
    dataset_dict = update_voicepersona_dataset(source="local")
    
    repo_id = "Paranoiid/VoicePersona"
    print(f"\n🚀 Auto-uploading to {repo_id}...")
    
    success = upload_updated_dataset(dataset_dict, repo_id)
    
    if success:
        print("\n🎉 Dataset updated and uploaded successfully!")
    else:
        print("\n💡 Upload failed. Dataset saved locally.")

if __name__ == "__main__":
    main()