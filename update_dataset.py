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
    
    # Step 3: Process each split with disk-based batching
    dataset_dict = {}
    temp_dir = Path("temp_splits")
    temp_dir.mkdir(exist_ok=True)
    
    for split_name, indices in splits.items():
        print(f"🔧 Processing {split_name} split...")
        
        split_checkpoint = f"{split_name}_checkpoint.pkl"
        processed_count = 0
        
        # Check existing checkpoint
        if Path(split_checkpoint).exists():
            with open(split_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            # Handle old checkpoint format (list) vs new format (int)
            if isinstance(checkpoint_data, list):
                processed_count = len(checkpoint_data)
                # Remove old checkpoint
                Path(split_checkpoint).unlink()
            else:
                processed_count = checkpoint_data
            print(f"📁 Resuming from sample {processed_count}")
        
        # Process in very small batches and save immediately
        batch_datasets = []
        batch_size = 10
        
        # Skip if already complete
        if processed_count >= len(indices):
            print(f"  ✅ Split already complete, loading existing batches...")
            # Find existing batch files
            for i in range(1000):  # Max batches to check
                batch_file = temp_dir / f"{split_name}_batch_{i}"
                if batch_file.exists():
                    batch_datasets.append(batch_file)
                else:
                    break
        else:
            for i in range(processed_count, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                print(f"  Batch {(i//batch_size) + 1}/{(len(indices) + batch_size - 1)//batch_size}")
                
                batch_data = []
                for j, orig_idx in enumerate(batch_indices):
                    try:
                        item = dataset.select([orig_idx])[0]
                        
                        processed_item = {
                            'speaker_id': i + j,
                            'transcript': item['transcript'],
                            'audio': item['audio'],
                            'voice_description': item['voice_description'],
                            'gender': item.get('gender', 'unknown'),
                            'age': item.get('age', 'unknown'),
                            'accent': item.get('accent', 'unknown'),
                            'duration': item['duration'],
                            'dataset': item['dataset']
                        }
                        batch_data.append(processed_item)
                        
                        del item
                        gc.collect()
                        
                    except Exception as e:
                        print(f"    ⚠️ Skipped sample {orig_idx}: {e}")
                        continue
                
                # Save batch immediately if we have data
                if batch_data:
                    batch_dataset = Dataset.from_list(batch_data)
                    batch_dataset = batch_dataset.cast_column("audio", Audio(sampling_rate=16000))
                    
                    batch_file = temp_dir / f"{split_name}_batch_{len(batch_datasets)}"
                    batch_dataset.save_to_disk(str(batch_file))
                    batch_datasets.append(batch_file)
                    
                    del batch_data, batch_dataset
                    gc.collect()
                
                # Save checkpoint
                with open(split_checkpoint, 'wb') as f:
                    pickle.dump(i + batch_size, f)
        
        # Concatenate all batches
        print(f"  📦 Combining {len(batch_datasets)} batches...")
        if batch_datasets:
            # Load and concatenate
            datasets_to_concat = []
            for batch_file in batch_datasets:
                batch_ds = Dataset.load_from_disk(str(batch_file))
                datasets_to_concat.append(batch_ds)
            
            from datasets import concatenate_datasets
            final_split = concatenate_datasets(datasets_to_concat)
            dataset_dict[split_name] = final_split
            
            # Cleanup batch files
            for batch_file in batch_datasets:
                import shutil
                shutil.rmtree(batch_file)
        
        print(f"✅ {split_name}: {len(dataset_dict[split_name])} samples")
        
        # Remove checkpoint
        if Path(split_checkpoint).exists():
            Path(split_checkpoint).unlink()
    
    # Create final dataset
    final_dataset = DatasetDict(dataset_dict)
    
    # Save locally
    print(f"💾 Saving to {output_path}...")
    final_dataset.save_to_disk(output_path)
    
    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    for f in ['filter_checkpoint.pkl']:
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