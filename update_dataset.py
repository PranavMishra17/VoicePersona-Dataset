#!/usr/bin/env python3
"""
Update VoicePersona Dataset with new structure and splits
"""
import random
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Audio
from pathlib import Path
import pandas as pd

def update_voicepersona_dataset(source="local", output_path="voicepersona_updated"):
    """
    Update dataset with new structure and splits
    
    Args:
        source: "local" for local dataset or "hub" for HuggingFace
        output_path: Path to save updated dataset
    """
    print("🔄 Loading dataset...")
    
    # Load dataset
    if source == "local" and Path("voicepersona_combined").exists():
        dataset = load_from_disk("voicepersona_combined")
    else:
        dataset = load_dataset("Paranoiid/VoicePersona")['train']
    
    print(f"📊 Original dataset: {len(dataset):,} samples")
    
    # Convert to list for processing
    data = list(dataset)
    
    # Filter out samples with insufficient voice descriptions
    print("🔍 Filtering samples...")
    filtered_data = []
    
    for item in data:
        voice_desc = item.get('voice_description', '')
        
        # Skip if voice description is null, empty, or < 60 characters
        if not voice_desc or len(voice_desc.strip()) < 60:
            continue
            
        filtered_data.append(item)
    
    print(f"✅ After filtering: {len(filtered_data):,} samples ({len(data) - len(filtered_data):,} removed)")
    
    # Shuffle data for random splits
    print("🔀 Shuffling data...")
    random.seed(42)  # For reproducibility
    random.shuffle(filtered_data)
    
    # Create splits
    train_size = 10000
    val_size = 3000
    test_size = 2000
    total_needed = train_size + val_size + test_size
    
    if len(filtered_data) < total_needed:
        print(f"⚠️ Warning: Only {len(filtered_data):,} samples available, need {total_needed:,}")
        # Adjust sizes proportionally
        ratio = len(filtered_data) / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(filtered_data) - train_size - val_size
    
    print(f"📊 Creating splits: Train={train_size:,}, Val={val_size:,}, Test={test_size:,}")
    
    # Split data
    train_data = filtered_data[:train_size]
    val_data = filtered_data[train_size:train_size + val_size]
    test_data = filtered_data[train_size + val_size:train_size + val_size + test_size]
    
    # Process each split
    def process_split(split_data, split_name):
        print(f"🔧 Processing {split_name} split...")
        processed = []
        
        for i, item in enumerate(split_data):
            # Create new structure
            processed_item = {
                'speaker_id': i,  # New speaker_id as index
                'transcript': item['transcript'],
                'audio': item['audio'],
                'voice_description': item['voice_description'],
                'gender': item.get('gender', 'unknown'),
                'age': item.get('age', 'unknown'),
                'accent': item.get('accent', 'unknown'),
                'duration': item['duration'],
                'dataset': item['dataset']
            }
            processed.append(processed_item)
        
        return processed
    
    # Process all splits
    train_processed = process_split(train_data, "train")
    val_processed = process_split(val_data, "validation")
    test_processed = process_split(test_data, "test")
    
    # Create HuggingFace datasets
    print("📦 Creating HuggingFace datasets...")
    
    train_dataset = Dataset.from_list(train_processed)
    val_dataset = Dataset.from_list(val_processed)
    test_dataset = Dataset.from_list(test_processed)
    
    # Cast audio column
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save locally
    print(f"💾 Saving to {output_path}...")
    dataset_dict.save_to_disk(output_path)
    
    # Show sample
    print("\n📋 Sample from train split:")
    sample = train_dataset[0]
    for key, value in sample.items():
        if key == 'audio':
            print(f"  {key}: array shape {value['array'].shape}, sr {value['sampling_rate']}")
        elif key == 'voice_description':
            print(f"  {key}: {len(value)} chars - {value[:60]}...")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✅ Dataset updated successfully!")
    print(f"📊 Final stats:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples") 
    print(f"  Test: {len(test_dataset):,} samples")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,} samples")
    
    return dataset_dict

def upload_updated_dataset(dataset_dict, repo_id="Paranoiid/VoicePersona"):
    """Upload updated dataset to HuggingFace Hub"""
    
    print(f"🚀 Uploading updated dataset to {repo_id}...")
    
    try:
        # Push to hub
        dataset_dict.push_to_hub(repo_id, private=False)
        print("✅ Successfully uploaded to HuggingFace!")
        print(f"🌐 Dataset available at: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    print("🎯 VoicePersona Dataset Updater")
    print("=" * 50)
    
    # Load from local dataset and update
    dataset_dict = update_voicepersona_dataset(source="local")
    
    # Auto-upload to HuggingFace
    repo_id = "Paranoiid/VoicePersona"
    print(f"\n🚀 Auto-uploading to {repo_id}...")
    
    success = upload_updated_dataset(dataset_dict, repo_id)
    
    if success:
        print("\n🎉 Dataset updated and uploaded successfully!")
        print("✅ Your HuggingFace dataset page now has the updated structure and splits")
    else:
        print("\n💡 Upload failed. Dataset saved locally. To upload manually:")
        print("from datasets import load_from_disk")
        print("dataset = load_from_disk('voicepersona_updated')")
        print(f"dataset.push_to_hub('{repo_id}')")

if __name__ == "__main__":
    main()