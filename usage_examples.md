
# VoicePersona Dataset Usage Examples

## Installation
```bash
pip install datasets soundfile librosa torch
```

## Quick Start
```python
from datasets import load_dataset
import soundfile as sf

# Load the dataset
dataset = load_dataset("Paranoiid/VoicePersona")
print(f"Dataset size: {len(dataset['train'])}")

# Access a sample
sample = dataset['train'][0]
print(f"Speaker: {sample['speaker_id']}")
print(f"Transcript: {sample['transcript']}")
print(f"Voice description: {sample['voice_description'][:100]}...")
```

## Working with Audio
```python
import numpy as np
import soundfile as sf
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Paranoiid/VoicePersona")

# Get audio data
sample = dataset['train'][0]
audio_array = sample['audio']['array']
sampling_rate = sample['audio']['sampling_rate']

# Save audio to file
sf.write("sample_audio.wav", audio_array, sampling_rate)

# Play audio (if in Jupyter)
import IPython.display as ipd
ipd.Audio(audio_array, rate=sampling_rate)
```

## Filtering by Demographics
```python
# Filter by gender
female_samples = dataset['train'].filter(lambda x: x['gender'] == 'female')
print(f"Female samples: {len(female_samples)}")

# Filter by age group
young_samples = dataset['train'].filter(lambda x: x['age'] in ['teens', 'twenties'])

# Filter by accent
american_samples = dataset['train'].filter(
    lambda x: 'American' in x['accent']
)
```

## Dataset Analysis
```python
import pandas as pd
from collections import Counter

# Convert to pandas for analysis
df = dataset['train'].to_pandas()

# Basic statistics
print("Dataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Unique speakers: {df['speaker_id'].nunique()}")
print(f"Total duration: {df['duration'].sum()/3600:.1f} hours")
print(f"Average duration: {df['duration'].mean():.1f} seconds")

# Demographics distribution
print("\nGender distribution:")
print(df['gender'].value_counts())

print("\nAge distribution:")
print(df['age'].value_counts())

print("\nTop 10 accents:")
print(df['accent'].value_counts().head(10))

# Source dataset distribution
print("\nSource datasets:")
print(df['dataset'].value_counts())
```

## Voice Characteristic Analysis
```python
# Analyze voice descriptions
descriptions = df['voice_description'].tolist()

# Common voice characteristics
import re
characteristics = []
for desc in descriptions:
    # Extract common terms
    terms = re.findall(r'\b(warm|cool|bright|dark|deep|high|low|smooth|raspy|clear|breathy)\b', 
                      desc.lower())
    characteristics.extend(terms)

from collections import Counter
common_traits = Counter(characteristics)
print("Most common voice traits:")
for trait, count in common_traits.most_common(10):
    print(f"{trait}: {count}")
```

## Training Data Preparation
```python
# Prepare for voice synthesis training
def prepare_training_data(dataset, max_duration=10.0):
    """Prepare dataset for voice synthesis training"""
    
    def filter_sample(sample):
        # Filter by duration
        if sample['duration'] > max_duration:
            return False
        # Filter out unknown demographics
        if sample['gender'] == 'unknown' or sample['age'] == 'unknown':
            return False
        return True
    
    # Filter dataset
    filtered = dataset.filter(filter_sample)
    
    # Create training examples
    training_data = []
    for sample in filtered:
        training_example = {
            'audio': sample['audio'],
            'text': sample['transcript'],
            'speaker_embedding': {
                'gender': sample['gender'],
                'age': sample['age'],
                'accent': sample['accent'],
                'voice_profile': sample['voice_description']
            }
        }
        training_data.append(training_example)
    
    return training_data

# Prepare training data
training_data = prepare_training_data(dataset['train'])
print(f"Training samples: {len(training_data)}")
```

## Voice Similarity Search
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_voices(target_description, dataset, top_k=5):
    """Find voices similar to target description"""
    
    # Get all voice descriptions
    descriptions = [sample['voice_description'] for sample in dataset['train']]
    
    # Add target description
    all_descriptions = [target_description] + descriptions
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    
    # Calculate similarities
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get top matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        sample = dataset['train'][idx]
        results.append({
            'similarity': similarities[idx],
            'speaker_id': sample['speaker_id'],
            'description': sample['voice_description'],
            'demographics': f"{sample['gender']}, {sample['age']}, {sample['accent']}"
        })
    
    return results

# Example usage
target = "warm, medium-pitched female voice with clear articulation"
similar_voices = find_similar_voices(target, dataset)

for i, voice in enumerate(similar_voices):
    print(f"{i+1}. {voice['speaker_id']} (similarity: {voice['similarity']:.3f})")
    print(f"   {voice['demographics']}")
    print(f"   {voice['description'][:100]}...")
```

## Audio Processing Pipeline
```python
import librosa
import torch
import torchaudio

def preprocess_audio(audio_array, target_sr=16000):
    """Preprocess audio for ML models"""
    
    # Resample if needed
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Normalize
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_array).float()
    
    return audio_tensor

# Process a batch of samples
def create_audio_batch(dataset_subset, target_sr=16000):
    """Create batched audio data"""
    audios = []
    metadata = []
    
    for sample in dataset_subset:
        audio = preprocess_audio(sample['audio']['array'])
        audios.append(audio)
        
        metadata.append({
            'speaker_id': sample['speaker_id'],
            'transcript': sample['transcript'],
            'gender': sample['gender'],
            'age': sample['age'],
            'accent': sample['accent']
        })
    
    return audios, metadata

# Example usage
subset = dataset['train'].select(range(10))
audio_batch, meta_batch = create_audio_batch(subset)
```

## Streaming Large Dataset
```python
# For large datasets, use streaming
dataset_stream = load_dataset("Paranoiid/VoicePersona", streaming=True)

# Process in chunks
chunk_size = 100
for i, batch in enumerate(dataset_stream['train'].iter(batch_size=chunk_size)):
    print(f"Processing chunk {i+1}: {len(batch['audio'])} samples")
    
    # Process your batch here
    # ... your processing code ...
    
    if i >= 10:  # Process only first 10 chunks for demo
        break
```

## Export Functionality
```python
# Export subset for specific use case
def export_subset(dataset, criteria, output_path):
    """Export dataset subset based on criteria"""
    
    filtered = dataset.filter(criteria)
    
    # Save as different formats
    filtered.save_to_disk(output_path)
    
    # Or save as JSON with audio paths
    export_data = []
    for i, sample in enumerate(filtered):
        audio_path = f"{output_path}/audio_{i:06d}.wav"
        sf.write(audio_path, sample['audio']['array'], sample['audio']['sampling_rate'])
        
        export_data.append({
            'audio_path': audio_path,
            'transcript': sample['transcript'],
            'speaker_id': sample['speaker_id'],
            'voice_description': sample['voice_description']
        })
    
    with open(f"{output_path}/metadata.json", 'w') as f:
        json.dump(export_data, f, indent=2)

# Example: Export only high-quality samples
high_quality_criteria = lambda x: (
    x['duration'] > 1.0 and 
    x['duration'] < 10.0 and 
    x['gender'] != 'unknown'
)

export_subset(dataset['train'], high_quality_criteria, "high_quality_voices")
```
