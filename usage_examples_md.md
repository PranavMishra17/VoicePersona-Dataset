# VoicePersona Dataset Usage Guide

Comprehensive examples for using the VoicePersona dataset for voice synthesis, analysis, and research.

## 🚀 Quick Start

### Installation
```bash
pip install datasets soundfile librosa torch pandas scikit-learn
```

### Basic Loading
```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("Paranoiid/VoicePersona")
print(f"Dataset size: {len(dataset['train']):,} samples")

# Access a sample
sample = dataset['train'][0]
print(f"Speaker: {sample['speaker_id']}")
print(f"Transcript: {sample['transcript']}")
print(f"Duration: {sample['duration']:.1f}s")
print(f"Demographics: {sample['gender']}, {sample['age']}, {sample['accent']}")
```

## 🎵 Working with Audio

### Basic Audio Operations
```python
import soundfile as sf
import numpy as np

# Get audio data from sample
sample = dataset['train'][100]
audio_array = sample['audio']['array']
sampling_rate = sample['audio']['sampling_rate']

# Save to file
sf.write("voice_sample.wav", audio_array, sampling_rate)

# Basic audio info
print(f"Audio shape: {audio_array.shape}")
print(f"Duration: {len(audio_array) / sampling_rate:.2f} seconds")
print(f"Sample rate: {sampling_rate} Hz")

# Play in Jupyter notebook
import IPython.display as ipd
ipd.Audio(audio_array, rate=sampling_rate)
```

### Audio Preprocessing
```python
import librosa

def preprocess_audio(audio_array, target_sr=16000, max_duration=10.0):
    """Preprocess audio for ML models"""
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Trim silence
    audio_array, _ = librosa.effects.trim(audio_array)
    
    # Limit duration
    max_samples = int(max_duration * target_sr)
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]
    
    # Normalize
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array

# Apply preprocessing
processed_audio = preprocess_audio(audio_array)
```

## 🔍 Data Filtering and Analysis

### Demographic Filtering
```python
# Filter by gender
female_voices = dataset['train'].filter(lambda x: x['gender'] == 'female')
male_voices = dataset['train'].filter(lambda x: x['gender'] == 'male')

print(f"Female voices: {len(female_voices):,}")
print(f"Male voices: {len(male_voices):,}")

# Filter by age groups
young_voices = dataset['train'].filter(
    lambda x: x['age'] in ['teens', 'twenties']
)

# Filter by accent
american_accents = dataset['train'].filter(
    lambda x: 'American' in x.get('accent', '')
)

# Complex filtering
high_quality_samples = dataset['train'].filter(
    lambda x: (
        x['duration'] >= 1.0 and 
        x['duration'] <= 8.0 and
        x['gender'] != 'unknown' and
        len(x['transcript']) > 10
    )
)
```

### Dataset Statistics
```python
import pandas as pd
from collections import Counter

# Convert to pandas for analysis
df = dataset['train'].to_pandas()

print("📊 Dataset Overview:")
print(f"Total samples: {len(df):,}")
print(f"Unique speakers: {df['speaker_id'].nunique():,}")
print(f"Total audio hours: {df['duration'].sum()/3600:.1f}")
print(f"Average duration: {df['duration'].mean():.1f}s")

# Demographics breakdown
print("\n👥 Demographics:")
print("Gender distribution:")
print(df['gender'].value_counts())

print("\nAge distribution:")  
print(df['age'].value_counts())

print("\nSource datasets:")
print(df['dataset'].value_counts())

# Top accents
print("\n🌍 Top 10 Accents:")
print(df['accent'].value_counts().head(10))
```

## 🎯 Voice Characteristic Analysis

### Extract Voice Features
```python
import re
from collections import Counter

def analyze_voice_characteristics(descriptions):
    """Extract common voice characteristics from descriptions"""
    
    # Common voice descriptors
    pitch_terms = r'\b(high|low|medium|deep|pitch)\b'
    tone_terms = r'\b(warm|cool|bright|dark|rich|thin)\b'
    quality_terms = r'\b(smooth|raspy|clear|breathy|nasal|gravelly)\b'
    
    characteristics = {
        'pitch': [],
        'tone': [], 
        'quality': []
    }
    
    for desc in descriptions:
        desc_lower = desc.lower()
        characteristics['pitch'].extend(re.findall(pitch_terms, desc_lower))
        characteristics['tone'].extend(re.findall(tone_terms, desc_lower))
        characteristics['quality'].extend(re.findall(quality_terms, desc_lower))
    
    return {
        key: Counter(values) for key, values in characteristics.items()
    }

# Analyze voice descriptions
descriptions = df['voice_description'].tolist()
voice_stats = analyze_voice_characteristics(descriptions)

print("🎵 Voice Characteristics:")
for category, counts in voice_stats.items():
    print(f"\n{category.title()} descriptors:")
    for term, count in counts.most_common(5):
        print(f"  {term}: {count}")
```

### Voice Similarity Search
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_voices(target_description, dataset, top_k=5):
    """Find voices similar to a target description"""
    
    # Get all descriptions
    all_descriptions = [sample['voice_description'] for sample in dataset]
    search_corpus = [target_description] + all_descriptions
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(search_corpus)
    
    # Calculate similarities
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get top matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        sample = dataset[idx]
        results.append({
            'similarity': similarities[idx],
            'speaker_id': sample['speaker_id'],
            'description': sample['voice_description'],
            'demographics': f"{sample['gender']}, {sample['age']}, {sample['accent']}",
            'audio': sample['audio']
        })
    
    return results

# Example usage
target_voice = "warm female voice with clear articulation and medium pitch"
similar_voices = find_similar_voices(target_voice, dataset['train'], top_k=5)

print("🔍 Similar voices found:")
for i, voice in enumerate(similar_voices):
    print(f"{i+1}. {voice['speaker_id']} (similarity: {voice['similarity']:.3f})")
    print(f"   Demographics: {voice['demographics']}")
    print(f"   Description: {voice['description'][:100]}...")
    print()
```

## 🤖 Machine Learning Applications

### Prepare Training Data
```python
def create_voice_synthesis_dataset(dataset, max_duration=10.0):
    """Prepare dataset for voice synthesis training"""
    
    training_samples = []
    
    for sample in dataset:
        # Filter by quality criteria
        if (sample['duration'] > max_duration or 
            sample['gender'] == 'unknown' or
            len(sample['transcript']) < 5):
            continue
        
        # Create training example
        training_example = {
            # Audio data
            'audio': sample['audio']['array'],
            'sampling_rate': sample['audio']['sampling_rate'],
            
            # Text data
            'text': sample['transcript'],
            
            # Speaker characteristics
            'speaker_id': sample['speaker_id'],
            'gender': sample['gender'],
            'age': sample['age'],
            'accent': sample['accent'],
            'voice_profile': sample['voice_description'],
            
            # Metadata
            'duration': sample['duration'],
            'source_dataset': sample['dataset']
        }
        
        training_samples.append(training_example)
    
    return training_samples

# Create training dataset
train_data = create_voice_synthesis_dataset(dataset['train'])
print(f"Training samples prepared: {len(train_data):,}")
```

### Character Voice Classification
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare data for character classification
def prepare_character_classification():
    """Classify speakers by character type based on voice description"""
    
    # Define character categories based on voice descriptions
    def categorize_character(description):
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['young', 'youthful', 'teen', 'child']):
            return 'young_character'
        elif any(word in desc_lower for word in ['mature', 'adult', 'professional']):
            return 'adult_character'
        elif any(word in desc_lower for word in ['warm', 'friendly', 'gentle']):
            return 'friendly_character'
        elif any(word in desc_lower for word in ['deep', 'powerful', 'strong']):
            return 'authoritative_character'
        else:
            return 'neutral_character'
    
    # Prepare features and labels
    descriptions = []
    labels = []
    
    for sample in dataset['train']:
        descriptions.append(sample['voice_description'])
        labels.append(categorize_character(sample['voice_description']))
    
    return descriptions, labels

# Train character classifier
descriptions, character_labels = prepare_character_classification()

# Vectorize descriptions
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(descriptions)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, character_labels, test_size=0.2, random_state=42
)

# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
print("Character Classification Results:")
print(classification_report(y_test, y_pred))
```

## 📊 Data Visualization

### Create Visualizations
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

# Duration distribution
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(df['duration'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Audio Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')

# Gender distribution pie chart
plt.subplot(2, 2, 2)
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')

# Age distribution
plt.subplot(2, 2, 3)
age_counts = df['age'].value_counts()
plt.bar(age_counts.index, age_counts.values, alpha=0.7)
plt.title('Age Group Distribution')
plt.xticks(rotation=45)

# Dataset sources
plt.subplot(2, 2, 4)
dataset_counts = df['dataset'].value_counts()
plt.bar(dataset_counts.index, dataset_counts.values, alpha=0.7)
plt.title('Source Dataset Distribution')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## 💾 Export and Streaming

### Export Subsets
```python
def export_voice_subset(dataset, criteria_func, output_dir, format='wav'):
    """Export a subset of voices based on criteria"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/audio", exist_ok=True)
    
    filtered_data = []
    audio_files = []
    
    for i, sample in enumerate(dataset):
        if criteria_func(sample):
            # Save audio
            audio_filename = f"voice_{i:06d}.{format}"
            audio_path = f"{output_dir}/audio/{audio_filename}"
            
            sf.write(audio_path, sample['audio']['array'], sample['audio']['sampling_rate'])
            
            # Prepare metadata
            metadata = {
                'audio_file': audio_filename,
                'speaker_id': sample['speaker_id'],
                'transcript': sample['transcript'],
                'duration': sample['duration'],
                'gender': sample['gender'],
                'age': sample['age'],
                'accent': sample['accent'],
                'voice_description': sample['voice_description']
            }
            
            filtered_data.append(metadata)
    
    # Save metadata
    import json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Exported {len(filtered_data)} samples to {output_dir}")

# Example: Export female voices under 5 seconds
criteria = lambda x: x['gender'] == 'female' and x['duration'] < 5.0
export_voice_subset(dataset['train'], criteria, "female_short_voices")
```

### Streaming for Large Datasets
```python
# Use streaming for memory efficiency with large datasets
def process_dataset_streaming(batch_size=100):
    """Process dataset in streaming mode"""
    
    dataset_stream = load_dataset("Paranoiid/VoicePersona", streaming=True)
    
    batch_count = 0
    total_duration = 0
    
    for batch in dataset_stream['train'].iter(batch_size=batch_size):
        batch_count += 1
        batch_duration = sum(batch['duration'])
        total_duration += batch_duration
        
        print(f"Batch {batch_count}: {len(batch['audio'])} samples, "
              f"{batch_duration:.1f}s total")
        
        # Process your batch here
        # Example: Extract features, train model, etc.
        
        if batch_count >= 10:  # Limit for demo
            break
    
    print(f"Total processed: {total_duration:.1f} seconds of audio")

# Run streaming processing
process_dataset_streaming()
```

## 🎬 Creative Applications

### Voice Character Generator
```python
def generate_character_voice_profile(gender=None, age=None, personality=None):
    """Generate a character voice profile from dataset"""
    
    # Filter based on criteria
    filtered_samples = dataset['train']
    
    if gender:
        filtered_samples = filtered_samples.filter(lambda x: x['gender'] == gender)
    
    if age:
        filtered_samples = filtered_samples.filter(lambda x: x['age'] == age)
    
    if personality:
        # Search voice descriptions for personality traits
        filtered_samples = filtered_samples.filter(
            lambda x: personality.lower() in x['voice_description'].lower()
        )
    
    if len(filtered_samples) == 0:
        return "No matching voices found"
    
    # Select random sample
    import random
    sample = filtered_samples[random.randint(0, len(filtered_samples)-1)]
    
    return {
        'character_profile': {
            'gender': sample['gender'],
            'age': sample['age'], 
            'accent': sample['accent'],
            'personality': personality,
            'voice_description': sample['voice_description']
        },
        'sample_audio': sample['audio'],
        'sample_text': sample['transcript'],
        'speaker_id': sample['speaker_id']
    }

# Generate character voices
hero_voice = generate_character_voice_profile(
    gender='male', age='thirties', personality='confident'
)

narrator_voice = generate_character_voice_profile(
    gender='female', age='forties', personality='warm'
)

villain_voice = generate_character_voice_profile(
    gender='male', age='fifties', personality='deep'
)

print("🎭 Generated Character Voices:")
print(f"Hero: {hero_voice['character_profile']}")
print(f"Narrator: {narrator_voice['character_profile']}")
print(f"Villain: {villain_voice['character_profile']}")
```

### Voice Synthesis Pipeline
```python
def create_voice_synthesis_pipeline():
    """Complete pipeline for voice synthesis applications"""
    
    class VoiceSynthesisPipeline:
        def __init__(self, dataset):
            self.dataset = dataset
            self.voice_embeddings = self._create_voice_embeddings()
        
        def _create_voice_embeddings(self):
            """Create embeddings for voice characteristics"""
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            descriptions = [sample['voice_description'] for sample in self.dataset]
            vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
            embeddings = vectorizer.fit_transform(descriptions)
            
            return {
                'vectorizer': vectorizer,
                'embeddings': embeddings,
                'samples': self.dataset
            }
        
        def find_voice_by_description(self, description, top_k=3):
            """Find voices matching a text description"""
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Vectorize target description
            target_vector = self.voice_embeddings['vectorizer'].transform([description])
            
            # Calculate similarities
            similarities = cosine_similarity(
                target_vector, 
                self.voice_embeddings['embeddings']
            ).flatten()
            
            # Get top matches
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            matches = []
            for idx in top_indices:
                sample = self.voice_embeddings['samples'][idx]
                matches.append({
                    'similarity': similarities[idx],
                    'sample': sample,
                    'audio': sample['audio']
                })
            
            return matches
        
        def get_voice_by_speaker(self, speaker_id):
            """Get specific speaker's voice samples"""
            return [
                sample for sample in self.dataset 
                if sample['speaker_id'] == speaker_id
            ]
        
        def create_voice_dataset_for_training(self, speaker_ids=None, min_duration=1.0):
            """Create a focused dataset for training"""
            training_data = []
            
            for sample in self.dataset:
                # Filter criteria
                if (sample['duration'] < min_duration or 
                    sample['gender'] == 'unknown'):
                    continue
                
                if speaker_ids and sample['speaker_id'] not in speaker_ids:
                    continue
                
                training_data.append({
                    'audio': sample['audio']['array'],
                    'text': sample['transcript'],
                    'speaker_id': sample['speaker_id'],
                    'voice_characteristics': {
                        'gender': sample['gender'],
                        'age': sample['age'],
                        'accent': sample['accent'],
                        'description': sample['voice_description']
                    }
                })
            
            return training_data
    
    return VoiceSynthesisPipeline(dataset['train'])

# Create and use the pipeline
pipeline = create_voice_synthesis_pipeline()

# Find voices by description
matches = pipeline.find_voice_by_description("deep authoritative male voice")
print(f"Found {len(matches)} matching voices")

# Create training dataset for specific speakers
popular_speakers = df['speaker_id'].value_counts().head(10).index.tolist()
training_set = pipeline.create_voice_dataset_for_training(speaker_ids=popular_speakers)
print(f"Training dataset: {len(training_set)} samples")
```

## 🔬 Research Applications

### Voice Analysis Research
```python
def analyze_voice_diversity():
    """Analyze diversity in the voice dataset for research"""
    
    # Accent diversity analysis
    accent_stats = {
        'total_accents': df['accent'].nunique(),
        'accent_distribution': df['accent'].value_counts().to_dict(),
        'dominant_accents': df['accent'].value_counts().head(5).to_dict()
    }
    
    # Gender balance analysis
    gender_balance = df['gender'].value_counts(normalize=True).to_dict()
    
    # Age distribution analysis
    age_distribution = df['age'].value_counts(normalize=True).to_dict()
    
    # Duration analysis
    duration_stats = {
        'mean_duration': df['duration'].mean(),
        'median_duration': df['duration'].median(),
        'std_duration': df['duration'].std(),
        'total_hours': df['duration'].sum() / 3600
    }
    
    # Voice characteristic analysis
    descriptions = ' '.join(df['voice_description'].tolist())
    common_words = Counter(descriptions.lower().split()).most_common(20)
    
    research_report = {
        'dataset_size': len(df),
        'accent_diversity': accent_stats,
        'gender_balance': gender_balance,
        'age_distribution': age_distribution,
        'duration_statistics': duration_stats,
        'common_descriptors': dict(common_words),
        'source_datasets': df['dataset'].value_counts().to_dict()
    }
    
    return research_report

# Generate research analysis
research_data = analyze_voice_diversity()
print("🔬 Research Analysis:")
for key, value in research_data.items():
    print(f"{key}: {value}")
```

### Cross-Dataset Comparison
```python
def compare_source_datasets():
    """Compare characteristics across source datasets"""
    
    comparison = {}
    
    for source in df['dataset'].unique():
        subset = df[df['dataset'] == source]
        
        comparison[source] = {
            'sample_count': len(subset),
            'avg_duration': subset['duration'].mean(),
            'gender_distribution': subset['gender'].value_counts().to_dict(),
            'age_distribution': subset['age'].value_counts().to_dict(),
            'unique_accents': subset['accent'].nunique(),
            'top_accents': subset['accent'].value_counts().head(3).to_dict()
        }
    
    # Print comparison
    print("📊 Source Dataset Comparison:")
    for source, stats in comparison.items():
        print(f"\n{source.upper()}:")
        print(f"  Samples: {stats['sample_count']:,}")
        print(f"  Avg Duration: {stats['avg_duration']:.1f}s")
        print(f"  Unique Accents: {stats['unique_accents']}")
        print(f"  Gender Split: {stats['gender_distribution']}")
    
    return comparison

# Run comparison
dataset_comparison = compare_source_datasets()
```

## 🎯 Advanced Use Cases

### Multi-Speaker Voice Cloning Preparation
```python
def prepare_multispeaker_dataset(min_samples_per_speaker=3, max_duration=8.0):
    """Prepare dataset for multi-speaker voice cloning"""
    
    # Group by speaker
    speaker_groups = df.groupby('speaker_id')
    
    # Filter speakers with enough samples
    valid_speakers = []
    for speaker_id, group in speaker_groups:
        # Quality filter
        quality_samples = group[
            (group['duration'] <= max_duration) & 
            (group['duration'] >= 1.0) &
            (group['gender'] != 'unknown')
        ]
        
        if len(quality_samples) >= min_samples_per_speaker:
            valid_speakers.append({
                'speaker_id': speaker_id,
                'sample_count': len(quality_samples),
                'total_duration': quality_samples['duration'].sum(),
                'demographics': {
                    'gender': quality_samples['gender'].iloc[0],
                    'age': quality_samples['age'].iloc[0],
                    'accent': quality_samples['accent'].iloc[0]
                },
                'voice_profile': quality_samples['voice_description'].iloc[0]
            })
    
    print(f"Multi-speaker dataset: {len(valid_speakers)} speakers")
    print(f"Total samples: {sum(s['sample_count'] for s in valid_speakers)}")
    
    return valid_speakers

# Prepare multi-speaker dataset
multispeaker_data = prepare_multispeaker_dataset()

# Show top speakers by sample count
top_speakers = sorted(multispeaker_data, key=lambda x: x['sample_count'], reverse=True)[:10]
print("\n🎤 Top Speakers by Sample Count:")
for speaker in top_speakers:
    print(f"{speaker['speaker_id']}: {speaker['sample_count']} samples, "
          f"{speaker['demographics']['gender']}, {speaker['demographics']['age']}")
```

### Voice Quality Assessment
```python
def assess_voice_quality(sample_threshold=50):
    """Assess overall voice quality in the dataset"""
    
    quality_metrics = {
        'duration_distribution': {
            'very_short': len(df[df['duration'] < 1.0]),
            'short': len(df[(df['duration'] >= 1.0) & (df['duration'] < 3.0)]),
            'medium': len(df[(df['duration'] >= 3.0) & (df['duration'] < 8.0)]),
            'long': len(df[df['duration'] >= 8.0])
        },
        'metadata_completeness': {
            'complete_demographics': len(df[
                (df['gender'] != 'unknown') & 
                (df['age'] != 'unknown') & 
                (df['accent'] != 'unknown')
            ]),
            'missing_gender': len(df[df['gender'] == 'unknown']),
            'missing_age': len(df[df['age'] == 'unknown']),
            'missing_accent': len(df[df['accent'] == 'unknown'])
        },
        'text_quality': {
            'empty_transcripts': len(df[df['transcript'].str.len() == 0]),
            'short_transcripts': len(df[df['transcript'].str.len() < 10]),
            'avg_transcript_length': df['transcript'].str.len().mean()
        },
        'speaker_diversity': {
            'total_speakers': df['speaker_id'].nunique(),
            'speakers_with_multiple_samples': len(
                df.groupby('speaker_id').size()[
                    df.groupby('speaker_id').size() > 1
                ]
            )
        }
    }
    
    # Calculate quality score
    total_samples = len(df)
    quality_score = (
        (quality_metrics['metadata_completeness']['complete_demographics'] / total_samples) * 0.3 +
        (quality_metrics['duration_distribution']['medium'] / total_samples) * 0.3 +
        ((total_samples - quality_metrics['text_quality']['empty_transcripts']) / total_samples) * 0.2 +
        (quality_metrics['speaker_diversity']['speakers_with_multiple_samples'] / 
         quality_metrics['speaker_diversity']['total_speakers']) * 0.2
    ) * 100
    
    quality_metrics['overall_quality_score'] = quality_score
    
    print("🏆 Voice Quality Assessment:")
    print(f"Overall Quality Score: {quality_score:.1f}/100")
    print(f"Complete Demographics: {quality_metrics['metadata_completeness']['complete_demographics']:,} samples")
    print(f"Optimal Duration Samples: {quality_metrics['duration_distribution']['medium']:,} samples")
    print(f"Speaker Diversity: {quality_metrics['speaker_diversity']['total_speakers']:,} unique speakers")
    
    return quality_metrics

# Assess dataset quality
quality_assessment = assess_voice_quality()
```

## 📖 Documentation and Citation

### Dataset Information
```python
def get_dataset_citation():
    """Get proper citation for the dataset"""
    
    citation = '''
@dataset{voicepersona2025,
  title={VoicePersona Dataset: Comprehensive Voice Character Profiles for Synthesis Consistency},
  author={Pranav Mishra and Pranav Vasist},
  year={2025},
  url={https://huggingface.co/datasets/Paranoiid/VoicePersona},
  note={Training dataset for VoiceForge character voice synthesis}
}
'''
    
    return citation

def get_dataset_info():
    """Get comprehensive dataset information"""
    
    info = {
        'name': 'VoicePersona Dataset',
        'version': '1.0',
        'description': 'Comprehensive voice persona dataset for character consistency in voice synthesis',
        'total_samples': len(dataset['train']),
        'total_duration_hours': df['duration'].sum() / 3600,
        'languages': ['English', 'Japanese', 'Multiple'],
        'sample_rate': '16kHz',
        'format': 'WAV audio with text descriptions',
        'license': 'CC0 1.0 Universal',
        'source_datasets': df['dataset'].unique().tolist(),
        'applications': [
            'Voice synthesis',
            'Character voice generation', 
            'Voice cloning',
            'Speech analysis',
            'Audio-language modeling'
        ],
        'citation': get_dataset_citation()
    }
    
    return info

# Get dataset information
dataset_info = get_dataset_info()
print("📋 Dataset Information:")
for key, value in dataset_info.items():
    if key != 'citation':
        print(f"{key}: {value}")

print(f"\n📚 Citation:")
print(dataset_info['citation'])
```

## 🚀 Getting Started Template

```python
# Complete getting started template
def voice_persona_quickstart():
    """Quick start template for new users"""
    
    print("🎯 VoicePersona Dataset Quick Start")
    print("=" * 40)
    
    # 1. Load dataset
    print("1. Loading dataset...")
    dataset = load_dataset("Paranoiid/VoicePersona")
    print(f"   ✅ Loaded {len(dataset['train']):,} samples")
    
    # 2. Explore a sample
    print("\n2. Exploring sample data...")
    sample = dataset['train'][0]
    print(f"   Speaker: {sample['speaker_id']}")
    print(f"   Text: {sample['transcript'][:50]}...")
    print(f"   Voice: {sample['voice_description'][:50]}...")
    
    # 3. Basic statistics
    print("\n3. Dataset statistics...")
    df = dataset['train'].to_pandas()
    print(f"   Total duration: {df['duration'].sum()/3600:.1f} hours")
    print(f"   Unique speakers: {df['speaker_id'].nunique():,}")
    print(f"   Gender split: {dict(df['gender'].value_counts())}")
    
    # 4. Save a sample
    print("\n4. Saving sample audio...")
    sf.write("sample_voice.wav", sample['audio']['array'], sample['audio']['sampling_rate'])
    print("   ✅ Saved to sample_voice.wav")
    
    # 5. Find similar voices
    print("\n5. Finding similar voices...")
    target = "warm female voice"
    # ... similarity search code here ...
    print("   ✅ Use find_similar_voices() function above")
    
    print("\n🎉 Quick start complete! Explore the examples above for more advanced usage.")

# Run quick start
voice_persona_quickstart()
```

---

## 🔗 Related Resources

- **VoiceForge Project**: [GitHub Repository](https://github.com/PranavMishra17/VoiceForge--Forge-Character-Voices-from-Pure-Text)
- **Dataset Repository**: [HuggingFace Dataset](https://huggingface.co/datasets/Paranoiid/VoicePersona)
- **Documentation**: [Full Documentation](https://huggingface.co/datasets/Paranoiid/VoicePersona)

## 💬 Support

For questions, issues, or contributions:
- Open an issue on the VoiceForge repository
- Contact the developers through their portfolios
- Join the HuggingFace community discussions

Happy voice synthesis! 🎵