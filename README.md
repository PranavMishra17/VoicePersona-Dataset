# VoicePersona Dataset generation Pipeline with Qwen2-Audio-7B 

A GPU-optimized pipeline for generating detailed voice descriptions using Qwen2-Audio-7B model. Designed to work on consumer GPUs (6GB VRAM) through quantization and streaming.

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This pipeline processes speech audio from GLOBE_V2 dataset and generates:
- **Voice characteristics**: Tone, pitch, timbre, speaking style
- **Character personas**: What type of character would have this voice
- **Technical analysis**: Acoustic features and patterns

### Key Features
- **4-bit Quantization**: Runs 7B model on 6GB VRAM (RTX 3060)
- **Streaming Mode**: Process without downloading 70GB dataset
- **Checkpoint System**: Resume from interruptions
- **Multiple Fallbacks**: CPU offloading, alternative models, API options

## 💻 System Requirements

### Minimum (with optimizations)
- **GPU**: 6GB VRAM (RTX 3060 or better)
- **RAM**: 8GB (16GB recommended)
- **Disk**: 20GB free space
- **CUDA**: 11.8+ with cuDNN

### What Happens With Limited Resources?
1. **Low VRAM (<6GB)**: Model automatically offloads layers to CPU
2. **Low Disk Space**: Streaming mode processes data without full download
3. **No GPU**: Falls back to CPU mode (very slow)

## 📁 Project Structure

```
globe_v2_qwen2_audio/
├── src/
│   ├── config.py              # All settings and paths
│   ├── model_manager.py       # Handles model loading with quantization
│   ├── dataset_processor.py   # Processes audio samples with checkpointing
│   ├── prompts.py            # Voice analysis prompt templates
│   └── utils.py              # Analysis and visualization tools
├── main.py                   # Entry point with CLI
├── requirements.txt          # Dependencies (includes bitsandbytes)
│
├── cache/                    # Model downloads (~3.5GB with 4-bit)
├── checkpoints/             # Resume points (saves every 100 samples)
├── data/                    # Temporary audio files
├── output/                  # Results in JSON/JSONL format
└── logs/                    # Processing logs
```

### File Descriptions

#### `src/config.py`
- **Purpose**: Central configuration for all settings
- **Key settings**:
  - `LOAD_IN_4BIT = True`: Enables 4-bit quantization (3.5GB instead of 14GB)
  - `USE_STREAMING = True`: Stream dataset to save disk space
  - `CHECKPOINT_INTERVAL = 100`: Auto-save frequency
  - `MAX_AUDIO_LENGTH = 30`: Clip audio to 30 seconds

#### `src/model_manager.py`
- **Purpose**: Handles Qwen2-Audio model with GPU optimization
- **Features**:
  - 4-bit quantization using BitsAndBytes
  - Automatic CPU offloading when VRAM is full
  - Mixed precision (FP16) for faster processing
  - Memory monitoring and cleanup

#### `src/dataset_processor.py`
- **Purpose**: Processes GLOBE_V2 dataset samples
- **Features**:
  - Streaming mode to avoid 70GB download
  - Checkpoint system for resuming
  - Saves results incrementally to JSONL
  - GPU memory management

#### `src/prompts.py`
- **Purpose**: Templates for voice analysis
- **Includes**:
  - Voice characteristics prompt
  - Character/persona prompt
  - Technical analysis prompt
  - Combined comprehensive prompt

## 🔧 How It Works

### 1. Model Loading Process
```python
# What happens when you run the code:
1. Loads Qwen2-Audio-7B-Instruct model
2. Applies 4-bit quantization (reduces 14GB → 3.5GB)
3. Splits model between GPU (5GB) and CPU (remaining)
4. Uses BitsAndBytes for efficient quantization
```

### 2. Dataset Loading
```python
# Two modes available:
STREAMING MODE (default):
- Downloads samples on-demand
- No full 70GB download needed
- Processes one at a time

FULL MODE:
- Downloads entire dataset
- Requires 70GB+ disk space
- Faster batch processing
```

### 3. Processing Pipeline
```
Audio Sample → Temporary WAV → Qwen2-Audio → Voice Description → Save to JSON
     ↓                                              ↓
  30s max clip                              Checkpoint every 100
```

### 4. Quantization Explained
Quantization reduces model precision to save memory:
- **Original**: 32-bit floats (14GB)
- **8-bit**: Integer approximation (7GB) - still too large
- **4-bit**: Further compressed (3.5GB) - fits in 6GB VRAM
- Trade-off: Slightly lower quality for 75% memory savings

## 🚀 Installation

```bash
# Clone repository
git clone <repository>
cd globe_v2_qwen2_audio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (includes bitsandbytes for quantization)
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📖 Usage

### Quick Start
```bash
# 1. Test on 2 samples (recommended first)
python main.py test --samples 2

# 2. Process 100 samples
python main.py process --start 0 --end 100

# 3. Resume from checkpoint
python main.py process  # Automatically resumes

# 4. Analyze results
python main.py analyze
```

### Output Format
Results saved as JSON with structure:
```json
{
  "index": 0,
  "speaker_id": "speaker_001",
  "transcript": "Hello, how are you today?",
  "accent": "US",
  "age": "20-29",
  "gender": "female",
  "duration": 3.45,
  "voice_analysis": "The speaker has a warm, mid-range voice...",
  "character_description": "This voice would suit a friendly narrator...",
  "combined_description": "Full analysis combining both aspects..."
}
```

## ⚙️ Configuration

### Key Settings in `src/config.py`

```python
# Model Settings
LOAD_IN_4BIT = True         # Must be True for 6GB VRAM
MAX_MEMORY = {0: "5GB", "cpu": "16GB"}  # GPU/CPU split

# Dataset Settings  
USE_STREAMING = True        # Stream to save disk space
USE_SUBSET = True          # Process subset only
SUBSET_SIZE = 1000         # Number of samples

# Processing Settings
CHECKPOINT_INTERVAL = 100   # Save progress frequency
CLEAR_CACHE_INTERVAL = 50   # GPU memory cleanup
```

### Model Storage Locations
- **Model cache**: `./cache/` or `~/.cache/huggingface/`
- **First run**: Downloads ~3.5GB model files
- **Subsequent runs**: Loads from cache

### Data Flow
1. **Streaming**: Downloads audio samples individually
2. **Processing**: Saves to temporary WAV file
3. **Analysis**: Sends through Qwen2-Audio
4. **Output**: Appends to JSONL file
5. **Cleanup**: Removes temporary files

## 🔨 Troubleshooting

### "CUDA out of memory"
```python
# In config.py, try:
MAX_MEMORY = {0: "4GB", "cpu": "20GB"}  # Use less GPU
```

### "Some modules are dispatched on CPU"
This is normal! The model is split between GPU/CPU. Add to config:
```python
llm_int8_enable_fp32_cpu_offload = True  # Already set
```

### "No space left on device"
1. Enable streaming: `USE_STREAMING = True`
2. Clear cache: `rm -rf ./cache/*`
3. Process smaller subset: `SUBSET_SIZE = 100`

### Slow Processing
- Normal speed: ~10-15 seconds per sample
- If slower: Check if using CPU fallback
- Solution: Ensure CUDA is properly installed

### Alternative Fallbacks
If Qwen2-Audio doesn't work:
1. **Whisper mode**: Set `USE_ALTERNATIVE_MODEL = True`
2. **API mode**: Set `USE_API = True` with OpenAI key
3. **CPU only**: Set `DEVICE = "cpu"` (very slow)

## 📊 Performance Metrics

| Configuration | VRAM | Speed | Quality |
|--------------|------|-------|---------|
| 4-bit quantization | 3.5GB | 10-15s/sample | Good |
| 8-bit quantization | 7GB | 8-12s/sample | Better |
| Full precision | 14GB | 5-10s/sample | Best |
| CPU mode | 0GB | 60-120s/sample | Good |

## 🛠️ Advanced Usage

### Process Specific Accents
```python
# In dataset_processor.py, add filter:
dataset = dataset.filter(lambda x: x['accent'] == 'US')
```

### Change Analysis Focus
```python
# In prompts.py, modify prompts for different analysis:
- Emotion detection focus
- accent analysis focus  
- Speech impediment detection
```

### Export for Training
```bash
python main.py export --format instruction
```

## 📝 Notes

- First run downloads 3.5GB model (one-time)
- Each audio sample generates ~2-3KB of description
- Full GLOBE_V2 has ~50k samples = ~150MB output
- Checkpoint allows resuming anytime
- Results append to existing files (safe for interruptions)

## 🤝 Contributing

Contributions welcome for:
- Memory optimization techniques
- Alternative model support
- Better voice analysis features
- Multi-GPU support
