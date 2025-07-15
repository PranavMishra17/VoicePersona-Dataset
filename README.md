# GLOBE_V2 + Qwen2-Audio Voice Description Pipeline

A GPU-optimized pipeline for processing the GLOBE_V2 dataset with Qwen2-Audio to generate detailed voice descriptions and character personas.

## Features

- **GPU Optimization**: Full CUDA support with mixed precision (FP16)
- **Checkpointing**: Resume processing from interruptions
- **Comprehensive Analysis**: Voice characteristics + character personas
- **Error Handling**: Robust error recovery and logging
- **Multiple Export Formats**: JSON, HuggingFace Dataset, training formats

## Directory Structure

```
globe_v2_qwen2_audio/
├── src/
│   ├── config.py          # Configuration and settings
│   ├── prompts.py         # Voice analysis prompts
│   ├── model_manager.py   # Qwen2-Audio model management
│   ├── dataset_processor.py # Dataset processing logic
│   └── utils.py           # Utility functions
├── main.py               # Main execution script
├── requirements.txt      # Dependencies
├── checkpoints/         # Processing checkpoints
├── data/               # Temporary data
├── output/             # Processed outputs
├── logs/               # Processing logs
└── README.md          # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository>
cd globe_v2_qwen2_audio
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify GPU availability**:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### 1. Test Processing (Recommended First Step)

Test on 2 samples to verify setup:
```bash
python main.py test --samples 2
```

### 2. Process Full Dataset

Process entire dataset with automatic checkpointing:
```bash
python main.py process
```

Process specific range:
```bash
python main.py process --start 0 --end 1000
```

Start fresh (ignore checkpoint):
```bash
python main.py process --no-resume
```

### 3. Analyze Results

Generate statistics and visualizations:
```bash
python main.py analyze
```

### 4. Export for Training

Export in instruction-tuning format:
```bash
python main.py export --format instruction
```

Export in conversation format:
```bash
python main.py export --format conversation
```

## Output Files

- `output/globe_v2_with_descriptions.json` - Complete processed dataset
- `output/globe_v2_hf_dataset/` - HuggingFace Dataset format
- `output/visualizations/` - Statistical plots
- `logs/processing.log` - Detailed processing logs
- `checkpoints/checkpoint.pkl` - Resume checkpoint

## Output Format

Each processed sample contains:
```json
{
  "index": 0,
  "transcript": "Original transcript",
  "speaker_id": "speaker_001",
  "accent": "US",
  "age": "20-29",
  "gender": "female",
  "duration": 5.43,
  "voice_analysis": "Detailed voice characteristics...",
  "character_description": "Character/persona description...",
  "combined_description": "Complete analysis...",
  "processing_timestamp": "2024-01-15T10:30:00"
}
```

## GPU Memory Management

- Automatic mixed precision (FP16) for efficiency
- Periodic cache clearing every 50 samples
- Single sample processing to minimize memory
- ~10-12GB GPU memory recommended

## Error Recovery

The pipeline includes:
- Automatic checkpoint saves every 100 samples
- Retry logic for transient errors
- Detailed error logging
- GPU OOM recovery

## Processing Time Estimates

- Average: ~5-10 seconds per sample
- Full dataset (~50k samples): ~70-140 hours
- Use `python main.py analyze` to check progress

## Troubleshooting

### CUDA Out of Memory
- Reduce `MAX_NEW_TOKENS` in `src/config.py`
- Increase `CLEAR_CACHE_INTERVAL`
- Use smaller model variant

### Checkpoint Corruption
```python
# Validate checkpoint
from src.utils import validate_checkpoint
from pathlib import Path
print(validate_checkpoint(Path("checkpoints/checkpoint.pkl")))
```

### Resume from Specific Index
```bash
python main.py process --start 5000
```

## Advanced Configuration

Edit `src/config.py` for:
- Model selection
- Generation parameters
- Checkpoint intervals
- Memory management settings

## Python API Usage

```python
from src.dataset_processor import GlobeV2Processor
from src.utils import analyze_processed_dataset

# Process dataset
processor = GlobeV2Processor()
processor.load_dataset()
results = processor.process_dataset(start_idx=0, end_idx=100)
processor.save_results()

# Analyze results
stats = analyze_processed_dataset("output/globe_v2_with_descriptions.json")
```

## License

This project uses the GLOBE_V2 dataset and Qwen2-Audio model. Please refer to their respective licenses.

## Citation

If using this pipeline, please cite:
- GLOBE_V2 dataset
- Qwen2-Audio model
- This processing pipeline