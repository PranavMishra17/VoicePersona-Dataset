"""
Configuration file for GLOBE_V2 + Qwen2-Audio processing pipeline
"""
import os
import torch
import logging
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
DATASET_NAME = "MushanW/GLOBE_V2"

# Processing configuration
BATCH_SIZE = 1  # Process one at a time due to model size
MAX_AUDIO_LENGTH = 30  # seconds (Qwen2-Audio performs best under 30s)
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N samples
TEST_SAMPLES = 2  # Number of samples for testing

# GPU configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_VISIBLE_DEVICES = "0"  # Which GPU to use
MIXED_PRECISION = True  # Use FP16 for faster processing

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.95
DO_SAMPLE = True

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "processing.log"

# Output file names
OUTPUT_JSON = OUTPUT_DIR / "globe_v2_with_descriptions.json"
OUTPUT_HF_DATASET = OUTPUT_DIR / "globe_v2_hf_dataset"
OUTPUT_TRAINING_DATA = OUTPUT_DIR / "globe_v2_training_format.json"
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.pkl"

# Memory management
CLEAR_CACHE_INTERVAL = 50  # Clear GPU cache every N samples

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
def get_device_info():
    """Get device information for logging"""
    if torch.cuda.is_available():
        return {
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
        }
    else:
        return {"device": "cpu", "message": "CUDA not available"}