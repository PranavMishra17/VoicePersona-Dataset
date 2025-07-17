"""
Lightweight configuration for GLOBE_V2 processing with smaller models or API options
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
CACHE_DIR = BASE_DIR / "cache"  # Custom cache directory

# Create directories
for dir_path in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to custom directory
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / "datasets")

# ========== MODEL OPTIONS ==========
# Primary: Use Qwen2-Audio without quantization (for HPC with sufficient GPU memory)
USE_QUANTIZATION = False  # Changed to False - quantization as backup only
LOAD_IN_8BIT = False  # Backup option only
LOAD_IN_4BIT = False  # Backup option only

# Backup quantization settings (used if primary loading fails)
ENABLE_QUANTIZATION_FALLBACK = True  # Set to False to disable fallback entirely
FALLBACK_4BIT = True  # Try 4-bit quantization if normal loading fails
FALLBACK_8BIT = False  # Try 8-bit quantization if 4-bit fails

# Option 2: Use alternative smaller models
USE_ALTERNATIVE_MODEL = False
ALTERNATIVE_MODELS = {
    "whisper-large": "openai/whisper-large-v3",  # 1.5B params - good for transcription
    "wav2vec2": "facebook/wav2vec2-large-960h",  # 300M params
    "speecht5": "microsoft/speecht5_asr",  # 145M params
}

# Option 3: Use API (requires API key)
USE_API = False
API_PROVIDER = "openai"  # or "anthropic", "cohere"
API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model selection
if USE_API:
    MODEL_NAME = "gpt-4-vision-preview"  # Can analyze audio spectrograms
elif USE_ALTERNATIVE_MODEL:
    MODEL_NAME = ALTERNATIVE_MODELS["whisper-large"]
else:
    MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

# ========== DATASET OPTIONS ==========
DATASET_NAME = "MushanW/GLOBE_V2"

# Streaming options (to avoid downloading entire dataset)
USE_STREAMING = True
STREAM_BATCH_SIZE = 10  # Process N samples at a time

# Subset options
USE_SUBSET = True
SUBSET_SIZE = 100  # Only process first N samples
SUBSET_SPLIT = f"train[:{SUBSET_SIZE}]" if USE_SUBSET else "train"

# Sampling options
USE_SAMPLING = False
SAMPLE_RATE = 0.01  # Sample 1% of data

# ========== PROCESSING OPTIONS ==========
BATCH_SIZE = 1
MAX_AUDIO_LENGTH = 30  # seconds
CHECKPOINT_INTERVAL = 10  # Save more frequently for testing
TEST_SAMPLES = 2

# GPU configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_VISIBLE_DEVICES = "0"
MIXED_PRECISION = True

# Generation parameters (for Qwen2-Audio)
MAX_NEW_TOKENS = 256  # Reduced from 512
TEMPERATURE = 0.7
TOP_P = 0.95
DO_SAMPLE = True

# Memory management
CLEAR_CACHE_INTERVAL = 5  # Clear cache more frequently
MAX_MEMORY_MB = 4000  # Maximum memory usage in MB

# ========== OUTPUT OPTIONS ==========
# Compressed output to save space
COMPRESS_OUTPUT = True
OUTPUT_FORMAT = "jsonl"  # More efficient than JSON

# File paths
OUTPUT_JSON = OUTPUT_DIR / f"globe_v2_descriptions_{SUBSET_SIZE if USE_SUBSET else 'full'}.{OUTPUT_FORMAT}"
OUTPUT_HF_DATASET = OUTPUT_DIR / "globe_v2_hf_dataset_lite"
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint_lite.pkl"

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "processing_lite.log"

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

def get_disk_space():
    """Get available disk space"""
    import shutil
    stat = shutil.disk_usage(BASE_DIR)
    return {
        "total_gb": stat.total / (1024**3),
        "used_gb": stat.used / (1024**3),
        "free_gb": stat.free / (1024**3),
        "free_percent": (stat.free / stat.total) * 100
    }

def estimate_model_size():
    """Estimate model size based on configuration"""
    base_sizes = {
        "Qwen/Qwen2-Audio-7B-Instruct": 14_000,  # MB
        "openai/whisper-large-v3": 3_000,
        "facebook/wav2vec2-large-960h": 1_200,
        "microsoft/speecht5_asr": 600,
    }
    
    size = base_sizes.get(MODEL_NAME, 10_000)
    
    if USE_QUANTIZATION:
        if LOAD_IN_8BIT:
            size = size * 0.5
        elif LOAD_IN_4BIT:
            size = size * 0.25
        
    return size

def check_resources():
    """Check if system has enough resources"""
    disk = get_disk_space()
    model_size_mb = estimate_model_size()
    
    print(f"\n=== Resource Check ===")
    print(f"Available disk space: {disk['free_gb']:.1f} GB")
    print(f"Estimated model size: {model_size_mb/1000:.1f} GB")
    print(f"Dataset streaming: {'ON' if USE_STREAMING else 'OFF'}")
    print(f"Quantization: {'OFF' if not USE_QUANTIZATION else 'ON'}")
    print(f"Quantization fallback: {'ON' if ENABLE_QUANTIZATION_FALLBACK else 'OFF'}")
    
    if disk['free_gb'] < (model_size_mb/1000 + 5):  # Need 5GB buffer
        print("WARNING: Low disk space! Consider:")
        print("- Set USE_STREAMING = True")
        print("- Set USE_SUBSET = True with smaller SUBSET_SIZE")
        print("- Set ENABLE_QUANTIZATION_FALLBACK = True")
        print("- Use USE_API = True with an API key")
        return False
    
    return True