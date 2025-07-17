"""
Lightweight main script for GLOBE_V2 processing
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    setup_logging, check_resources, get_disk_space,
    USE_STREAMING, USE_SUBSET, SUBSET_SIZE, USE_API,
    USE_ALTERNATIVE_MODEL, MODEL_NAME, OUTPUT_JSON,
    USE_QUANTIZATION, ENABLE_QUANTIZATION_FALLBACK
)
import config as config
from dataset_processor import GlobeV2Processor
from utils import analyze_processed_dataset, sample_descriptions

# Import or define test_processing
from dataset_processor import test_processing

logger = logging.getLogger(__name__)

def show_configuration():
    """Display current configuration"""
    print("\n" + "="*50)
    print("CURRENT CONFIGURATION")
    print("="*50)
    print(f"Model: {MODEL_NAME}")
    print(f"Streaming: {USE_STREAMING}")
    print(f"Subset: {USE_SUBSET} (size: {SUBSET_SIZE if USE_SUBSET else 'full'})")
    print(f"Using API: {USE_API}")
    print(f"Alternative model: {USE_ALTERNATIVE_MODEL}")
    print(f"Quantization: {'ENABLED' if USE_QUANTIZATION else 'DISABLED'}")
    print(f"Quantization fallback: {'ENABLED' if ENABLE_QUANTIZATION_FALLBACK else 'DISABLED'}")
    if ENABLE_QUANTIZATION_FALLBACK:
        print(f"  - 4-bit fallback: {config.FALLBACK_4BIT}")
        print(f"  - 8-bit fallback: {config.FALLBACK_8BIT}")
    
    disk = get_disk_space()
    print(f"\nDisk space: {disk['free_gb']:.1f} GB free")
    print("="*50 + "\n")

def run_test(n_samples: int = 2):
    """Run test processing"""
    show_configuration()
    
    if not check_resources():
        response = input("\nResource check failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    logger.info("Starting test processing...")
    processed = test_processing(config, n_samples)
    logger.info(f"Test completed. Processed {processed} samples.")

def run_processing(max_samples: int = None, resume: bool = True):
    """Run main processing"""
    show_configuration()
    
    if not check_resources():
        response = input("\nResource check failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    processor = GlobeV2Processor(config)
    
    # Load dataset
    processor.load_dataset()
    
    # Determine max samples
    if max_samples is None and USE_SUBSET:
        max_samples = SUBSET_SIZE
    
    logger.info(f"Starting processing (max samples: {max_samples or 'unlimited'})...")
    
    # Process
    processed = processor.process_dataset(start_idx=0, end_idx=max_samples, resume=resume)
    
    # Save results
    processor.save_results()
    
    logger.info(f"Processing completed. Total processed: {len(processed)}")

def analyze():
    """Analyze processed data"""
    if not OUTPUT_JSON.exists():
        logger.error(f"No output file found at {OUTPUT_JSON}")
        return
    
    # Handle both JSON and JSONL formats
    import jsonlines
    import json
    
    data = []
    try:
        # Try JSONL first
        with jsonlines.open(OUTPUT_JSON) as reader:
            for obj in reader:
                data.append(obj)
    except:
        # Fall back to JSON
        try:
            with open(OUTPUT_JSON, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load output file: {e}")
            return
    
    if not data:
        logger.error("No data found in output file")
        return
    
    # Save as JSON for analysis functions if needed
    temp_json = OUTPUT_JSON.with_suffix('.json')
    with open(temp_json, 'w') as f:
        json.dump(data, f)
    
    # Analyze
    logger.info("Analyzing results...")
    stats = analyze_processed_dataset(temp_json)
    
    logger.info(f"\nTotal samples: {stats.get('total_samples', 0)}")
    
    # Show samples
    samples = sample_descriptions(temp_json, n_samples=3)
    for i, sample in enumerate(samples):
        logger.info(f"\n--- Sample {i+1} ---")
        logger.info(f"Speaker: {sample['speaker_id']}")
        logger.info(f"Description: {sample.get('voice_analysis_preview', sample.get('description', ''))[:200]}")

def configure():
    """Interactive configuration"""
    print("\n" + "="*50)
    print("CONFIGURATION WIZARD")
    print("="*50)
    
    print("\n1. Model Selection:")
    print("   a) Qwen2-Audio (no quantization by default)")
    print("   b) Qwen2-Audio with quantization enabled")
    print("   c) Whisper + analysis (1-2 GB)")
    print("   d) API-based (no local model)")
    
    choice = input("\nSelect option (a/b/c/d): ").lower()
    
    if choice == 'a':
        config_updates = {
            "USE_ALTERNATIVE_MODEL": False,
            "USE_API": False,
            "USE_QUANTIZATION": False,
            "ENABLE_QUANTIZATION_FALLBACK": True
        }
    elif choice == 'b':
        print("\nQuantization level:")
        print("   1) 8-bit (better quality, ~7GB)")
        print("   2) 4-bit (lower quality, ~3.5GB)")
        quant = input("Select (1/2): ")
        
        config_updates = {
            "USE_ALTERNATIVE_MODEL": False,
            "USE_API": False,
            "USE_QUANTIZATION": True,
            "LOAD_IN_8BIT": quant == '1',
            "LOAD_IN_4BIT": quant == '2'
        }
    elif choice == 'c':
        config_updates = {
            "USE_ALTERNATIVE_MODEL": True,
            "USE_API": False
        }
    else:
        api_key = input("Enter your API key: ")
        config_updates = {
            "USE_API": True,
            "API_KEY": api_key
        }
    
    print("\n2. Dataset Options:")
    use_streaming = input("Use streaming? (y/n): ").lower() == 'y'
    
    if use_streaming:
        subset_size = input("Process how many samples? (enter number or 'all'): ")
        config_updates["USE_STREAMING"] = True
        if subset_size != 'all':
            config_updates["USE_SUBSET"] = True
            config_updates["SUBSET_SIZE"] = int(subset_size)
    
    # Save configuration
    print("\nConfiguration summary:")
    for k, v in config_updates.items():
        print(f"  {k}: {v}")
    
    save = input("\nSave configuration? (y/n): ").lower() == 'y'
    if save:
        print("Configuration saved! (Note: In real implementation, would update config.py)")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GLOBE_V2 Voice Description Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Commands
    test_parser = subparsers.add_parser('test', help='Test processing')
    test_parser.add_argument('--samples', type=int, default=2, help='Number of test samples')
    
    process_parser = subparsers.add_parser('process', help='Process dataset')
    process_parser.add_argument('--max', type=int, help='Maximum samples to process')
    process_parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    
    config_parser = subparsers.add_parser('configure', help='Interactive configuration')
    
    info_parser = subparsers.add_parser('info', help='Show current configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'test':
            run_test(args.samples)
        elif args.command == 'process':
            run_processing(args.max, not args.no_resume)
        elif args.command == 'analyze':
            analyze()
        elif args.command == 'configure':
            configure()
        elif args.command == 'info':
            show_configuration()
        else:
            parser.print_help()
            print("\n" + "="*50)
            print("QUICK START:")
            print("="*50)
            print("1. Check configuration: python main.py info")
            print("2. Run test: python main.py test")
            print("3. Process dataset: python main.py process")
            print("4. Analyze results: python main.py analyze")
            
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()