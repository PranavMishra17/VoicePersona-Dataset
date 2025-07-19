"""
Multi-dataset voice processing pipeline
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
    USE_ALTERNATIVE_MODEL, MODEL_NAME, OUTPUT_DIR,
    USE_QUANTIZATION, ENABLE_QUANTIZATION_FALLBACK
)
import config as config
from src.dataset_processor import get_processor, DATASET_PROCESSORS
from utils import analyze_processed_dataset, sample_descriptions

logger = logging.getLogger(__name__)

def show_configuration():
    """Display current configuration"""
    print("\n" + "="*50)
    print("MULTI-DATASET VOICE PROCESSING")
    print("="*50)
    print(f"Model: {MODEL_NAME}")
    print(f"Quantization: {'ENABLED' if USE_QUANTIZATION else 'DISABLED'}")
    print(f"Quantization fallback: {'ENABLED' if ENABLE_QUANTIZATION_FALLBACK else 'DISABLED'}")
    print(f"Available datasets: {list(DATASET_PROCESSORS.keys())}")
    
    disk = get_disk_space()
    print(f"\nDisk space: {disk['free_gb']:.1f} GB free")
    print("="*50 + "\n")

def run_test(dataset_name: str, n_samples: int = 2):
    """Run test processing on dataset"""
    show_configuration()
    
    if not check_resources():
        response = input("\nResource check failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    logger.info(f"Starting test processing on {dataset_name}...")
    
    # Get processor
    processor = get_processor(dataset_name, config)
    
    # Load dataset
    processor.load_dataset()
    
    # Process samples
    processed = processor.process_dataset(max_samples=n_samples, resume=False)
    
    # Save results
    processor.save_results(processed)
    
    logger.info(f"Test completed. Processed {len(processed)} samples.")

def run_processing(dataset_name: str, max_samples: int = None, resume: bool = True):
    """Run main processing on dataset"""
    show_configuration()
    
    if not check_resources():
        response = input("\nResource check failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    logger.info(f"Starting processing on {dataset_name}...")
    
    # Get processor
    processor = get_processor(dataset_name, config)
    
    # Load dataset
    processor.load_dataset()
    
    # Process dataset
    processed = processor.process_dataset(max_samples=max_samples, resume=resume)
    
    # Save results
    processor.save_results(processed)
    
    logger.info(f"Processing completed. Total processed: {len(processed)}")

def analyze_dataset(dataset_name: str):
    """Analyze processed dataset"""
    output_json = OUTPUT_DIR / dataset_name / f"{dataset_name}_descriptions.json"
    
    if not output_json.exists():
        logger.error(f"No output file found at {output_json}")
        return
    
    # Analyze
    logger.info(f"Analyzing {dataset_name} results...")
    stats = analyze_processed_dataset(output_json)
    
    logger.info(f"\nTotal samples: {stats.get('total_samples', 0)}")
    
    # Show samples
    samples = sample_descriptions(output_json, n_samples=3)
    for i, sample in enumerate(samples):
        logger.info(f"\n--- Sample {i+1} ---")
        logger.info(f"Speaker: {sample.get('speaker_id', 'unknown')}")
        logger.info(f"Dataset: {sample.get('dataset', 'unknown')}")
        logger.info(f"Description: {sample.get('voice_description', '')[:200]}...")

def list_datasets():
    """List available datasets"""
    print("\nAvailable datasets:")
    for name, processor_class in DATASET_PROCESSORS.items():
        print(f"  {name}: {processor_class.__doc__ or 'Voice dataset processor'}")
    print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Voice Processing Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test processing')
    test_parser.add_argument('dataset', choices=list(DATASET_PROCESSORS.keys()), 
                           help='Dataset to process')
    test_parser.add_argument('--samples', type=int, default=2, 
                           help='Number of test samples')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process dataset')
    process_parser.add_argument('dataset', choices=list(DATASET_PROCESSORS.keys()),
                              help='Dataset to process')
    process_parser.add_argument('--max', type=int, help='Maximum samples to process')
    process_parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('dataset', choices=list(DATASET_PROCESSORS.keys()),
                              help='Dataset to analyze')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available datasets')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'test':
            run_test(args.dataset, args.samples)
        elif args.command == 'process':
            run_processing(args.dataset, args.max, not args.no_resume)
        elif args.command == 'analyze':
            analyze_dataset(args.dataset)
        elif args.command == 'list':
            list_datasets()
        elif args.command == 'info':
            show_configuration()
        else:
            parser.print_help()
            print("\n" + "="*50)
            print("QUICK START:")
            print("="*50)
            print("1. List datasets: python main.py list")
            print("2. Test dataset: python main.py test globe_v2 --samples 2")
            print("3. Process dataset: python main.py process laions --max 100")
            print("4. Analyze results: python main.py analyze animevox")
            
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()