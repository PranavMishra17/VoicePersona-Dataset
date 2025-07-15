"""
Main execution script for GLOBE_V2 + Qwen2-Audio processing
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import setup_logging, get_device_info, OUTPUT_JSON, OUTPUT_TRAINING_DATA
from src.dataset_processor import GlobeV2Processor
from src.utils import (
    analyze_processed_dataset, export_for_training,
    create_visualization, sample_descriptions,
    validate_checkpoint, estimate_remaining_time
)

logger = logging.getLogger(__name__)

def test_processing(n_samples: int = 2):
    """Test processing on a few samples"""
    logger.info("Starting test processing...")
    logger.info(f"Device info: {get_device_info()}")
    
    processor = GlobeV2Processor()
    processor.load_dataset()
    
    # Process test samples
    processed = processor.process_dataset(
        start_idx=0,
        end_idx=n_samples,
        resume=False
    )
    
    # Save results
    if processed:
        processor.save_results(processed)
        
        # Show samples
        logger.info("\nSample results:")
        for item in processed:
            logger.info(f"\nSpeaker: {item['speaker_id']}")
            logger.info(f"Transcript: {item['transcript'][:100]}...")
            logger.info(f"Voice Analysis (preview): {item['voice_analysis'][:200]}...")
    
    return processed

def process_full_dataset(start_idx: int = 0, end_idx: int = None, resume: bool = True):
    """Process the full dataset"""
    logger.info("Starting full dataset processing...")
    logger.info(f"Device info: {get_device_info()}")
    
    processor = GlobeV2Processor()
    processor.load_dataset()
    
    # Check checkpoint status
    if resume:
        checkpoint_info = validate_checkpoint(processor.checkpoint_data)
        if checkpoint_info['valid']:
            logger.info(f"Valid checkpoint found: {checkpoint_info['processed_count']} samples processed")
            
            # Estimate remaining time
            time_est = estimate_remaining_time(processor.checkpoint_data, len(processor.dataset))
            logger.info(f"Remaining samples: {time_est['remaining_samples']}")
            logger.info(f"Estimated time: {time_est['estimated_hours']:.2f} hours")
    
    # Process dataset
    processed = processor.process_dataset(
        start_idx=start_idx,
        end_idx=end_idx,
        resume=resume
    )
    
    # Save results
    processor.save_results()
    
    return processed

def analyze_results():
    """Analyze processed dataset"""
    if not OUTPUT_JSON.exists():
        logger.error(f"No processed data found at {OUTPUT_JSON}")
        return
    
    logger.info("Analyzing processed dataset...")
    
    # Get statistics
    stats = analyze_processed_dataset(OUTPUT_JSON)
    
    # Print summary
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total samples: {stats.get('total_samples', 0)}")
    
    for desc_type in ['voice_analysis', 'character_description']:
        avg_key = f'avg_{desc_type}_length'
        if avg_key in stats:
            logger.info(f"Average {desc_type} length: {stats[avg_key]:.0f} chars")
    
    # Create visualizations
    viz_dir = OUTPUT_JSON.parent / "visualizations"
    create_visualization(stats, viz_dir)
    
    # Show sample descriptions
    samples = sample_descriptions(OUTPUT_JSON, n_samples=3)
    logger.info("\nSample descriptions:")
    for i, sample in enumerate(samples):
        logger.info(f"\n--- Sample {i+1} ---")
        logger.info(f"Speaker: {sample['speaker_id']} ({sample['metadata']})")
        logger.info(f"Voice preview: {sample['voice_analysis_preview']}")

def export_data(format_type: str = "instruction"):
    """Export data for training"""
    if not OUTPUT_JSON.exists():
        logger.error(f"No processed data found at {OUTPUT_JSON}")
        return
    
    logger.info(f"Exporting data in {format_type} format...")
    
    output_path = OUTPUT_JSON.parent / f"globe_v2_{format_type}_format.json"
    count = export_for_training(OUTPUT_JSON, output_path, format_type)
    
    logger.info(f"Exported {count} samples to {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GLOBE_V2 + Qwen2-Audio Processing")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test on a few samples')
    test_parser.add_argument('--samples', type=int, default=2, help='Number of test samples')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process dataset')
    process_parser.add_argument('--start', type=int, default=0, help='Start index')
    process_parser.add_argument('--end', type=int, default=None, help='End index')
    process_parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoint')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze processed data')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export for training')
    export_parser.add_argument('--format', choices=['instruction', 'conversation'], 
                              default='instruction', help='Export format')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'test':
            test_processing(args.samples)
        elif args.command == 'process':
            process_full_dataset(args.start, args.end, not args.no_resume)
        elif args.command == 'analyze':
            analyze_results()
        elif args.command == 'export':
            export_data(args.format)
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()