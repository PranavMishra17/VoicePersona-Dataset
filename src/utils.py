"""
Utility functions for data analysis and export
"""
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

logger = logging.getLogger(__name__)

def analyze_processed_dataset(json_path: Path) -> Dict[str, Any]:
    """Analyze processed dataset and return statistics"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            logger.warning("Empty dataset")
            return {}
        
        # Basic statistics
        stats = {
            'total_samples': len(data),
            'processing_dates': [],
            'description_lengths': {
                'voice_analysis': [],
                'character_description': [],
                'combined': []
            },
            'metadata_distributions': {
                'gender': {},
                'age': {},
                'accent': {}
            }
        }
        
        # Collect statistics
        for item in data:
            # Description lengths
            stats['description_lengths']['voice_analysis'].append(
                len(item.get('voice_analysis', ''))
            )
            stats['description_lengths']['character_description'].append(
                len(item.get('character_description', ''))
            )
            stats['description_lengths']['combined'].append(
                len(item.get('combined_description', ''))
            )
            
            # Metadata distributions
            for field in ['gender', 'age', 'accent']:
                value = item.get(field, 'unknown')
                stats['metadata_distributions'][field][value] = \
                    stats['metadata_distributions'][field].get(value, 0) + 1
            
            # Processing dates
            if 'processing_timestamp' in item:
                stats['processing_dates'].append(item['processing_timestamp'])
        
        # Calculate averages
        for desc_type, lengths in stats['description_lengths'].items():
            if lengths:
                stats[f'avg_{desc_type}_length'] = np.mean(lengths)
                stats[f'std_{desc_type}_length'] = np.std(lengths)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        return {}

def export_for_training(json_path: Path, output_path: Path, 
                       format_type: str = "instruction") -> int:
    """Export data in training-ready format"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if format_type == "instruction":
            training_data = _format_instruction_tuning(data)
        elif format_type == "conversation":
            training_data = _format_conversation(data)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(training_data)} samples to {output_path}")
        return len(training_data)
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise

def _format_instruction_tuning(data: List[Dict]) -> List[Dict]:
    """Format for instruction tuning"""
    formatted = []
    for item in data:
        formatted.append({
            'instruction': "Analyze this voice recording and provide a detailed description.",
            'input': f"Speaker transcript: '{item['transcript']}'",
            'output': item['combined_description'],
            'metadata': {
                'speaker_id': item['speaker_id'],
                'accent': item['accent'],
                'age': item['age'],
                'gender': item['gender'],
                'duration': item['duration']
            }
        })
    return formatted

def _format_conversation(data: List[Dict]) -> List[Dict]:
    """Format as conversations"""
    formatted = []
    for item in data:
        formatted.append({
            'conversations': [
                {
                    'from': 'human',
                    'value': f"Please analyze this voice recording. The speaker says: '{item['transcript']}'"
                },
                {
                    'from': 'assistant',
                    'value': item['combined_description']
                }
            ],
            'metadata': {
                'speaker_id': item['speaker_id'],
                'accent': item['accent'],
                'age': item['age'],
                'gender': item['gender']
            }
        })
    return formatted

def create_visualization(stats: Dict[str, Any], output_dir: Path):
    """Create visualizations of dataset statistics"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Description length distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (desc_type, lengths) in enumerate(stats['description_lengths'].items()):
            if lengths:
                axes[idx].hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[idx].set_title(f'{desc_type.replace("_", " ").title()} Lengths')
                axes[idx].set_xlabel('Character Count')
                axes[idx].set_ylabel('Frequency')
                axes[idx].axvline(np.mean(lengths), color='red', linestyle='--', 
                                label=f'Mean: {np.mean(lengths):.0f}')
                axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'description_lengths.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metadata distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (field, dist) in enumerate(stats['metadata_distributions'].items()):
            if dist and idx < 3:
                # Sort by count
                sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
                labels, counts = zip(*sorted_items)
                
                axes[idx].bar(range(len(labels)), counts, alpha=0.7, color='lightcoral')
                axes[idx].set_title(f'{field.title()} Distribution')
                axes[idx].set_xlabel(field.title())
                axes[idx].set_ylabel('Count')
                axes[idx].set_xticks(range(len(labels)))
                axes[idx].set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metadata_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")

def sample_descriptions(json_path: Path, n_samples: int = 5) -> List[Dict]:
    """Get random sample descriptions for review"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            return []
        
        n_samples = min(n_samples, len(data))
        samples = random.sample(data, n_samples)
        
        # Format for display
        formatted_samples = []
        for sample in samples:
            formatted_samples.append({
                'speaker_id': sample['speaker_id'],
                'transcript': sample['transcript'][:100] + '...' if len(sample['transcript']) > 100 else sample['transcript'],
                'metadata': f"{sample['gender']}, {sample['age']}, {sample['accent']}",
                'voice_analysis_preview': sample['voice_analysis'][:200] + '...',
                'character_preview': sample['character_description'][:200] + '...'
            })
        
        return formatted_samples
        
    except Exception as e:
        logger.error(f"Error sampling descriptions: {str(e)}")
        return []

def validate_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Validate checkpoint integrity"""
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return {
            'valid': True,
            'processed_count': len(checkpoint.get('processed_indices', [])),
            'last_index': checkpoint.get('last_index', -1),
            'has_data': len(checkpoint.get('processed_data', [])) > 0,
            'total_time': checkpoint.get('total_time', 0)
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def estimate_remaining_time(checkpoint_data: Dict, total_samples: int) -> Dict[str, float]:
    """Estimate remaining processing time"""
    processed = len(checkpoint_data.get('processed_indices', []))
    
    if processed == 0:
        return {'remaining_samples': total_samples, 'estimated_hours': 'unknown'}
    
    total_time = checkpoint_data.get('total_time', 0)
    avg_time_per_sample = total_time / processed if processed > 0 else 0
    
    remaining = total_samples - processed
    estimated_seconds = remaining * avg_time_per_sample
    
    return {
        'remaining_samples': remaining,
        'estimated_hours': estimated_seconds / 3600,
        'avg_seconds_per_sample': avg_time_per_sample
    }