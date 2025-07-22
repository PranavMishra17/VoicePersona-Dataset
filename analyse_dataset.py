#!/usr/bin/env python3
"""
Comprehensive VoicePersona Dataset Analysis
Generates extensive visualizations and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datasets import load_from_disk
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class VoicePersonaAnalyzer:
    def __init__(self, dataset_path="voicepersona_combined"):
        self.dataset_path = dataset_path
        self.output_dir = Path("analyse")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        if Path(dataset_path).exists():
            self.dataset = load_from_disk(dataset_path)
            self.df = self.dataset.to_pandas()
        else:
            print(f"Dataset not found at {dataset_path}")
            return
            
        print(f"Loaded {len(self.df)} samples")
        
    def basic_statistics(self):
        """Generate basic dataset statistics"""
        print("\n📊 BASIC STATISTICS")
        print("=" * 50)
        
        stats = {
            'total_samples': len(self.df),
            'unique_speakers': self.df['speaker_id'].nunique(),
            'total_duration_hours': self.df['duration'].sum() / 3600,
            'avg_duration_seconds': self.df['duration'].mean(),
            'datasets': self.df['dataset'].value_counts().to_dict(),
            'genders': self.df['gender'].value_counts().to_dict(),
            'age_groups': self.df['age'].value_counts().to_dict(),
            'unique_accents': self.df['accent'].nunique(),
            'top_accents': self.df['accent'].value_counts().head(10).to_dict()
        }
        
        # Save statistics
        with open(self.output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        # Print key stats
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Unique Speakers: {stats['unique_speakers']:,}")
        print(f"Total Duration: {stats['total_duration_hours']:.1f} hours")
        print(f"Average Duration: {stats['avg_duration_seconds']:.1f} seconds")
        print(f"Unique Accents: {stats['unique_accents']}")
        
        return stats
    
    def plot_duration_analysis(self):
        """Audio duration analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Audio Duration Analysis', fontsize=16, fontweight='bold')
        
        # Duration histogram
        axes[0,0].hist(self.df['duration'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Duration Distribution')
        axes[0,0].set_xlabel('Duration (seconds)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.df['duration'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.df["duration"].mean():.1f}s')
        axes[0,0].legend()
        
        # Duration by dataset
        sns.boxplot(data=self.df, x='dataset', y='duration', ax=axes[0,1])
        axes[0,1].set_title('Duration by Dataset')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Cumulative duration
        sorted_durations = np.sort(self.df['duration'])
        cumulative = np.cumsum(sorted_durations) / 3600  # Convert to hours
        axes[1,0].plot(range(len(cumulative)), cumulative)
        axes[1,0].set_title('Cumulative Duration')
        axes[1,0].set_xlabel('Sample Index')
        axes[1,0].set_ylabel('Cumulative Hours')
        
        # Duration percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        duration_percentiles = [np.percentile(self.df['duration'], p) for p in percentiles]
        axes[1,1].bar(range(len(percentiles)), duration_percentiles, 
                      color='lightcoral', alpha=0.7)
        axes[1,1].set_title('Duration Percentiles')
        axes[1,1].set_xlabel('Percentile')
        axes[1,1].set_ylabel('Duration (seconds)')
        axes[1,1].set_xticks(range(len(percentiles)))
        axes[1,1].set_xticklabels([f'{p}th' for p in percentiles])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_demographic_analysis(self):
        """Gender and age analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Demographic Analysis', fontsize=16, fontweight='bold')
        
        # Gender pie chart
        gender_counts = self.df['gender'].value_counts()
        colors = ['lightblue', 'lightpink', 'lightgray']
        axes[0,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                      colors=colors[:len(gender_counts)], startangle=90)
        axes[0,0].set_title('Gender Distribution')
        
        # Age distribution
        age_counts = self.df['age'].value_counts()
        axes[0,1].bar(age_counts.index, age_counts.values, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Age Distribution')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylabel('Count')
        
        # Gender by dataset
        gender_dataset = pd.crosstab(self.df['dataset'], self.df['gender'])
        gender_dataset.plot(kind='bar', stacked=True, ax=axes[1,0], 
                           color=['lightblue', 'lightpink', 'lightgray'])
        axes[1,0].set_title('Gender by Dataset')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Gender')
        
        # Age by gender heatmap
        age_gender = pd.crosstab(self.df['age'], self.df['gender'])
        sns.heatmap(age_gender, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('Age vs Gender Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accent_analysis(self):
        """Accent distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Accent Analysis', fontsize=16, fontweight='bold')
        
        # Top 20 accents
        top_accents = self.df['accent'].value_counts().head(20)
        axes[0,0].barh(range(len(top_accents)), top_accents.values, color='coral', alpha=0.7)
        axes[0,0].set_yticks(range(len(top_accents)))
        axes[0,0].set_yticklabels(top_accents.index, fontsize=8)
        axes[0,0].set_title('Top 20 Accents')
        axes[0,0].set_xlabel('Count')
        
        # Accent diversity by dataset
        accent_diversity = self.df.groupby('dataset')['accent'].nunique().sort_values(ascending=False)
        axes[0,1].bar(accent_diversity.index, accent_diversity.values, color='lightsteelblue', alpha=0.7)
        axes[0,1].set_title('Accent Diversity by Dataset')
        axes[0,1].set_ylabel('Unique Accents')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Regional accent groups
        regional_groups = self.categorize_accents()
        regional_counts = pd.Series(regional_groups).value_counts()
        axes[1,0].pie(regional_counts.values, labels=regional_counts.index, autopct='%1.1f%%',
                      startangle=90)
        axes[1,0].set_title('Regional Accent Groups')
        
        # Accent frequency distribution
        accent_freq = self.df['accent'].value_counts()
        axes[1,1].hist(accent_freq.values, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1,1].set_title('Accent Frequency Distribution')
        axes[1,1].set_xlabel('Samples per Accent')
        axes[1,1].set_ylabel('Number of Accents')
        axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accent_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def categorize_accents(self):
        """Categorize accents by region"""
        accent_map = {}
        for accent in self.df['accent'].unique():
            accent_lower = accent.lower()
            if any(term in accent_lower for term in ['american', 'us', 'california', 'texas', 'new york']):
                accent_map[accent] = 'North American'
            elif any(term in accent_lower for term in ['british', 'english', 'scottish', 'welsh', 'irish']):
                accent_map[accent] = 'British Isles'
            elif any(term in accent_lower for term in ['australian', 'new zealand']):
                accent_map[accent] = 'Oceanic'
            elif any(term in accent_lower for term in ['indian', 'south asian']):
                accent_map[accent] = 'South Asian'
            elif any(term in accent_lower for term in ['african', 'south african']):
                accent_map[accent] = 'African'
            elif any(term in accent_lower for term in ['japanese', 'chinese', 'korean', 'asian']):
                accent_map[accent] = 'East Asian'
            elif any(term in accent_lower for term in ['european', 'german', 'french', 'spanish', 'italian']):
                accent_map[accent] = 'European'
            else:
                accent_map[accent] = 'Other'
        
        return [accent_map[accent] for accent in self.df['accent']]
    
    def plot_dataset_analysis(self):
        """Dataset source analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Source Analysis', fontsize=16, fontweight='bold')
        
        # Dataset distribution pie chart
        dataset_counts = self.df['dataset'].value_counts()
        axes[0,0].pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%',
                      startangle=90)
        axes[0,0].set_title('Dataset Distribution')
        
        # Speakers per dataset
        speakers_per_dataset = self.df.groupby('dataset')['speaker_id'].nunique()
        axes[0,1].bar(speakers_per_dataset.index, speakers_per_dataset.values, 
                      color='lightseagreen', alpha=0.7)
        axes[0,1].set_title('Unique Speakers per Dataset')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylabel('Unique Speakers')
        
        # Average duration by dataset
        avg_duration = self.df.groupby('dataset')['duration'].mean()
        axes[1,0].bar(avg_duration.index, avg_duration.values, color='plum', alpha=0.7)
        axes[1,0].set_title('Average Duration by Dataset')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylabel('Average Duration (seconds)')
        
        # Total duration by dataset
        total_duration = self.df.groupby('dataset')['duration'].sum() / 3600  # Convert to hours
        axes[1,1].bar(total_duration.index, total_duration.values, color='khaki', alpha=0.7)
        axes[1,1].set_title('Total Duration by Dataset (Hours)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel('Total Hours')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_text_analysis(self):
        """Text content analysis"""
        # Calculate text metrics
        self.df['transcript_length'] = self.df['transcript'].str.len()
        self.df['word_count'] = self.df['transcript'].str.split().str.len()
        self.df['description_length'] = self.df['voice_description'].str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text Content Analysis', fontsize=16, fontweight='bold')
        
        # Transcript length distribution
        axes[0,0].hist(self.df['transcript_length'], bins=50, alpha=0.7, color='mediumpurple', edgecolor='black')
        axes[0,0].set_title('Transcript Length Distribution')
        axes[0,0].set_xlabel('Character Count')
        axes[0,0].set_ylabel('Frequency')
        
        # Word count distribution
        axes[0,1].hist(self.df['word_count'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0,1].set_title('Word Count Distribution')
        axes[0,1].set_xlabel('Word Count')
        axes[0,1].set_ylabel('Frequency')
        
        # Voice description length
        axes[1,0].hist(self.df['description_length'], bins=50, alpha=0.7, color='cyan', edgecolor='black')
        axes[1,0].set_title('Voice Description Length')
        axes[1,0].set_xlabel('Character Count')
        axes[1,0].set_ylabel('Frequency')
        
        # Words per second (speaking rate)
        self.df['speaking_rate'] = self.df['word_count'] / self.df['duration'] * 60  # WPM
        axes[1,1].hist(self.df['speaking_rate'].replace([np.inf, -np.inf], np.nan).dropna(), 
                       bins=50, alpha=0.7, color='lightgray', edgecolor='black')
        axes[1,1].set_title('Speaking Rate (Words per Minute)')
        axes[1,1].set_xlabel('WPM')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'text_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_analysis(self):
        """Correlation analysis between numerical features"""
        numerical_cols = ['duration', 'transcript_length', 'word_count', 'description_length', 'speaking_rate']
        correlation_data = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_metrics(self):
        """Dataset quality and completeness metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Quality Metrics', fontsize=16, fontweight='bold')
        
        # Missing values
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            axes[0,0].bar(missing_data.index, missing_data.values, color='red', alpha=0.7)
            axes[0,0].set_title('Missing Values by Column')
            axes[0,0].tick_params(axis='x', rotation=45)
        else:
            axes[0,0].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                          fontsize=14, color='green', weight='bold')
            axes[0,0].set_title('Missing Values by Column')
        
        # Unknown/empty values
        unknown_counts = {}
        for col in ['gender', 'age', 'accent']:
            unknown_counts[col] = (self.df[col] == 'unknown').sum()
        
        axes[0,1].bar(unknown_counts.keys(), unknown_counts.values(), color='orange', alpha=0.7)
        axes[0,1].set_title('Unknown Values Count')
        axes[0,1].set_ylabel('Count')
        
        # Duration outliers
        Q1 = self.df['duration'].quantile(0.25)
        Q3 = self.df['duration'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df['duration'] < lower_bound) | (self.df['duration'] > upper_bound)]
        
        axes[1,0].scatter(range(len(self.df)), self.df['duration'], alpha=0.5, s=1)
        axes[1,0].axhline(upper_bound, color='red', linestyle='--', label=f'Upper bound: {upper_bound:.1f}s')
        axes[1,0].axhline(lower_bound, color='red', linestyle='--', label=f'Lower bound: {lower_bound:.1f}s')
        axes[1,0].set_title(f'Duration Outliers ({len(outliers)} outliers)')
        axes[1,0].set_xlabel('Sample Index')
        axes[1,0].set_ylabel('Duration (seconds)')
        axes[1,0].legend()
        
        # Data completeness score
        completeness_scores = {}
        for col in ['gender', 'age', 'accent', 'transcript', 'voice_description']:
            if col in ['gender', 'age', 'accent']:
                completeness_scores[col] = ((self.df[col] != 'unknown').sum() / len(self.df)) * 100
            else:
                completeness_scores[col] = ((self.df[col].notna() & (self.df[col] != '')).sum() / len(self.df)) * 100
        
        axes[1,1].bar(completeness_scores.keys(), completeness_scores.values(), 
                      color='green', alpha=0.7)
        axes[1,1].set_title('Data Completeness (%)')
        axes[1,1].set_ylabel('Completeness %')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        stats = self.basic_statistics()
        
        print("\n🎯 Generating comprehensive analysis...")
        
        # Generate all visualizations
        self.plot_duration_analysis()
        print("✅ Duration analysis complete")
        
        self.plot_demographic_analysis()
        print("✅ Demographic analysis complete")
        
        self.plot_accent_analysis()
        print("✅ Accent analysis complete")
        
        self.plot_dataset_analysis()
        print("✅ Dataset analysis complete")
        
        self.plot_text_analysis()
        print("✅ Text analysis complete")
        
        self.plot_correlation_analysis()
        print("✅ Correlation analysis complete")
        
        self.plot_quality_metrics()
        print("✅ Quality metrics complete")
        
        # Create summary report
        self.create_summary_report(stats)
        print("✅ Summary report generated")
        
        print(f"\n🎉 Analysis complete! All files saved to: {self.output_dir}")
        print(f"📊 Generated {len(list(self.output_dir.glob('*.png')))} visualization files")
    
    def create_summary_report(self, stats):
        """Create text summary report"""
        report = f"""
VoicePersona Dataset Analysis Report
====================================

Dataset Overview:
- Total Samples: {stats['total_samples']:,}
- Unique Speakers: {stats['unique_speakers']:,}
- Total Duration: {stats['total_duration_hours']:.1f} hours
- Average Duration: {stats['avg_duration_seconds']:.1f} seconds
- Unique Accents: {stats['unique_accents']}

Dataset Distribution:
{chr(10).join([f"- {k}: {v:,} samples" for k, v in stats['datasets'].items()])}

Gender Distribution:
{chr(10).join([f"- {k}: {v:,} samples" for k, v in stats['genders'].items()])}

Age Distribution:
{chr(10).join([f"- {k}: {v:,} samples" for k, v in stats['age_groups'].items()])}

Top 10 Accents:
{chr(10).join([f"- {k}: {v:,} samples" for k, v in stats['top_accents'].items()])}

Text Metrics:
- Average Transcript Length: {self.df['transcript_length'].mean():.0f} characters
- Average Word Count: {self.df['word_count'].mean():.0f} words
- Average Description Length: {self.df['description_length'].mean():.0f} characters
- Average Speaking Rate: {self.df['speaking_rate'].mean():.0f} WPM

Quality Metrics:
- Missing Values: {self.df.isnull().sum().sum()}
- Unknown Genders: {(self.df['gender'] == 'unknown').sum()}
- Unknown Ages: {(self.df['age'] == 'unknown').sum()}
- Unknown Accents: {(self.df['accent'] == 'unknown').sum()}

Files Generated:
- duration_analysis.png: Audio duration distributions and statistics
- demographic_analysis.png: Gender and age breakdowns
- accent_analysis.png: Accent distributions and regional groupings
- dataset_analysis.png: Source dataset comparisons
- text_analysis.png: Transcript and description analysis
- correlation_analysis.png: Feature correlation heatmap
- quality_metrics.png: Data quality and completeness metrics
- statistics.json: Raw statistics in JSON format
"""
        
        with open(self.output_dir / "analysis_report.txt", 'w') as f:
            f.write(report)

def main():
    print("🎯 VoicePersona Dataset Analyzer")
    print("=" * 50)
    
    analyzer = VoicePersonaAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main()


"""
8 Analysis Categories:
    Duration Analysis: Histograms, box plots, percentiles, cumulative plots
    Demographics: Gender pie charts, age distributions, cross-tabulations
    Accent Analysis: Top 20 accents, regional groupings, diversity metrics
    Dataset Sources: Distribution comparisons, speaker counts, duration totals
    Text Content: Transcript lengths, word counts, speaking rates
    Correlations: Feature correlation heatmaps
    Quality Metrics: Missing values, outliers, completeness scores
    Summary Report: Comprehensive text statistics

    
Generated Files:
    7 PNG visualization files
    statistics.json with raw data
    analysis_report.txt with summary

"""