#!/usr/bin/env python3
"""
Compute batch distribution for MassBench Adenocarcinoma dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_batch_distribution(data_path=None):
    """
    Analyze batch distribution in training and test sets.
    
    Args:
        data_path: Path to data directory containing CSV files
        
    Returns:
        dict with batch statistics for train/test/valid splits
    """
    # Look for amide_data.csv or adenocarcinoma data
    if data_path is None:
        data_path = Path('data')
    else:
        data_path = Path(data_path)
    
    csv_files = list(data_path.glob('*amide*.csv')) + list(data_path.glob('*adenocarcinoma*.csv'))
    
    if not csv_files:
        print(f"No data files found in {data_path}")
        print("Looking for files with 'amide' or 'adenocarcinoma' in name...")
        print("Available files:", list(data_path.glob('*.csv'))[:5])
        return None
    
    csv_file = csv_files[0]
    print(f"Loading data from: {csv_file}")
    
    df = pd.read_csv(csv_file, index_col=0)
    
    # Identify batch, label, and feature columns
    # Common patterns: batch, batches, domain, group, set
    batch_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'batch' in col_lower or 'domain' in col_lower or 'set' in col_lower:
            if batch_col is None:
                batch_col = col
        if 'label' in col_lower or 'group' in col_lower or 'class' in col_lower:
            if label_col is None:
                label_col = col
    
    if batch_col is None:
        print("Available columns:", df.columns.tolist())
        print("Could not identify batch column. Please specify manually.")
        return None
    
    print(f"\nBatch column: {batch_col}")
    if label_col:
        print(f"Label column: {label_col}")
    
    # Get batch and label distributions
    batches = df[batch_col]
    batch_counts = Counter(batches)
    unique_batches = sorted(batch_counts.keys())
    
    # Compute overall statistics
    stats = {
        'total_samples': len(df),
        'n_batches': len(unique_batches),
        'unique_batches': unique_batches,
        'batch_distribution': dict(batch_counts),
        'batch_percentages': {b: f"{count/len(df)*100:.1f}%" for b, count in batch_counts.items()},
    }
    
    if label_col:
        label_counts = Counter(df[label_col])
        stats['n_classes'] = len(label_counts)
        stats['class_distribution'] = dict(label_counts)
        stats['class_percentages'] = {c: f"{count/len(df)*100:.1f}%" for c, count in label_counts.items()}
        
        # Cross-tabulation: batches vs labels
        crosstab = pd.crosstab(df[batch_col], df[label_col], margins=True)
        stats['batch_label_crosstab'] = crosstab.to_dict()
    
    return stats

def print_batch_statistics(stats):
    """Print formatted batch statistics."""
    if stats is None:
        return
    
    print("\n" + "="*70)
    print("BATCH DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nTotal Samples: {stats['total_samples']}")
    print(f"Number of Batches: {stats['n_batches']}")
    print(f"Unique Batches: {stats['unique_batches']}")
    
    print("\nBatch Distribution:")
    print("-" * 50)
    for batch, count in sorted(stats['batch_distribution'].items()):
        pct = stats['batch_percentages'][batch]
        print(f"  Batch {str(batch):10s}: {count:4d} samples ({pct})")
    
    if 'class_distribution' in stats:
        print(f"\nNumber of Classes: {stats['n_classes']}")
        print("\nClass Distribution:")
        print("-" * 50)
        for cls, count in sorted(stats['class_distribution'].items()):
            pct = stats['class_percentages'][cls]
            print(f"  Class {str(cls):10s}: {count:4d} samples ({pct})")
        
        print("\nBatch x Class Cross-tabulation:")
        print("-" * 50)
        crosstab_df = pd.DataFrame(stats['batch_label_crosstab'])
        print(crosstab_df)
    
    print("\n" + "="*70)

def format_dataset_metrics(stats):
    """Format statistics as dataset documentation."""
    if stats is None:
        return
    
    doc = f"""
# MassBench Adenocarcinoma - Batch Distribution

## Overall Statistics
- Total Samples: {stats['total_samples']}
- Number of Batches: {stats['n_batches']}
- Unique Batches: {', '.join(map(str, stats['unique_batches']))}

## Batch Distribution
| Batch | Samples | Percentage |
|-------|---------|-----------|
"""
    for batch in sorted(stats['batch_distribution'].keys()):
        count = stats['batch_distribution'][batch]
        pct = stats['batch_percentages'][batch]
        doc += f"| {batch} | {count} | {pct} |\n"
    
    if 'class_distribution' in stats:
        doc += f"""
## Class Distribution (by Batch)

| Batch | """
        for cls in sorted(stats['class_distribution'].keys()):
            doc += f"Class {cls} | "
        doc += "Total |\n|-------|"
        for cls in sorted(stats['class_distribution'].keys()):
            doc += "---|"
        doc += "---|\n"
        
        for batch in sorted(stats['batch_distribution'].keys()):
            doc += f"| {batch} |"
            # Would need crosstab to fill this
            doc += " |\n"
    
    return doc

if __name__ == '__main__':
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data'
    
    stats = analyze_batch_distribution(data_path)
    if stats:
        print_batch_statistics(stats)
        
        # Save documentation
        doc = format_dataset_metrics(stats)
        if doc:
            with open('BATCH_DISTRIBUTION.md', 'w') as f:
                f.write(doc)
            print("\n✓ Documentation saved to BATCH_DISTRIBUTION.md")
