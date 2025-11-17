"""
Create subject-exclusive train/val/test splits from the archive manifest.
Ensures no subject appears in multiple splits to prevent data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_subject_exclusive_splits(
    manifest_path,
    output_dir,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42
):
    """
    Create subject-exclusive splits ensuring no subject leakage.
    
    Args:
        manifest_path: Path to the full dataset manifest CSV
        output_dir: Directory to save split CSV files
        train_ratio: Proportion of subjects for training
        val_ratio: Proportion of subjects for validation
        test_ratio: Proportion of subjects for testing
        random_seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"Total samples: {len(df)}")
    
    # Get unique subjects
    subjects = df['subject'].unique()
    np.random.seed(random_seed)
    subjects = np.random.permutation(subjects)
    
    print(f"\nSubjects: {sorted(subjects)}")
    print(f"Number of subjects: {len(subjects)}")
    
    # Calculate split sizes
    n_subjects = len(subjects)
    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    # Remaining goes to test to ensure all subjects are included
    
    # Adjust if we don't have enough subjects
    if n_train + n_val >= n_subjects:
        # For small datasets, ensure at least 1 subject per split
        n_train = max(1, n_subjects - 2)
        n_val = 1
    
    # Split subjects
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    print(f"\nSplit assignment:")
    print(f"Train subjects: {sorted(train_subjects)} ({len(train_subjects)} subjects)")
    print(f"Val subjects:   {sorted(val_subjects)} ({len(val_subjects)} subjects)")
    print(f"Test subjects:  {sorted(test_subjects)} ({len(test_subjects)} subjects)")
    
    # Create splits
    train_df = df[df['subject'].isin(train_subjects)].copy()
    val_df = df[df['subject'].isin(val_subjects)].copy()
    test_df = df[df['subject'].isin(test_subjects)].copy()
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Split Statistics:")
    print(f"{'='*60}")
    
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        print(f"  Samples: {len(split_df)}")
        print(f"  Label distribution:")
        label_counts = split_df['label'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(split_df)
            print(f"    {label}: {count} ({pct:.1f}%)")
        print(f"  Subject distribution:")
        for subject in sorted(split_df['subject'].unique()):
            count = len(split_df[split_df['subject'] == subject])
            print(f"    {subject}: {count}")
    
    # Verify no leakage
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    
    assert len(train_set & val_set) == 0, "Subject leakage between train and val!"
    assert len(train_set & test_set) == 0, "Subject leakage between train and test!"
    assert len(val_set & test_set) == 0, "Subject leakage between val and test!"
    print("✓ No subject leakage detected")
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    assert total_samples == len(df), "Sample count mismatch!"
    print(f"✓ All samples accounted for ({total_samples} == {len(df)})")
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n{'='*60}")
    print("Saved splits:")
    print(f"{'='*60}")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")
    
    # Also save a combined manifest with split labels
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_path = output_dir / 'manifest_with_splits.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined: {combined_path}")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create subject-exclusive train/val/test splits'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='data/manifests/archive_manifest.csv',
        help='Path to dataset manifest CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/splits',
        help='Directory to save split files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.6,
        help='Proportion of subjects for training (default: 0.6)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Proportion of subjects for validation (default: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Proportion of subjects for testing (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    create_subject_exclusive_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
