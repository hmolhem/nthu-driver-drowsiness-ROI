"""Utility script to generate dataset manifests from directory structure"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def generate_manifest(
    data_root: str,
    output_path: str,
    extensions: tuple = ('.jpg', '.png', '.jpeg')
) -> List[Dict[str, Any]]:
    """
    Generate dataset manifest from directory structure
    
    Expected structure:
    data_root/
        subject_XX/
            awake/
                image1.jpg
                ...
            drowsy/
                image1.jpg
                ...
    
    Args:
        data_root: Root directory of dataset
        output_path: Path to save manifest JSON
        extensions: Valid image extensions
    
    Returns:
        List of sample dictionaries
    """
    data_root = Path(data_root)
    samples = []
    
    # Scan directory structure
    for subject_dir in sorted(data_root.glob("subject_*")):
        try:
            subject_id = int(subject_dir.name.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Invalid subject directory name: {subject_dir.name}")
            continue
        
        # Process awake images
        awake_dir = subject_dir / "awake"
        if awake_dir.exists():
            for ext in extensions:
                for img_path in sorted(awake_dir.glob(f"*{ext}")):
                    samples.append({
                        'image_path': str(img_path.relative_to(data_root)),
                        'absolute_path': str(img_path),
                        'label': 0,  # awake
                        'label_name': 'awake',
                        'subject': subject_id
                    })
        
        # Process drowsy images
        drowsy_dir = subject_dir / "drowsy"
        if drowsy_dir.exists():
            for ext in extensions:
                for img_path in sorted(drowsy_dir.glob(f"*{ext}")):
                    samples.append({
                        'image_path': str(img_path.relative_to(data_root)),
                        'absolute_path': str(img_path),
                        'label': 1,  # drowsy
                        'label_name': 'drowsy',
                        'subject': subject_id
                    })
    
    # Save manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Print statistics
    print(f"\nManifest Statistics:")
    print(f"Total samples: {len(samples)}")
    
    subjects = sorted(set(s['subject'] for s in samples))
    print(f"Number of subjects: {len(subjects)}")
    print(f"Subject IDs: {subjects}")
    
    awake_count = sum(1 for s in samples if s['label'] == 0)
    drowsy_count = sum(1 for s in samples if s['label'] == 1)
    print(f"Awake samples: {awake_count} ({awake_count/len(samples)*100:.1f}%)")
    print(f"Drowsy samples: {drowsy_count} ({drowsy_count/len(samples)*100:.1f}%)")
    
    print(f"\nManifest saved to: {output_path}")
    
    return samples


def split_manifest_by_subjects(
    manifest: List[Dict[str, Any]],
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: List[int],
    output_dir: str
):
    """
    Split manifest into train/val/test based on subjects
    
    Args:
        manifest: List of samples
        train_subjects: Training subject IDs
        val_subjects: Validation subject IDs
        test_subjects: Test subject IDs
        output_dir: Directory to save split manifests
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split samples
    train_samples = [s for s in manifest if s['subject'] in train_subjects]
    val_samples = [s for s in manifest if s['subject'] in val_subjects]
    test_samples = [s for s in manifest if s['subject'] in test_subjects]
    
    # Save split manifests
    with open(output_dir / 'train_manifest.json', 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(output_dir / 'val_manifest.json', 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    with open(output_dir / 'test_manifest.json', 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    print(f"\nSplit manifests saved to: {output_dir}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset manifest')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of NTHU-DDD2 dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='manifest.json',
        help='Output path for manifest file'
    )
    args = parser.parse_args()
    
    print(f"Scanning dataset at: {args.data_root}")
    manifest = generate_manifest(args.data_root, args.output)
    
    print(f"\nManifest generation complete!")


if __name__ == '__main__':
    main()
