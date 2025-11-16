"""Utility script to generate ROI masks for dataset."""

import argparse
from pathlib import Path
from src.data.roi_masks import batch_generate_masks


def main(args):
    """Generate ROI masks for all images in dataset."""
    dataset_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    
    if not dataset_root.exists():
        print(f"Error: Dataset directory {dataset_root} does not exist")
        return
    
    print(f"Generating ROI masks for dataset: {dataset_root}")
    print(f"Output directory: {output_root}")
    print(f"Mask type: {args.mask_type}")
    
    # Process each subject
    total_count = 0
    for subject_dir in dataset_root.iterdir():
        if not subject_dir.is_dir():
            continue
        
        print(f"\nProcessing {subject_dir.name}...")
        
        # Process alert images
        alert_dir = subject_dir / 'alert'
        if alert_dir.exists():
            output_dir = output_root / subject_dir.name / 'alert'
            batch_generate_masks(
                str(alert_dir),
                str(output_dir),
                mask_type=args.mask_type
            )
        
        # Process drowsy images
        drowsy_dir = subject_dir / 'drowsy'
        if drowsy_dir.exists():
            output_dir = output_root / subject_dir.name / 'drowsy'
            batch_generate_masks(
                str(drowsy_dir),
                str(output_dir),
                mask_type=args.mask_type
            )
    
    print(f"\n{'='*60}")
    print("ROI mask generation completed!")
    print(f"Masks saved to: {output_root}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ROI masks for dataset')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for ROI masks')
    parser.add_argument('--mask-type', type=str, default='eye_mouth',
                       choices=['eye_mouth', 'eye', 'mouth', 'upper_face'],
                       help='Type of ROI mask to generate')
    
    args = parser.parse_args()
    main(args)
