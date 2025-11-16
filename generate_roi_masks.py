"""Script to generate ROI masks for the dataset"""

import argparse
from pathlib import Path
from src.roi.mask_generator import ROIMaskGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate ROI masks for NTHU-DDD2 dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of NTHU-DDD2 dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for ROI masks'
    )
    parser.add_argument(
        '--predictor-path',
        type=str,
        default=None,
        help='Path to dlib shape predictor model (optional)'
    )
    args = parser.parse_args()
    
    print("Initializing ROI mask generator...")
    generator = ROIMaskGenerator(predictor_path=args.predictor_path)
    
    print(f"Generating ROI masks for dataset at: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    
    generator.generate_dataset_masks(
        image_dir=args.data_root,
        output_dir=args.output_dir
    )
    
    print("\nROI mask generation complete!")


if __name__ == '__main__':
    main()
