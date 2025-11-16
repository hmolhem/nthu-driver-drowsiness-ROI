"""
Build manifest CSV file for the archive dataset (drowsy/notdrowsy folders).
This creates a CSV with columns: filename, label, subject, glasses, behavior, frame
"""

import os
import pandas as pd
from pathlib import Path


def parse_filename(filename):
    """
    Parse filename to extract metadata.
    Expected format: {subject}_{glasses}_{behavior}_{frame}_{label}.jpg
    Example: 006_glasses_sleepyCombination_1234_drowsy.jpg
    """
    parts = filename.replace('.jpg', '').split('_')
    
    # The label is always the last part
    label = parts[-1]
    
    # Frame number is second to last
    frame = parts[-2]
    
    # Subject is first part
    subject = parts[0]
    
    # Glasses status is second part
    glasses = parts[1]
    
    # Behavior is everything in between (could be multi-word)
    behavior = '_'.join(parts[2:-2])
    
    return {
        'subject': subject,
        'glasses': glasses,
        'behavior': behavior,
        'frame': frame,
        'label': label
    }


def build_manifest(data_root, output_path):
    """
    Build manifest CSV from drowsy/notdrowsy folder structure.
    
    Args:
        data_root: Path to folder containing drowsy/ and notdrowsy/ subfolders
        output_path: Path where manifest CSV will be saved
    """
    data_root = Path(data_root)
    
    records = []
    
    # Process both drowsy and notdrowsy folders
    for folder_name in ['drowsy', 'notdrowsy']:
        folder_path = data_root / folder_name
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist, skipping...")
            continue
        
        print(f"Processing {folder_name} folder...")
        
        # Get all jpg files
        image_files = list(folder_path.glob('*.jpg'))
        print(f"  Found {len(image_files)} images")
        
        for img_path in image_files:
            try:
                # Parse filename for metadata
                metadata = parse_filename(img_path.name)
                
                # Add relative path
                rel_path = f"{folder_name}/{img_path.name}"
                
                record = {
                    'filename': rel_path,
                    'label': metadata['label'],
                    'subject': metadata['subject'],
                    'glasses': metadata['glasses'],
                    'behavior': metadata['behavior'],
                    'frame': metadata['frame']
                }
                
                records.append(record)
                
            except Exception as e:
                print(f"  Error parsing {img_path.name}: {e}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by subject, behavior, frame for consistency
    df = df.sort_values(['subject', 'behavior', 'frame']).reset_index(drop=True)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nManifest created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Total images: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nSubject distribution:")
    print(df['subject'].value_counts().sort_index())
    print(f"\nBehavior distribution:")
    print(df['behavior'].value_counts())
    
    return df


if __name__ == '__main__':
    # Define paths
    data_root = Path(__file__).parent.parent.parent / 'datasets' / 'archive'
    output_path = Path(__file__).parent.parent.parent / 'data' / 'manifests' / 'archive_manifest.csv'
    
    print(f"Data root: {data_root}")
    print(f"Output path: {output_path}\n")
    
    # Build manifest
    df = build_manifest(data_root, output_path)
