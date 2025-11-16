"""NTHU-DDD2 Dataset Loader with subject-exclusive splits"""

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class NTHUDDDDataset(Dataset):
    """
    NTHU-DDD2 Driver Drowsiness Detection Dataset
    
    Supports subject-exclusive splits for robust evaluation.
    Labels: 0 = Awake, 1 = Drowsy
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        subjects: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        use_roi: bool = True,
        roi_dir: Optional[str] = None
    ):
        """
        Args:
            data_root: Root directory of NTHU-DDD2 dataset
            split: 'train', 'val', or 'test'
            subjects: List of subject IDs to include (for subject-exclusive splits)
            image_size: Target image size (height, width)
            transform: Optional data augmentation transforms
            use_roi: Whether to apply ROI masks
            roi_dir: Directory containing precomputed ROI masks
        """
        self.data_root = Path(data_root)
        self.split = split
        self.subjects = subjects
        self.image_size = image_size
        self.transform = transform
        self.use_roi = use_roi
        self.roi_dir = Path(roi_dir) if roi_dir else None
        
        # Load dataset manifest
        self.samples = self._load_samples()
        
        # Class distribution
        labels = [s['label'] for s in self.samples]
        self.class_counts = {
            0: labels.count(0),
            1: labels.count(1)
        }
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples from manifest or directory structure"""
        samples = []
        
        # Try to load from manifest file if exists
        manifest_path = self.data_root / f"{self.split}_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                all_samples = json.load(f)
            
            # Filter by subjects if specified
            if self.subjects is not None:
                samples = [s for s in all_samples if s['subject'] in self.subjects]
            else:
                samples = all_samples
        else:
            # Fallback: scan directory structure
            # Expected structure: data_root/subject_XX/awake|drowsy/image.jpg
            for subject_dir in sorted(self.data_root.glob("subject_*")):
                subject_id = int(subject_dir.name.split('_')[1])
                
                # Skip if not in subject list
                if self.subjects is not None and subject_id not in self.subjects:
                    continue
                
                # Process awake images
                awake_dir = subject_dir / "awake"
                if awake_dir.exists():
                    for img_path in sorted(awake_dir.glob("*.jpg")) + sorted(awake_dir.glob("*.png")):
                        samples.append({
                            'image_path': str(img_path),
                            'label': 0,  # awake
                            'subject': subject_id
                        })
                
                # Process drowsy images
                drowsy_dir = subject_dir / "drowsy"
                if drowsy_dir.exists():
                    for img_path in sorted(drowsy_dir.glob("*.jpg")) + sorted(drowsy_dir.glob("*.png")):
                        samples.append({
                            'image_path': str(img_path),
                            'label': 1,  # drowsy
                            'subject': subject_id
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            label: Integer label (0=awake, 1=drowsy)
            metadata: Dictionary with additional info
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size
        
        # Resize image
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        image = np.array(image)
        
        # Apply ROI mask if enabled
        if self.use_roi and self.roi_dir is not None:
            roi_path = self._get_roi_path(sample['image_path'])
            if roi_path.exists():
                roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
                roi_mask = cv2.resize(roi_mask, (self.image_size[1], self.image_size[0]))
                roi_mask = roi_mask.astype(np.float32) / 255.0
                roi_mask = np.expand_dims(roi_mask, axis=-1)
                image = image * roi_mask
        
        # Apply augmentations
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        label = sample['label']
        
        metadata = {
            'subject': sample['subject'],
            'image_path': sample['image_path'],
            'original_size': original_size
        }
        
        return image, label, metadata
    
    def _get_roi_path(self, image_path: str) -> Path:
        """Get corresponding ROI mask path"""
        img_path = Path(image_path)
        relative_path = img_path.relative_to(self.data_root)
        roi_path = self.roi_dir / relative_path.with_suffix('.png')
        return roi_path
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training"""
        total = sum(self.class_counts.values())
        weights = [total / (len(self.class_counts) * count) 
                  for count in self.class_counts.values()]
        return torch.tensor(weights, dtype=torch.float32)


def create_subject_splits(
    data_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create subject-exclusive train/val/test splits
    
    Args:
        data_root: Root directory of dataset
        train_ratio: Proportion of subjects for training
        val_ratio: Proportion of subjects for validation
        test_ratio: Proportion of subjects for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Get all subject IDs
    data_root = Path(data_root)
    subject_dirs = sorted(data_root.glob("subject_*"))
    subjects = [int(d.name.split('_')[1]) for d in subject_dirs]
    
    # Shuffle subjects
    np.random.seed(seed)
    np.random.shuffle(subjects)
    
    # Split subjects
    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    return train_subjects, val_subjects, test_subjects


def get_dataloader(
    dataset: NTHUDDDDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader with standard settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle  # Drop last incomplete batch for training
    )
