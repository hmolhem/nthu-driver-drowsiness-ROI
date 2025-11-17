"""
PyTorch Dataset class for NTHU Driver Drowsiness Detection.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd


# KERAS COMPARISON: Dataset replaces ImageDataGenerator
# In Keras: datagen.flow_from_directory() or datagen.flow_from_dataframe()
# In PyTorch: Custom Dataset class with __getitem__ method
class DrowsinessDataset(Dataset):
    """
    PyTorch Dataset for drowsiness detection.
    
    Args:
        csv_path: Path to split CSV file (train.csv, val.csv, or test.csv)
        data_root: Root directory containing the dataset images
        transform: Optional transform to apply to images
        label_map: Dictionary mapping label strings to integers
    """
    
    def __init__(
        self,
        csv_path,
        data_root='datasets/archive',
        transform=None,
        label_map=None
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Load manifest
        self.df = pd.read_csv(csv_path)
        
        # Create label map
        if label_map is None:
            self.label_map = {'notdrowsy': 0, 'drowsy': 1}
        else:
            self.label_map = label_map
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        KERAS COMPARISON: This method replaces Keras data generator's image loading
        In Keras: Generator yields batches automatically
        In PyTorch: __getitem__ gets ONE sample, DataLoader batches them
        
        Returns:
            tuple: (image, label, metadata_dict)
        """
        row = self.df.iloc[idx]
        
        # Load image (like Keras load_img)
        img_path = self.data_root / row['filename']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform (like Keras preprocessing_function + augmentation)
        if self.transform:
            image = self.transform(image)  # Returns tensor (C, H, W)
        
        # Get label
        label = self.label_map[row['label']]
        
        # Metadata
        metadata = {
            'filename': row['filename'],
            'subject': row['subject'],
            'glasses': row['glasses'],
            'behavior': row['behavior'],
            'frame': row['frame']
        }
        
        return image, label, metadata
    
    def get_labels(self):
        """Get all labels as a list (useful for stratified sampling)."""
        return [self.label_map[label] for label in self.df['label']]
    
    def get_class_weights(self):
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            torch.Tensor: Weights for each class
        """
        label_counts = self.df['label'].value_counts()
        total = len(self.df)
        
        weights = torch.zeros(len(self.label_map))
        for label_str, count in label_counts.items():
            label_idx = self.label_map[label_str]
            weights[label_idx] = total / (len(self.label_map) * count)
        
        return weights


# KERAS COMPARISON: create_dataloaders() replaces Keras flow_from_* methods
# In Keras: train_gen = datagen.flow_from_dataframe(df, batch_size=32)
# In PyTorch: train_loader = DataLoader(dataset, batch_size=32)
def create_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    data_root='datasets/archive',
    train_transform=None,
    val_transform=None,
    batch_size=32,
    num_workers=4,
    pin_memory=True
):
    """
    Create train, val, and test dataloaders.
    
    Args:
        train_csv: Path to training split CSV
        val_csv: Path to validation split CSV
        test_csv: Path to test split CSV
        data_root: Root directory of dataset
        train_transform: Transform for training data
        val_transform: Transform for val/test data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' dataloaders
    """
    from torch.utils.data import DataLoader  # Like Keras generators
    
    # Create datasets
    train_dataset = DrowsinessDataset(
        train_csv, data_root, transform=train_transform
    )
    val_dataset = DrowsinessDataset(
        val_csv, data_root, transform=val_transform
    )
    test_dataset = DrowsinessDataset(
        test_csv, data_root, transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }
