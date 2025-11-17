"""
Main training script for baseline drowsiness classifiers.

Usage:
    python src/training/train_baseline.py --config configs/baseline_resnet50.yaml
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.classifier import create_model
from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.training.trainer import Trainer
from src.utils.config import get_config
import random
import numpy as np


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train baseline drowsiness classifier')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu), overrides config'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = get_config(args.config)
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Set random seed to {seed}")
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = config.get('device', 'cuda')
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create transforms
    print("\nCreating data transforms...")
    image_size = config.data.get('image_size', 224)
    augment = config.augmentation.get('enabled', True)
    
    train_transform = get_train_transforms(image_size, augment=augment)
    val_transform = get_val_transforms(image_size)
    
    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(
        train_csv=config.data.train_csv,
        val_csv=config.data.val_csv,
        test_csv=config.data.test_csv,
        data_root=config.data.data_root,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.data.get('batch_size', 32),
        num_workers=config.data.get('num_workers', 4),
        pin_memory=config.data.get('pin_memory', True)
    )
    
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}")
    
    # Create model
    print(f"\nCreating model: {config.model.architecture}")
    model = create_model(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=config,
        device=device
    )
    
    # Train
    num_epochs = config.training.get('epochs', 50)
    trainer.train(num_epochs)
    
    print("\n✓ Training completed successfully!")


if __name__ == '__main__':
    main()
