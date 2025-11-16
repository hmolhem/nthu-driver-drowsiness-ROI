"""Helper utilities for training and evaluation"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import json


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    scheduler: Any = None
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load model to
    
    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")


def load_metrics(load_path: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file
    
    Args:
        load_path: Path to metrics file
    
    Returns:
        Dictionary of metrics
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def get_device() -> str:
    """Get the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
