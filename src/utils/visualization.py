"""Visualization utilities for training and evaluation"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dictionary of metric_name -> list of values
        val_metrics: Dictionary of metric_name -> list of values
        save_path: Path to save figure
        figsize: Figure size
    """
    num_metrics = len(train_metrics) + 1  # +1 for loss
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    # Plot loss
    axes[0].plot(train_losses, label='Train', marker='o', markersize=3)
    axes[0].plot(val_losses, label='Val', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    for idx, (metric_name, train_values) in enumerate(train_metrics.items(), 1):
        if metric_name in val_metrics:
            axes[idx].plot(train_values, label='Train', marker='o', markersize=3)
            axes[idx].plot(val_metrics[metric_name], label='Val', marker='s', markersize=3)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name.capitalize())
            axes[idx].set_title(f'{metric_name.capitalize()} Curve')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row (true labels)
    """
    if class_names is None:
        class_names = ['Awake', 'Drowsy']
    
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_display = cm_normalized
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc: Area under curve
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    num_samples: int = 16,
    figsize: Tuple[int, int] = (16, 16)
):
    """
    Visualize sample predictions
    
    Args:
        images: Batch of images (B, C, H, W)
        predictions: Predicted labels (B,)
        labels: True labels (B,)
        probabilities: Class probabilities (B, num_classes)
        class_names: List of class names
        save_path: Path to save figure
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Awake', 'Drowsy']
    
    num_samples = min(num_samples, len(images))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Denormalize and convert to numpy
        img = images[idx] * std + mean
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        pred = predictions[idx].item()
        label = labels[idx].item()
        prob = probabilities[idx, pred].item()
        
        # Display image
        ax.imshow(img)
        
        # Set title with prediction info
        color = 'green' if pred == label else 'red'
        title = f"Pred: {class_names[pred]} ({prob:.2f})\nTrue: {class_names[label]}"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot class distribution
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['Awake', 'Drowsy']
    
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar([class_names[i] for i in unique], counts, color=['skyblue', 'salmon'])
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, pad=20)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{count}\n({count/len(labels)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10
        )
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    plt.close()
