"""Visualization utilities for plots and charts."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history and len(history['train_acc']) > 0:
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        if 'val_acc' in history and len(history['val_acc']) > 0:
            axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, 
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = ['Alert', 'Drowsy']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                   save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores/probabilities
        save_path: Path to save the plot (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           save_path: Optional[str] = None):
    """
    Plot comparison of metrics across different models/experiments.
    
    Args:
        metrics_dict: Dictionary mapping experiment names to metrics
        save_path: Path to save the plot (optional)
    """
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    experiments = list(metrics_dict.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.8 / len(experiments)
    
    for i, exp_name in enumerate(experiments):
        values = [metrics_dict[exp_name].get(m, 0.0) for m in metric_names]
        offset = (i - len(experiments)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=exp_name)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    plt.close()
