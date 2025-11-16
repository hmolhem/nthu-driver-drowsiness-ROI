"""Metrics calculation and tracking utilities."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from typing import Dict, List, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_score: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction scores/probabilities (optional, for AUC)
    
    Returns:
        Dictionary containing accuracy, precision, recall, f1, and optionally AUC
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    if y_score is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        except ValueError:
            # Handle case where only one class is present
            metrics['auc'] = 0.0
    
    return metrics


class MetricsTracker:
    """Track metrics across training epochs."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
        }
        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_f1': 0.0,
            'best_epoch': 0,
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to record
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Update best metrics
        if 'val_acc' in metrics and metrics['val_acc'] > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = metrics['val_acc']
            self.best_metrics['best_epoch'] = epoch
        
        if 'val_f1' in metrics and metrics['val_f1'] > self.best_metrics['best_val_f1']:
            self.best_metrics['best_val_f1'] = metrics['val_f1']
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full training history."""
        return self.history
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best validation metrics."""
        return self.best_metrics
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key in self.history and len(self.history[key]) > 0:
            return self.history[key][-1]
        return 0.0
