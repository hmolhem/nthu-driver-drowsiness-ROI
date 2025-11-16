"""Metrics computation for drowsiness detection"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, List


class MetricsCalculator:
    """Calculate and track metrics for drowsiness detection"""
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated predictions and labels"""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor = None
    ):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predicted class indices (B,)
            labels: True class indices (B,)
            probabilities: Class probabilities (B, num_classes)
        """
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric name -> value
        """
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1': f1_score(labels, preds, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        if self.num_classes == 2:
            metrics['precision_awake'] = precision_score(
                labels, preds, pos_label=0, zero_division=0
            )
            metrics['recall_awake'] = recall_score(
                labels, preds, pos_label=0, zero_division=0
            )
            metrics['f1_awake'] = f1_score(
                labels, preds, pos_label=0, zero_division=0
            )
            
            metrics['precision_drowsy'] = precision_score(
                labels, preds, pos_label=1, zero_division=0
            )
            metrics['recall_drowsy'] = recall_score(
                labels, preds, pos_label=1, zero_division=0
            )
            metrics['f1_drowsy'] = f1_score(
                labels, preds, pos_label=1, zero_division=0
            )
        
        # AUC if probabilities available
        if len(self.probabilities) > 0 and self.num_classes == 2:
            probs = np.array(self.probabilities)
            try:
                metrics['auc'] = roc_auc_score(labels, probs[:, 1])
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        return confusion_matrix(labels, preds)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        target_names = ['Awake', 'Drowsy'] if self.num_classes == 2 else None
        return classification_report(labels, preds, target_names=target_names)
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data (for binary classification)"""
        if len(self.probabilities) == 0 or self.num_classes != 2:
            return None, None, None
        
        labels = np.array(self.labels)
        probs = np.array(self.probabilities)
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
        return fpr, tpr, thresholds


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_class_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2
) -> Dict[int, float]:
    """
    Compute per-class accuracy
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping class index to accuracy
    """
    class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_acc[c] = (predictions[mask] == labels[mask]).mean()
        else:
            class_acc[c] = 0.0
    return class_acc


def compute_balanced_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2
) -> float:
    """
    Compute balanced accuracy (average of per-class accuracies)
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Balanced accuracy
    """
    class_acc = compute_class_accuracy(predictions, labels, num_classes)
    return np.mean(list(class_acc.values()))
