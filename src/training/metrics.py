"""Evaluation metrics for drowsiness detection."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """Calculate and track various classification metrics."""
    
    def __init__(self, num_classes=2, class_names=None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all accumulated predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels, probs=None):
        """
        Update with new predictions and labels.
        
        Args:
            preds: Predicted labels (numpy array or torch tensor)
            labels: True labels (numpy array or torch tensor)
            probs: Prediction probabilities (optional)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
        if probs is not None:
            self.all_probs.extend(probs)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            String containing classification report
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        return classification_report(
            labels,
            preds,
            target_names=self.class_names,
            digits=4
        )
    
    def print_metrics(self):
        """Print all computed metrics in a readable format."""
        metrics = self.compute()
        
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Macro Precision:    {metrics['precision_macro']:.4f}")
        print(f"  Macro Recall:       {metrics['recall_macro']:.4f}")
        print(f"  Macro F1-Score:     {metrics['f1_macro']:.4f}")
        print(f"  Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Weighted Recall:    {metrics['recall_weighted']:.4f}")
        print(f"  Weighted F1-Score:  {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for class_name in self.class_names:
            print(f"  {class_name}:")
            print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_{class_name}']:.4f}")
            print(f"    F1-Score:  {metrics[f'f1_{class_name}']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  Predicted →")
        print(f"  Actual ↓    {' '.join([f'{name:>12}' for name in self.class_names])}")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:>12}  {' '.join([f'{cm[i,j]:>12}' for j in range(len(self.class_names))])}")
        
        print("\n" + "="*60)


class AverageMeter:
    """Compute and store the average and current value."""
    
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
        self.avg = self.sum / self.count if self.count > 0 else 0


def calculate_macro_f1(preds, labels, num_classes=2):
    """
    Calculate macro F1-score (primary metric for this project).
    
    Args:
        preds: Predicted labels
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Macro F1-score
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return f1_score(labels, preds, average='macro', zero_division=0)
