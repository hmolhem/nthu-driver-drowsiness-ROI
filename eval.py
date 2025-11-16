"""Evaluation script for drowsiness detection model"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.config.config import Config
from src.dataset.loader import NTHUDDDDataset, get_dataloader
from src.dataset.augmentation import get_val_augmentation
from src.models.drowsiness_detector import create_model
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    visualize_predictions, plot_class_distribution
)
from src.utils.helpers import (
    set_seed, load_checkpoint, save_metrics, get_device
)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    save_predictions: bool = False
):
    """
    Evaluate model on dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on
        save_predictions: Whether to save all predictions
    
    Returns:
        Dictionary of metrics and MetricsCalculator
    """
    model.eval()
    
    metrics_calc = MetricsCalculator()
    all_images = []
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_metadata = []
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for images, labels, metadata in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        metrics_calc.update(preds, labels, probs)
        
        if save_predictions:
            all_images.append(images.cpu())
            all_predictions.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probs.cpu())
            all_metadata.extend(metadata)
    
    # Compute metrics
    metrics = metrics_calc.compute()
    
    # Get confusion matrix
    cm = metrics_calc.get_confusion_matrix()
    
    # Get classification report
    report = metrics_calc.get_classification_report()
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'metrics_calculator': metrics_calc
    }
    
    if save_predictions:
        results['images'] = torch.cat(all_images)
        results['predictions'] = torch.cat(all_predictions)
        results['labels'] = torch.cat(all_labels)
        results['probabilities'] = torch.cat(all_probabilities)
        results['metadata'] = all_metadata
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate drowsiness detection model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(config.experiment.output_dir) / config.experiment.experiment_name / 'evaluation'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get subjects for the split
    if args.split == 'train':
        subjects = config.data.train_subjects
    elif args.split == 'val':
        subjects = config.data.val_subjects
    else:
        subjects = config.data.test_subjects
    
    # Create dataset
    val_transform = get_val_augmentation(config.data.image_size)
    
    dataset = NTHUDDDDataset(
        data_root=config.data.data_root,
        split=args.split,
        subjects=subjects,
        image_size=config.data.image_size,
        transform=val_transform,
        use_roi=config.data.use_roi
    )
    
    print(f"Evaluating on {args.split} split")
    print(f"Subjects: {subjects}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Class distribution: {dataset.class_counts}")
    
    # Create dataloader
    dataloader = get_dataloader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Create model
    model = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=False,  # Will load from checkpoint
        dropout=config.model.dropout,
        use_roi_attention=config.model.use_roi_attention,
        freeze_backbone=False
    )
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(model, args.checkpoint, device=device)
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate(
        model, dataloader, device,
        save_predictions=args.visualize
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nMetrics:")
    for metric_name, value in results['metrics'].items():
        print(f"  {metric_name}: {value:.4f}")
    
    print(f"\nClassification Report:")
    print(results['classification_report'])
    
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save results
    metrics_to_save = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in results['metrics'].items()}
    save_metrics(metrics_to_save, str(output_dir / f'{args.split}_metrics.json'))
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names=['Awake', 'Drowsy'],
            save_path=str(output_dir / f'{args.split}_confusion_matrix.png')
        )
        
        # ROC curve (if available)
        fpr, tpr, _ = results['metrics_calculator'].get_roc_curve()
        if fpr is not None:
            auc = results['metrics']['auc']
            plot_roc_curve(
                fpr, tpr, auc,
                save_path=str(output_dir / f'{args.split}_roc_curve.png')
            )
        
        # Sample predictions
        visualize_predictions(
            results['images'][:16],
            results['predictions'][:16],
            results['labels'][:16],
            results['probabilities'][:16],
            class_names=['Awake', 'Drowsy'],
            save_path=str(output_dir / f'{args.split}_predictions.png'),
            num_samples=16
        )
        
        # Class distribution
        plot_class_distribution(
            results['labels'].numpy(),
            class_names=['Awake', 'Drowsy'],
            save_path=str(output_dir / f'{args.split}_class_distribution.png')
        )
        
        print(f"Visualizations saved to {output_dir}")
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
