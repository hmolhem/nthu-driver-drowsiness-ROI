"""Evaluation script for driver drowsiness detection."""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.utils.config import Config
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve
from src.models.builder import build_model
from src.data.dataset import NTHUDrowsinessDataset, create_subject_splits
from src.data.transforms import get_val_transforms


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Collect results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def main(args):
    """Main evaluation function."""
    # Load configuration
    if args.config:
        config = Config(config_path=args.config)
    else:
        # Try to load from experiment directory
        exp_dir = Path(args.checkpoint).parent
        config_path = exp_dir / 'config.yaml'
        if config_path.exists():
            config = Config(config_path=str(config_path))
        else:
            config = Config()
    
    # Override config
    if args.data_root:
        config.set('data.dataset_root', args.data_root)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create subject splits
    data_root = config.get('data.dataset_root')
    train_subjects, val_subjects, test_subjects = create_subject_splits(data_root)
    
    # Determine which split to evaluate
    if args.split == 'train':
        eval_subjects = train_subjects
    elif args.split == 'val':
        eval_subjects = val_subjects
    else:  # test
        eval_subjects = test_subjects
    
    print(f"Evaluating on {args.split} set with {len(eval_subjects)} subjects")
    
    # Create dataset
    image_size = tuple(config.get('data.image_size', [224, 224]))
    use_roi = config.get('model.use_roi', False)
    
    eval_dataset = NTHUDrowsinessDataset(
        root_dir=data_root,
        split=args.split,
        subject_ids=eval_subjects,
        transform=get_val_transforms(image_size),
        use_roi=use_roi
    )
    
    print(f"Dataset size: {len(eval_dataset)}")
    
    # Create dataloader
    batch_size = config.get('data.batch_size', 32)
    num_workers = config.get('data.num_workers', 4)
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    y_true, y_pred, y_score = evaluate(model, eval_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_score)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics.get('auc', 0.0):.4f}")
    print("=" * 60)
    
    # Save visualizations if output directory specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=['Alert', 'Drowsy'],
            save_path=str(output_path / f'confusion_matrix_{args.split}.png')
        )
        
        # ROC curve
        plot_roc_curve(
            y_true, y_score,
            save_path=str(output_path / f'roc_curve_{args.split}.png')
        )
        
        # Save metrics to file
        metrics_file = output_path / f'metrics_{args.split}.txt'
        with open(metrics_file, 'w') as f:
            f.write("Evaluation Metrics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC:       {metrics.get('auc', 0.0):.4f}\n")
        
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate drowsiness detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to config file (optional, will try to load from checkpoint dir)')
    parser.add_argument('--data-root', type=str,
                       help='Dataset root directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    main(args)
