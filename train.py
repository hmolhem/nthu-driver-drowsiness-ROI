"""Training script for driver drowsiness detection."""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from src.utils.config import Config
from src.utils.metrics import calculate_metrics, MetricsTracker
from src.utils.visualization import plot_training_curves
from src.models.builder import build_model, print_model_summary
from src.data.dataset import NTHUDrowsinessDataset, create_subject_splits
from src.data.transforms import get_train_transforms, get_val_transforms


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions and scores
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Collect results
            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # Score for drowsy class
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    
    return avg_loss, metrics


def main(args):
    """Main training function."""
    # Load configuration
    if args.config:
        config = Config(config_path=args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.data_root:
        config.set('data.dataset_root', args.data_root)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    if args.lr:
        config.set('training.learning_rate', args.lr)
    if args.backbone:
        config.set('model.backbone', args.backbone)
    
    # Set random seed
    seed = config.get('experiment.seed', 42)
    set_seed(seed)
    
    # Setup experiment directory
    exp_name = config.get('experiment.name', 'default_experiment')
    exp_dir = Path('experiments') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(str(exp_dir / 'config.yaml'))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create subject splits
    data_root = config.get('data.dataset_root')
    train_subjects, val_subjects, test_subjects = create_subject_splits(data_root)
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    
    # Create datasets
    image_size = tuple(config.get('data.image_size', [224, 224]))
    use_augmentation = config.get('data.augmentation', True)
    use_roi = config.get('model.use_roi', False)
    
    train_dataset = NTHUDrowsinessDataset(
        root_dir=data_root,
        split='train',
        subject_ids=train_subjects,
        transform=get_train_transforms(image_size, use_augmentation),
        use_roi=use_roi
    )
    
    val_dataset = NTHUDrowsinessDataset(
        root_dir=data_root,
        split='val',
        subject_ids=val_subjects,
        transform=get_val_transforms(image_size),
        use_roi=use_roi
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    batch_size = config.get('data.batch_size', 32)
    num_workers = config.get('data.num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    print_model_summary(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    lr = config.get('training.learning_rate', 0.001)
    optimizer_name = config.get('training.optimizer', 'adam').lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler_name = config.get('training.scheduler', 'step')
    if scheduler_name == 'step':
        step_size = config.get('training.step_size', 10)
        gamma = config.get('training.gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        epochs = config.get('training.epochs', 50)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    
    # Tensorboard
    if config.get('logging.use_tensorboard', True):
        writer = SummaryWriter(exp_dir / 'runs')
    else:
        writer = None
    
    # Metrics tracker
    tracker = MetricsTracker()
    
    # Training loop
    epochs = config.get('training.epochs', 50)
    patience = config.get('training.early_stopping_patience', 10)
    save_frequency = config.get('logging.save_frequency', 5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        epoch_metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics.get('auc', 0.0),
        }
        
        tracker.update(epoch, epoch_metrics)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics.get('auc', 0.0):.4f}")
        
        # Tensorboard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            writer.add_scalar('AUC/val', val_metrics.get('auc', 0.0), epoch)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config.config
            }
            torch.save(checkpoint, exp_dir / 'best_model.pth')
            print(f"Saved best model with val_acc: {val_metrics['accuracy']:.4f}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % save_frequency == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config.config
            }
            torch.save(checkpoint, exp_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    # Plot training curves
    plot_training_curves(tracker.get_history(), str(exp_dir / 'training_curves.png'))
    
    # Save final metrics
    best_metrics = tracker.get_best_metrics()
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_metrics['best_val_acc']:.4f}")
    print(f"Best epoch: {best_metrics['best_epoch']}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train drowsiness detection model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-root', type=str, help='Dataset root directory')
    parser.add_argument('--backbone', type=str, help='Model backbone')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
