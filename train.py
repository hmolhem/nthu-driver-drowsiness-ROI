"""Training script for drowsiness detection model"""

import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.config.config import Config
from src.dataset.loader import NTHUDDDDataset, get_dataloader, create_subject_splits
from src.dataset.augmentation import get_train_augmentation, get_val_augmentation
from src.models.drowsiness_detector import create_model
from src.utils.metrics import MetricsCalculator, AverageMeter
from src.utils.visualization import (
    plot_training_curves, plot_confusion_matrix, 
    plot_roc_curve, plot_class_distribution
)
from src.utils.helpers import (
    set_seed, save_checkpoint, save_metrics,
    EarlyStopping, get_device, count_parameters, format_time
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool = True
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator()
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            metrics_calc.update(preds, labels, probs)
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    metrics = metrics_calc.compute()
    metrics['loss'] = loss_meter.avg
    
    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Validate for one epoch"""
    model.eval()
    
    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator()
    
    pbar = tqdm(dataloader, desc='Validation')
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        metrics_calc.update(preds, labels, probs)
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    metrics = metrics_calc.compute()
    metrics['loss'] = loss_meter.avg
    
    return metrics, metrics_calc


def train(config: Config):
    """Main training function"""
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Create output directories
    output_dir = Path(config.experiment.output_dir) / config.experiment.experiment_name
    checkpoint_dir = output_dir / config.experiment.checkpoint_dir
    log_dir = output_dir / config.experiment.log_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.to_yaml(str(output_dir / 'config.yaml'))
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create subject splits if not provided
    if config.data.train_subjects is None:
        print("Creating subject-exclusive splits...")
        train_subjects, val_subjects, test_subjects = create_subject_splits(
            config.data.data_root,
            seed=config.experiment.seed
        )
        config.data.train_subjects = train_subjects
        config.data.val_subjects = val_subjects
        config.data.test_subjects = test_subjects
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")
        print(f"Test subjects: {test_subjects}")
    
    # Create datasets
    train_transform = get_train_augmentation(config.data.image_size) if config.data.augmentation else None
    val_transform = get_val_augmentation(config.data.image_size)
    
    train_dataset = NTHUDDDDataset(
        data_root=config.data.data_root,
        split='train',
        subjects=config.data.train_subjects,
        image_size=config.data.image_size,
        transform=train_transform,
        use_roi=config.data.use_roi
    )
    
    val_dataset = NTHUDDDDataset(
        data_root=config.data.data_root,
        split='val',
        subjects=config.data.val_subjects,
        image_size=config.data.image_size,
        transform=val_transform,
        use_roi=config.data.use_roi
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Class distribution (train): {train_dataset.class_counts}")
    print(f"Class distribution (val): {val_dataset.class_counts}")
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Create model
    model = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        use_roi_attention=config.model.use_roi_attention,
        freeze_backbone=config.model.freeze_backbone
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model: {config.model.backbone}")
    print(f"Trainable parameters: {num_params:,}")
    
    # Create loss function with class weights
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer
    if config.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    
    # Create scheduler
    if config.training.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    elif config.training.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif config.training.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5
        )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, config.training.mixed_precision
        )
        
        # Validate
        if (epoch + 1) % config.experiment.eval_freq == 0:
            val_metrics, val_metrics_calc = validate_epoch(
                model, val_loader, criterion, device
            )
        else:
            val_metrics = {}
        
        # Update scheduler
        if config.training.scheduler == 'plateau':
            if 'accuracy' in val_metrics:
                scheduler.step(val_metrics['accuracy'])
        else:
            scheduler.step()
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        
        if val_metrics:
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Update history
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        
        # Save checkpoint
        if val_metrics and val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                str(checkpoint_dir / 'best_model.pth'),
                scheduler
            )
            print(f"✓ New best model saved! Acc: {best_val_acc:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.experiment.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                val_metrics if val_metrics else train_metrics,
                str(checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'),
                scheduler
            )
        
        # Early stopping
        if val_metrics and early_stopping(val_metrics['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining complete in {format_time(total_time)}")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch + 1,
        val_metrics if val_metrics else train_metrics,
        str(checkpoint_dir / 'final_model.pth'),
        scheduler
    )
    
    # Save training history
    save_metrics(history, str(log_dir / 'training_history.json'))
    
    # Plot training curves
    train_metrics_dict = {
        'accuracy': history['train_acc'],
        'f1': history['train_f1']
    }
    val_metrics_dict = {
        'accuracy': history['val_acc'],
        'f1': history['val_f1']
    }
    
    plot_training_curves(
        history['train_loss'],
        history['val_loss'],
        train_metrics_dict,
        val_metrics_dict,
        save_path=str(log_dir / 'training_curves.png')
    )
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train drowsiness detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/baseline.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = Config()
    
    print("Configuration:")
    print(config)
    print()
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
