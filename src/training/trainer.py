"""Training engine for drowsiness detection models."""

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:  # CPU-only or older torch
    autocast = None
    GradScaler = None
from pathlib import Path
from tqdm import tqdm
import json
import csv

from ..training.metrics import MetricsCalculator, AverageMeter, calculate_macro_f1


class Trainer:
    """Training engine for classification models."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optional AMP
        self.use_amp = bool(self.config.get('training', {}).get('amp', {}).get('enabled', False) and device.startswith('cuda') and GradScaler is not None)
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None

        # Freeze backbone if requested (keep classifier/trainable head)
        if self.config.get('model', {}).get('freeze_backbone', False):
            frozen = 0
            for name, param in self.model.named_parameters():
                # Heuristic: keep final classification layer trainable (resnet: fc, efficientnet: classifier)
                if name.startswith('fc') or name.startswith('classifier'):
                    continue
                param.requires_grad = False
                frozen += 1
            if frozen:
                print(f"Frozen backbone parameters: {frozen}")

        # Setup training components (after freezing so optimizer excludes frozen params)
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_scheduler()
        
        # Tracking
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        
        # Directories
        self.save_dir = Path(config.get('logging', {}).get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = config.get('logging', {}).get('experiment_name', 'experiment')
    
    def setup_optimizer(self):
        """Setup optimizer from config."""
        train_config = self.config.get('training', {})
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        lr = train_config.get('learning_rate', 0.001)
        weight_decay = train_config.get('weight_decay', 0.0)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def setup_criterion(self):
        """Setup loss criterion from config."""
        loss_config = self.config.get('training', {}).get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy')
        
        if loss_type == 'weighted_cross_entropy' and loss_config.get('use_class_weights'):
            # Get class weights from training dataset
            class_weights = self.train_loader.dataset.get_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def setup_scheduler(self):
        """Setup learning rate scheduler from config."""
        lr_config = self.config.get('training', {}).get('lr_scheduler', {})
        scheduler_type = lr_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=lr_config.get('mode', 'max'),
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 5),
                min_lr=lr_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        KERAS COMPARISON: This replaces model.fit() for one epoch
        In Keras: model.fit(x, y, epochs=1)  # All this happens automatically
        In PyTorch: You write the loop explicitly (more control!)
        """
        self.model.train()  # Like: model.trainable = True
        
        loss_meter = AverageMeter()
        metrics_calc = MetricsCalculator(
            num_classes=2,
            class_names=['notdrowsy', 'drowsy']
        )
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        # KERAS COMPARISON: This loop replaces the magic inside model.fit()
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            if self.use_amp and autocast is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.config.get('training', {}).get('gradient_clipping', {}).get('enabled', False):
                    max_norm = self.config['training']['gradient_clipping'].get('max_norm', 1.0)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.get('training', {}).get('gradient_clipping', {}).get('enabled', False):
                    max_norm = self.config['training']['gradient_clipping'].get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
            
            # Track metrics
            preds = outputs.argmax(dim=1)
            metrics_calc.update(preds, labels)
            loss_meter.update(loss.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Compute epoch metrics
        train_metrics = metrics_calc.compute()
        train_metrics['loss'] = loss_meter.avg
        
        return train_metrics
    
    @torch.no_grad()  # KERAS COMPARISON: Like training=False in Keras
    def validate(self):
        """
        Validate on validation set.
        
        KERAS COMPARISON: This replaces model.evaluate()
        In Keras: loss, acc = model.evaluate(val_x, val_y)
        In PyTorch: You write the validation loop explicitly
        """
        self.model.eval()  # Like: model.trainable = False (disables dropout, batchnorm updates)
        
        loss_meter = AverageMeter()
        metrics_calc = MetricsCalculator(
            num_classes=2,
            class_names=['notdrowsy', 'drowsy']
        )
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
        
        log_interval = self.config.get('logging', {}).get('log_interval', 0)
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            preds = outputs.argmax(dim=1)
            metrics_calc.update(preds, labels)
            loss_meter.update(loss.item(), images.size(0))
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

            # Optional intermediate validation snapshot (does not include final metrics)
            if log_interval and batch_idx > 0 and (batch_idx % log_interval == 0):
                interim = metrics_calc.compute()
                interim['loss'] = loss_meter.avg
                snapshot_path = self.save_dir / f'{self.experiment_name}_val_interim_batch{batch_idx}_epoch{self.current_epoch}.json'
                # Convert numpy arrays (e.g., confusion matrix) to lists for JSON serialization
                serializable = {}
                for k, v in interim.items():
                    if hasattr(v, 'tolist'):
                        serializable[k] = v.tolist()
                    else:
                        serializable[k] = v
                with open(snapshot_path, 'w') as f:
                    json.dump(serializable, f, indent=2)

        
        val_metrics = metrics_calc.compute()
        val_metrics['loss'] = loss_meter.avg
        # Add classification report text for detailed per-class view
        val_metrics['classification_report'] = metrics_calc.get_classification_report()

        # Persist full validation metrics for this epoch
        serializable_val = {}
        for k, v in val_metrics.items():
            if hasattr(v, 'tolist'):
                serializable_val[k] = v.tolist()
            else:
                serializable_val[k] = v
        val_json_path = self.save_dir / f'{self.experiment_name}_val_epoch{self.current_epoch}.json'
        with open(val_json_path, 'w') as f:
            json.dump(serializable_val, f, indent=2)

        # Confusion matrix CSV for easy inspection
        cm = val_metrics.get('confusion_matrix')
        if cm is not None:
            cm_csv_path = self.save_dir / f'{self.experiment_name}_val_confusion_epoch{self.current_epoch}.csv'
            class_names = self.config.get('data', {}).get('class_names', ['notdrowsy','drowsy'])
            with open(cm_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([''] + class_names)
                for i, row in enumerate(cm):
                    writer.writerow([class_names[i]] + list(row))

        # Also maintain a latest symlink-style JSON (overwrite each epoch)
        latest_json_path = self.save_dir / f'{self.experiment_name}_val_latest.json'
        with open(latest_json_path, 'w') as f:
            json.dump(serializable_val, f, indent=2)

        return val_metrics
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save last checkpoint
        last_path = self.save_dir / f'{self.experiment_name}_last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / f'{self.experiment_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model to {best_path}")
    
    def train(self, num_epochs):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        early_stop_config = self.config.get('training', {}).get('early_stopping', {})
        early_stop_enabled = early_stop_config.get('enabled', False)
        early_stop_patience = early_stop_config.get('patience', 10)
        monitor_metric = early_stop_config.get('monitor', 'val_macro_f1')
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()

            # Persist train metrics for epoch
            serializable_train = {}
            for k, v in train_metrics.items():
                if hasattr(v, 'tolist'):
                    serializable_train[k] = v.tolist()
                else:
                    serializable_train[k] = v
            train_json_path = self.save_dir / f'{self.experiment_name}_train_epoch{self.current_epoch}.json'
            with open(train_json_path, 'w') as f:
                json.dump(serializable_train, f, indent=2)
            
            # Validate
            val_metrics = self.validate()
            
            # Print metrics
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"Macro-F1: {train_metrics['f1_macro']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"Macro-F1: {val_metrics['f1_macro']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['f1_macro'])
            
            # Check if best model
            current_metric = val_metrics.get('f1_macro', 0)
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best=is_best)
            
            # Early stopping
            if early_stop_enabled and self.epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best {monitor_metric}: {self.best_metric:.4f}")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation macro-F1: {self.best_metric:.4f}")
        print(f"{'='*60}\n")
