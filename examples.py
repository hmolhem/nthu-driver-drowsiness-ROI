"""Example script showing basic usage of the drowsiness detection framework"""

import torch
from pathlib import Path

# Import framework components
from src.config.config import Config
from src.dataset.loader import NTHUDDDDataset, create_subject_splits, get_dataloader
from src.dataset.augmentation import get_train_augmentation, get_val_augmentation
from src.models.drowsiness_detector import create_model
from src.utils.metrics import MetricsCalculator
from src.utils.helpers import set_seed, get_device


def example_basic_usage():
    """Example: Basic usage of the framework"""
    
    print("=" * 60)
    print("Example: Basic Framework Usage")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load configuration
    config = Config()
    print(f"\nDefault configuration:")
    print(config)
    
    # Create a model
    print("\n" + "-" * 60)
    print("Creating model...")
    model = create_model(
        backbone="resnet18",
        num_classes=2,
        pretrained=True,
        dropout=0.5,
        use_roi_attention=True
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} trainable parameters")
    
    # Test forward pass
    print("\n" + "-" * 60)
    print("Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits):\n{output}")
    
    # Get predictions
    probs = torch.softmax(output, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    print(f"\nPredictions: {preds}")
    print(f"Probabilities:\n{probs}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def example_config_management():
    """Example: Configuration management"""
    
    print("\n" + "=" * 60)
    print("Example: Configuration Management")
    print("=" * 60)
    
    # Create custom configuration
    config = Config()
    config.experiment.experiment_name = "my_experiment"
    config.model.backbone = "efficientnet_b0"
    config.training.learning_rate = 0.001
    config.training.num_epochs = 50
    
    # Save configuration
    output_path = "/tmp/example_config.yaml"
    config.to_yaml(output_path)
    print(f"\nConfiguration saved to: {output_path}")
    
    # Load configuration
    loaded_config = Config.from_yaml(output_path)
    print("\nLoaded configuration:")
    print(loaded_config)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def example_model_comparison():
    """Example: Compare different model backbones"""
    
    print("\n" + "=" * 60)
    print("Example: Model Comparison")
    print("=" * 60)
    
    device = get_device()
    backbones = ["resnet18", "resnet50", "efficientnet_b0", "mobilenetv3_small"]
    
    print("\nComparing model backbones:\n")
    print(f"{'Backbone':<20} {'Parameters':>15} {'Output Shape'}")
    print("-" * 60)
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    for backbone in backbones:
        try:
            model = create_model(backbone=backbone, pretrained=False)
            model = model.to(device)
            
            num_params = sum(p.numel() for p in model.parameters())
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"{backbone:<20} {num_params:>15,} {str(output.shape):>15}")
        except Exception as e:
            print(f"{backbone:<20} {'Error':>15} {str(e)[:30]}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def example_data_augmentation():
    """Example: Data augmentation pipeline"""
    
    print("\n" + "=" * 60)
    print("Example: Data Augmentation")
    print("=" * 60)
    
    import numpy as np
    
    # Create sample image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Get augmentation transforms
    train_aug = get_train_augmentation()
    val_aug = get_val_augmentation()
    
    print("\nTraining augmentations:")
    print(train_aug)
    
    print("\nValidation augmentations:")
    print(val_aug)
    
    # Apply augmentation
    augmented = train_aug(image=image)
    print(f"\nOriginal image shape: {image.shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def main():
    """Run all examples"""
    
    print("\n" + "#" * 60)
    print("# NTHU Driver Drowsiness Detection - Examples")
    print("#" * 60)
    
    try:
        example_basic_usage()
    except Exception as e:
        print(f"\nError in basic usage example: {e}")
    
    try:
        example_config_management()
    except Exception as e:
        print(f"\nError in config management example: {e}")
    
    try:
        example_model_comparison()
    except Exception as e:
        print(f"\nError in model comparison example: {e}")
    
    try:
        example_data_augmentation()
    except Exception as e:
        print(f"\nError in data augmentation example: {e}")
    
    print("\n" + "#" * 60)
    print("# All examples complete!")
    print("#" * 60)


if __name__ == '__main__':
    main()
