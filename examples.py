"""Example script demonstrating programmatic usage of the drowsiness detection framework."""

import torch
from pathlib import Path

from src.utils.config import Config
from src.models.builder import build_model, count_parameters
from src.data.dataset import NTHUDrowsinessDataset, create_subject_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.roi_masks import create_roi_mask
import numpy as np


def example_1_basic_config():
    """Example 1: Creating and using configurations."""
    print("\n" + "="*60)
    print("Example 1: Configuration Management")
    print("="*60)
    
    # Create default configuration
    config = Config()
    print("\nDefault configuration:")
    print(f"  Backbone: {config.get('model.backbone')}")
    print(f"  Batch size: {config.get('data.batch_size')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    
    # Modify configuration
    config.set('model.backbone', 'resnet50')
    config.set('data.batch_size', 64)
    print("\nModified configuration:")
    print(f"  Backbone: {config.get('model.backbone')}")
    print(f"  Batch size: {config.get('data.batch_size')}")
    
    # Save configuration
    config.save('/tmp/my_config.yaml')
    print("\n✓ Configuration saved to /tmp/my_config.yaml")
    
    # Load configuration
    config2 = Config(config_path='/tmp/my_config.yaml')
    print(f"✓ Configuration loaded: backbone = {config2.get('model.backbone')}")


def example_2_build_models():
    """Example 2: Building different models."""
    print("\n" + "="*60)
    print("Example 2: Building Models")
    print("="*60)
    
    models_to_test = [
        ('resnet18', False),
        ('resnet18', True),
        ('efficientnet_b0', False),
        ('vgg16', False),
    ]
    
    for backbone, use_roi in models_to_test:
        config = Config(config_dict={
            'model': {
                'backbone': backbone,
                'num_classes': 2,
                'pretrained': False,
                'use_roi': use_roi,
                'dropout': 0.5
            }
        })
        
        model = build_model(config)
        num_params = count_parameters(model)
        
        roi_str = "with ROI" if use_roi else "standard"
        print(f"\n{backbone} ({roi_str}):")
        print(f"  Parameters: {num_params:,}")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        print(f"  Output shape: {output.shape}")


def example_3_dataset_loading():
    """Example 3: Dataset loading and subject splits."""
    print("\n" + "="*60)
    print("Example 3: Dataset Loading")
    print("="*60)
    
    # This example assumes dataset exists; will handle gracefully if not
    data_root = './data/NTHU-DDD2'
    
    # Create subject splits
    train_subjects, val_subjects, test_subjects = create_subject_splits(data_root)
    
    print(f"\nSubject splits:")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Test subjects: {len(test_subjects)}")
    
    # Create datasets
    train_transform = get_train_transforms(image_size=(224, 224), augmentation=True)
    val_transform = get_val_transforms(image_size=(224, 224))
    
    train_dataset = NTHUDrowsinessDataset(
        root_dir=data_root,
        split='train',
        subject_ids=train_subjects,
        transform=train_transform,
        use_roi=False
    )
    
    val_dataset = NTHUDrowsinessDataset(
        root_dir=data_root,
        split='val',
        subject_ids=val_subjects,
        transform=val_transform,
        use_roi=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    if len(train_dataset) > 0:
        # Get a sample
        image, label = train_dataset[0]
        print(f"\nSample data:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {'Drowsy' if label == 1 else 'Alert'}")


def example_4_roi_mask_generation():
    """Example 4: ROI mask generation."""
    print("\n" + "="*60)
    print("Example 4: ROI Mask Generation")
    print("="*60)
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Generate different types of masks
    mask_types = ['eye_mouth', 'eye', 'mouth', 'upper_face']
    
    for mask_type in mask_types:
        mask = create_roi_mask(dummy_image, mask_type=mask_type)
        print(f"\n{mask_type} mask:")
        print(f"  Shape: {mask.shape}")
        print(f"  Value range: [{mask.min()}, {mask.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(mask)} / {mask.size}")


def example_5_inference():
    """Example 5: Model inference."""
    print("\n" + "="*60)
    print("Example 5: Model Inference")
    print("="*60)
    
    # Build model
    config = Config(config_dict={
        'model': {
            'backbone': 'resnet18',
            'num_classes': 2,
            'pretrained': False,
            'use_roi': False,
            'dropout': 0.5
        }
    })
    
    model = build_model(config)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Inference
    with torch.no_grad():
        outputs = model(dummy_input)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
    
    print(f"\nInference results for batch of {batch_size}:")
    for i in range(batch_size):
        alert_prob = probabilities[i, 0].item()
        drowsy_prob = probabilities[i, 1].item()
        pred_label = "Drowsy" if predictions[i].item() == 1 else "Alert"
        
        print(f"\nSample {i+1}:")
        print(f"  Prediction: {pred_label}")
        print(f"  Alert probability: {alert_prob:.4f}")
        print(f"  Drowsy probability: {drowsy_prob:.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("NTHU Driver Drowsiness Detection - Usage Examples")
    print("="*70)
    
    try:
        example_1_basic_config()
        example_2_build_models()
        example_3_dataset_loading()
        example_4_roi_mask_generation()
        example_5_inference()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Prepare your NTHU-DDD2 dataset")
        print("  2. Run: python train.py --config configs/resnet18_baseline.yaml")
        print("  3. Monitor: tensorboard --logdir experiments/")
        print("  4. Evaluate: python eval.py --checkpoint <path> --split test")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
