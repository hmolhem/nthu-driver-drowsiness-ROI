"""Simple test script to verify the setup."""

import torch
from src.utils.config import Config
from src.models.builder import build_model, print_model_summary


def test_config():
    """Test configuration system."""
    print("Testing configuration system...")
    config = Config()
    assert config.get('model.backbone') == 'resnet18'
    assert config.get('training.epochs') == 50
    print("✓ Configuration system works")


def test_models():
    """Test model building."""
    print("\nTesting model architectures...")
    
    backbones = [
        'resnet18', 'resnet34', 'resnet50',
        'efficientnet_b0', 'efficientnet_b1',
        'vgg16', 'vgg19'
    ]
    
    for backbone in backbones:
        print(f"\n  Testing {backbone}...")
        config = Config(config_dict={
            'model': {
                'backbone': backbone,
                'num_classes': 2,
                'pretrained': False,  # Don't download weights for testing
                'dropout': 0.5,
                'use_roi': False
            }
        })
        
        model = build_model(config)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
        
        print(f"    ✓ {backbone} works")
    
    print("\n✓ All models working")


def test_roi_models():
    """Test ROI-aware models."""
    print("\nTesting ROI-aware models...")
    
    backbones = ['resnet18', 'efficientnet_b0', 'vgg16']
    
    for backbone in backbones:
        print(f"\n  Testing {backbone} with ROI...")
        config = Config(config_dict={
            'model': {
                'backbone': backbone,
                'num_classes': 2,
                'pretrained': False,
                'dropout': 0.5,
                'use_roi': True
            }
        })
        
        model = build_model(config)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
        
        print(f"    ✓ {backbone} with ROI works")
    
    print("\n✓ All ROI models working")


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics system...")
    from src.utils.metrics import calculate_metrics, MetricsTracker
    import numpy as np
    
    # Test metrics calculation
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_score = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.3])
    
    metrics = calculate_metrics(y_true, y_pred, y_score)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(1, {'train_loss': 0.5, 'val_acc': 0.8})
    assert tracker.get_latest('train_loss') == 0.5
    assert tracker.get_latest('val_acc') == 0.8
    
    print("✓ Metrics system works")


def main():
    """Run all tests."""
    print("="*60)
    print("Running setup verification tests")
    print("="*60)
    
    try:
        test_config()
        test_models()
        test_roi_models()
        test_metrics()
        
        print("\n" + "="*60)
        print("✓ All tests passed! Setup is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
