"""Model builder utility."""

import torch.nn as nn
from typing import Dict, Any

from .resnet import ResNetDrowsiness, ResNetWithROI
from .efficientnet import EfficientNetDrowsiness, EfficientNetWithROI
from .vgg import VGGDrowsiness, VGGWithROI


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build model from configuration.
    
    Args:
        config: Configuration dictionary or Config object
        
    Returns:
        Initialized model
    """
    # Extract model config
    if hasattr(config, 'config'):
        model_config = config.config.get('model', {})
    else:
        model_config = config.get('model', {})
    
    backbone = model_config.get('backbone', 'resnet18')
    num_classes = model_config.get('num_classes', 2)
    pretrained = model_config.get('pretrained', True)
    dropout = model_config.get('dropout', 0.5)
    use_roi = model_config.get('use_roi', False)
    
    # Build model based on backbone
    if backbone.startswith('resnet'):
        if use_roi:
            model = ResNetWithROI(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
        else:
            model = ResNetDrowsiness(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
    
    elif backbone.startswith('efficientnet'):
        if use_roi:
            model = EfficientNetWithROI(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
        else:
            model = EfficientNetDrowsiness(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
    
    elif backbone.startswith('vgg'):
        if use_roi:
            model = VGGWithROI(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
        else:
            model = VGGDrowsiness(
                backbone=backbone,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. "
                        "Choose from resnet*, efficientnet*, or vgg*")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    num_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {num_params:,}")
    print(f"Input size: {input_size}")
    print(f"\nModel architecture:\n{model}")
