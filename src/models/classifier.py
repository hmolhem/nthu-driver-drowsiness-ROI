"""Model architectures for drowsiness detection."""

import torch
import torch.nn as nn
from torchvision import models


# KERAS COMPARISON: nn.Module is like keras.Model
# Instead of model.add() or Sequential(), you define layers in __init__
# and connect them in forward() method
class DrowsinessClassifier(nn.Module):
    """
    Base drowsiness classifier using pretrained backbones.
    
    Supports:
    - ResNet (resnet18, resnet34, resnet50, resnet101)
    - EfficientNet (efficientnet_b0, efficientnet_b1, efficientnet_b2)
    """
    
    def __init__(
        self,
        architecture='resnet50',
        num_classes=2,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.5
    ):
        """
        Initialize classifier.
        
        Args:
            architecture: Backbone architecture name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            dropout: Dropout rate before final classifier
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Load backbone
        if architecture.startswith('resnet'):
            self.backbone = self._load_resnet(architecture, pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
        
        elif architecture.startswith('efficientnet'):
            self.backbone = self._load_efficientnet(architecture, pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def _load_resnet(self, name, pretrained):
        """Load ResNet backbone."""
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }
        
        if name not in resnet_models:
            raise ValueError(f"Unknown ResNet variant: {name}")
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        return resnet_models[name](weights=weights)
    
    def _load_efficientnet(self, name, pretrained):
        """Load EfficientNet backbone."""
        efficientnet_models = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
        }
        
        if name not in efficientnet_models:
            raise ValueError(f"Unknown EfficientNet variant: {name}")
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        return efficientnet_models[name](weights=weights)
    
    def forward(self, x):
        """
        Forward pass.
        
        KERAS COMPARISON: forward() is like Keras model's call() method
        In Keras: output = model(input)  # calls model.call()
        In PyTorch: output = model(input)  # calls model.forward()
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Logits (B, num_classes)
        """
        features = self.backbone(x)  # Like: x = base_model(x)
        logits = self.classifier(features)  # Like: output = Dense(2)(x)
        return logits
    
    def get_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Feature vectors (B, num_features)
        """
        return self.backbone(x)


def create_model(config):
    """
    Create model from configuration.
    
    Args:
        config: Configuration dict or DotDict
    
    Returns:
        PyTorch model
    """
    model_config = config.get('model', config)
    
    model = DrowsinessClassifier(
        architecture=model_config.get('architecture', 'resnet50'),
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', False),
        dropout=model_config.get('dropout', 0.5)
    )
    
    return model
