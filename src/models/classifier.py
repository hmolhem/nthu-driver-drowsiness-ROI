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

        elif architecture.startswith('mobilenet'):
            self.backbone = self._load_mobilenet(architecture, pretrained)
            num_features = self.backbone.classifier[0].in_features
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

    def _load_mobilenet(self, name, pretrained):
        """Load MobileNetV3 backbone."""
        mobilenet_models = {
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'mobilenet_v3_large': models.mobilenet_v3_large,
        }
        
        if name not in mobilenet_models:
            raise ValueError(f"Unknown MobileNet variant: {name}")
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        return mobilenet_models[name](weights=weights)
    
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


class SimpleCNN(nn.Module):
    """
    Lightweight CNN architecture inspired by Kaggle notebook 'CNN-Model_training-Group'.
    
    Architecture:
    1. Input: 224x224x3 (Rescaled 0-1)
    2. Conv2D(32, 3x3) -> MaxPool(2x2)
    3. Conv2D(64, 3x3) -> MaxPool(2x2)
    4. Conv2D(128, 3x3) -> MaxPool(2x2)
    5. Flatten -> Dense(256) -> Dropout(0.5) -> Dense(num_classes)
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3) # Valid padding: 224->222
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)     # 222->111
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 111->109
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)      # 109->54
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # 54->52
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)       # 52->26
        
        # Classifier
        self.flatten = nn.Flatten()
        
        # Input features calculation:
        # Final spatial dim: 26x26
        # Channels: 128
        # Total features: 26 * 26 * 128 = 86528
        self.fc1 = nn.Linear(86528, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


from src.models.roi_gating import create_roi_model

def create_model(config):
    """
    Create model from configuration.
    
    Args:
        config: Configuration dict or DotDict
    
    Returns:
        PyTorch model
    """
    model_config = config.get('model', config)
    architecture = model_config.get('architecture', 'resnet50')
    model_name = model_config.get('name', '')
    
    # Check if it's an ROI model
    # Check if it's an ROI model
    if architecture == 'roi_multistream':
        from src.models.roi_model import get_roi_model
        return get_roi_model(
            num_classes=model_config.get('num_classes', 2),
            pretrained=model_config.get('pretrained', True),
            freeze_backbone=model_config.get('freeze_backbone', False)
        )

    if 'roi' in model_name or config.get('model', {}).get('roi_config', {}).get('enabled', False):
        return create_roi_model(config)
        
    if architecture == 'simple_cnn':
        return SimpleCNN(
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.5)
        )
    
    model = DrowsinessClassifier(
        architecture=architecture,
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', False),
        dropout=model_config.get('dropout', 0.5)
    )
    
    return model
