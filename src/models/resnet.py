"""ResNet-based models for drowsiness detection."""

import torch
import torch.nn as nn
from torchvision import models


class ResNetDrowsiness(nn.Module):
    """
    ResNet-based drowsiness detection model.
    
    Supports ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152.
    """
    
    def __init__(self, 
                 backbone: str = 'resnet18',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize ResNet model.
        
        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', etc.)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final classifier
        """
        super(ResNetDrowsiness, self).__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features without classification."""
        return self.backbone(x)


class ResNetWithROI(nn.Module):
    """
    ResNet model with explicit ROI attention mechanism.
    
    Processes ROI regions with additional attention to improve drowsiness detection.
    """
    
    def __init__(self,
                 backbone: str = 'resnet18',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize ResNet with ROI attention.
        
        Args:
            backbone: ResNet variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super(ResNetWithROI, self).__init__()
        
        # Main ResNet backbone
        base_model = ResNetDrowsiness(backbone, num_classes, pretrained, dropout)
        self.backbone = base_model.backbone
        
        # Get feature dimension
        if 'resnet18' in backbone or 'resnet34' in backbone:
            feature_dim = 512
        else:
            feature_dim = 2048
        
        # ROI attention module
        self.roi_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass with ROI attention."""
        # Extract features through backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply ROI attention
        attention_map = self.roi_attention(x)
        x = x * attention_map
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Classification
        output = self.classifier(features)
        return output
