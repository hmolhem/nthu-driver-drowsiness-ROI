"""EfficientNet-based models for drowsiness detection."""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetDrowsiness(nn.Module):
    """
    EfficientNet-based drowsiness detection model.
    
    Compact and efficient architecture suitable for edge deployment.
    """
    
    def __init__(self,
                 backbone: str = 'efficientnet_b0',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize EfficientNet model.
        
        Args:
            backbone: EfficientNet variant ('efficientnet_b0' through 'efficientnet_b7')
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final classifier
        """
        super(EfficientNetDrowsiness, self).__init__()
        
        # Load backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif backbone == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        elif backbone == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            feature_dim = 1408
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = 1536
        elif backbone == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            feature_dim = 1792
        elif backbone == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'efficientnet_b6':
            self.backbone = models.efficientnet_b6(pretrained=pretrained)
            feature_dim = 2304
        elif backbone == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(pretrained=pretrained)
            feature_dim = 2560
        else:
            raise ValueError(f"Unsupported EfficientNet backbone: {backbone}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features without classification."""
        # Pass through all layers except classifier
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class EfficientNetWithROI(nn.Module):
    """
    EfficientNet model with ROI attention for drowsiness detection.
    """
    
    def __init__(self,
                 backbone: str = 'efficientnet_b0',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize EfficientNet with ROI attention.
        
        Args:
            backbone: EfficientNet variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super(EfficientNetWithROI, self).__init__()
        
        # Load base model
        if backbone.startswith('efficientnet_b'):
            variant = backbone.split('_')[1]
            feature_dims = {
                'b0': 1280, 'b1': 1280, 'b2': 1408, 'b3': 1536,
                'b4': 1792, 'b5': 2048, 'b6': 2304, 'b7': 2560
            }
            feature_dim = feature_dims.get(variant, 1280)
            
            if backbone == 'efficientnet_b0':
                base_model = models.efficientnet_b0(pretrained=pretrained)
            elif backbone == 'efficientnet_b1':
                base_model = models.efficientnet_b1(pretrained=pretrained)
            elif backbone == 'efficientnet_b2':
                base_model = models.efficientnet_b2(pretrained=pretrained)
            elif backbone == 'efficientnet_b3':
                base_model = models.efficientnet_b3(pretrained=pretrained)
            else:
                base_model = models.efficientnet_b0(pretrained=pretrained)
                feature_dim = 1280
        else:
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # ROI attention
        self.roi_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass with ROI attention."""
        # Extract features
        x = self.features(x)
        
        # Apply attention
        batch_size = x.size(0)
        attention = self.roi_attention(x)
        attention = attention.view(batch_size, -1, 1, 1)
        x = x * attention
        
        # Pool and classify
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
