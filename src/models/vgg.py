"""VGG-based models for drowsiness detection."""

import torch
import torch.nn as nn
from torchvision import models


class VGGDrowsiness(nn.Module):
    """
    VGG-based drowsiness detection model.
    
    Classical CNN architecture, good baseline for comparison.
    """
    
    def __init__(self,
                 backbone: str = 'vgg16',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize VGG model.
        
        Args:
            backbone: VGG variant ('vgg11', 'vgg13', 'vgg16', 'vgg19')
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final classifier
        """
        super(VGGDrowsiness, self).__init__()
        
        # Load backbone
        if backbone == 'vgg11':
            self.backbone = models.vgg11(pretrained=pretrained)
        elif backbone == 'vgg13':
            self.backbone = models.vgg13(pretrained=pretrained)
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
        elif backbone == 'vgg19':
            self.backbone = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported VGG backbone: {backbone}")
        
        # VGG feature dimension
        feature_dim = 4096
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features without classification."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class VGGWithROI(nn.Module):
    """
    VGG model with ROI attention mechanism.
    """
    
    def __init__(self,
                 backbone: str = 'vgg16',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize VGG with ROI attention.
        
        Args:
            backbone: VGG variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super(VGGWithROI, self).__init__()
        
        # Load base model
        if backbone == 'vgg11':
            base_model = models.vgg11(pretrained=pretrained)
        elif backbone == 'vgg13':
            base_model = models.vgg13(pretrained=pretrained)
        elif backbone == 'vgg16':
            base_model = models.vgg16(pretrained=pretrained)
        elif backbone == 'vgg19':
            base_model = models.vgg19(pretrained=pretrained)
        else:
            base_model = models.vgg16(pretrained=pretrained)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # ROI attention on feature maps
        self.roi_attention = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        """Forward pass with ROI attention."""
        # Extract features
        x = self.features(x)
        
        # Apply ROI attention
        attention = self.roi_attention(x)
        x = x * attention
        
        # Pool and classify
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
