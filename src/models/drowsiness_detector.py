"""CNN model architectures for driver drowsiness detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class ROIAttentionModule(nn.Module):
    """
    Attention module to focus on ROI (eye/mouth) regions
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
        
        Returns:
            Attention-weighted features
        """
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class DrowsinessDetector(nn.Module):
    """
    CNN-based drowsiness detector with configurable backbone
    
    Supports ResNet, EfficientNet, and VGG backbones with optional
    ROI attention mechanism.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        use_roi_attention: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: Backbone architecture name
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before classifier
            use_roi_attention: Whether to use ROI attention module
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_roi_attention = use_roi_attention
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.shape[1]
            self.feature_size = dummy_output.shape[2:]
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ROI attention module
        if use_roi_attention:
            self.roi_attention = ROIAttentionModule(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            Logits (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)  # (B, C, H', W')
        
        # Apply ROI attention
        if self.use_roi_attention:
            features = self.roi_attention(features)
        
        # Global pooling
        pooled = self.global_pool(features)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features = self.backbone(x)
        if self.use_roi_attention:
            features = self.roi_attention(features)
        pooled = self.global_pool(features)
        return pooled.flatten(1)


def create_model(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> DrowsinessDetector:
    """
    Factory function to create drowsiness detection models
    
    Supported backbones:
        - resnet18, resnet34, resnet50, resnet101
        - efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
        - vgg16, vgg19
        - mobilenetv3_small, mobilenetv3_large
    
    Args:
        backbone: Backbone architecture name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        **kwargs: Additional arguments for DrowsinessDetector
    
    Returns:
        DrowsinessDetector model
    """
    model = DrowsinessDetector(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
    return model


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Args:
            models: List of DrowsinessDetector models
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images
        
        Returns:
            Weighted average of model predictions
        """
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)
        
        return sum(outputs)
