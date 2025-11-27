"""
ROI (Region of Interest) Gating Module.

This module implements attention-based ROI gating for focusing on 
facial regions (eyes, mouth) critical for drowsiness detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttentionGate(nn.Module):
    """
    Spatial attention gate for ROI-based feature weighting.
    Learns to emphasize important regions in the feature map.
    """
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        # attention_map: (B, 1, H, W)
        attention_map = self.conv(x)
        # Gated features: (B, C, H, W)
        return x * attention_map, attention_map


class ROIGatedClassifier(nn.Module):
    """
    Drowsiness classifier with ROI-based gating.
    Integrates a backbone (ResNet) with a spatial attention mechanism.
    """
    
    def __init__(
        self,
        backbone_name='resnet50',
        num_classes=2,
        pretrained=True,
        dropout=0.5
    ):
        super().__init__()
        
        # Load backbone
        if backbone_name == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2]) # Remove AvgPool and FC
            in_channels = 2048
        elif backbone_name == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            in_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Attention Gate
        self.attention_gate = SpatialAttentionGate(in_channels)
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )
        
    def forward(self, x, roi_masks=None):
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            roi_masks: Optional ROI masks (B, num_rois, H, W) - Not used in this simple attention version yet
        
        Returns:
            logits: (B, num_classes)
            attention_map: (B, 1, H, W) - Visualization of where the model is looking
        """
        # Extract features
        features = self.feature_extractor(x) # (B, C, H', W')
        
        # Apply Attention Gating
        gated_features, attention_map = self.attention_gate(features)
        
        # Global Pooling
        pooled = self.global_pool(gated_features)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, attention_map

def create_roi_model(config):
    """
    Factory function to create ROI model from config.
    """
    model_config = config.get('model', {})
    return ROIGatedClassifier(
        backbone_name=model_config.get('architecture', 'resnet50'),
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        dropout=model_config.get('dropout', 0.5)
    )
