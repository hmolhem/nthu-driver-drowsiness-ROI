import torch
import torch.nn as nn
import torchvision.models as models

class ThreeStreamNetwork(nn.Module):
    """
    Multi-stream network processing Left Eye, Right Eye, and Mouth separately.
    Uses MobileNetV3-Small as backbone for each stream.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(ThreeStreamNetwork, self).__init__()
        
        # Define backbones
        self.left_eye_net = self._create_backbone(pretrained, freeze_backbone)
        self.right_eye_net = self._create_backbone(pretrained, freeze_backbone)
        self.mouth_net = self._create_backbone(pretrained, freeze_backbone)
        
        # Feature dimension for MobileNetV3-Small is 576
        self.feature_dim = 576
        
        # Fusion and Classifier
        # Concatenate features from 3 streams: 576 * 3 = 1728
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def _create_backbone(self, pretrained, freeze):
        # Load MobileNetV3 Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        # Remove classifier, keep only features and pooling
        # MobileNetV3 features output (B, 576, 1, 1) after pooling
        # We will use model.features and then adaptive avg pool
        
        backbone = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
                
        return backbone

    def forward(self, x):
        """
        Args:
            x (dict): Dictionary containing 'left_eye', 'right_eye', 'mouth' tensors.
        """
        # Check if input is a dict (expected)
        if not isinstance(x, dict):
             # Fallback if someone passes a single tensor (e.g. summary writer)
             # We assume it's one of the crops or the full image, but this model needs 3 inputs.
             # We'll just duplicate it to avoid crash, but this is not ideal.
             x = {'left_eye': x, 'right_eye': x, 'mouth': x}

        l_feat = self.left_eye_net(x['left_eye'])
        r_feat = self.right_eye_net(x['right_eye'])
        m_feat = self.mouth_net(x['mouth'])
        
        # Concatenate
        combined = torch.cat([l_feat, r_feat, m_feat], dim=1)
        
        return self.classifier(combined)

def get_roi_model(num_classes=2, pretrained=True, freeze_backbone=False):
    return ThreeStreamNetwork(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
