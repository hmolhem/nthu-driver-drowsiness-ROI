import torch
from torchvision import models

model = models.mobilenet_v3_small(pretrained=True)
print(model)

# Check classifier structure
print("\nClassifier:")
print(model.classifier)

# Check output shape
x = torch.randn(1, 3, 224, 224)
features = model.features(x)
print(f"\nFeatures shape: {features.shape}")
# MobileNetV3 usually has an avgpool after features
features = model.avgpool(features)
print(f"AvgPool shape: {features.shape}")
features = torch.flatten(features, 1)
print(f"Flatten shape: {features.shape}")
