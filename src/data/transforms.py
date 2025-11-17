"""
Data transformations for drowsiness detection dataset.
Includes preprocessing, normalization, and augmentation.
"""

import torch
from torchvision import transforms
import numpy as np


def get_train_transforms(image_size=224, augment=True):
    """
    Get training data transforms with optional augmentation.
    
    Args:
        image_size: Target image size (default: 224 for ResNet/EfficientNet)
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size=224):
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: Target image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


# Robustness test transforms (for future experiments)
def get_robustness_transforms(image_size=224, noise_level='medium'):
    """
    Get transforms for robustness testing.
    
    Args:
        image_size: Target image size
        noise_level: 'low', 'medium', or 'high'
    
    Returns:
        torchvision.transforms.Compose object
    """
    noise_params = {
        'low': {'brightness': 0.1, 'contrast': 0.1, 'blur_kernel': 3},
        'medium': {'brightness': 0.3, 'contrast': 0.3, 'blur_kernel': 5},
        'high': {'brightness': 0.5, 'contrast': 0.5, 'blur_kernel': 7}
    }
    
    params = noise_params.get(noise_level, noise_params['medium'])
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(
            brightness=params['brightness'],
            contrast=params['contrast']
        ),
        transforms.GaussianBlur(kernel_size=params['blur_kernel']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
