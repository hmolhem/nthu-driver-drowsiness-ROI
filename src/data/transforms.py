"""
Data transformations for drowsiness detection dataset.
Includes preprocessing, normalization, and augmentation.

KERAS COMPARISON: transforms replace ImageDataGenerator preprocessing
In Keras: ImageDataGenerator(rescale=1./255, rotation_range=10, ...)
In PyTorch: transforms.Compose([transforms.Resize(), transforms.RandomRotation(), ...])
"""

import torch
from torchvision import transforms
import numpy as np


class GaussianNoise(object):
    """Add gaussian noise to tensor."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_train_transforms(image_size=224, augment=True, augment_config=None):
    """
    Get training data transforms with optional augmentation.
    
    KERAS COMPARISON: This replaces ImageDataGenerator for training
    In Keras: 
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
    
    Args:
        image_size: Target image size (default: 224 for ResNet/EfficientNet)
        augment: Whether to apply data augmentation
        augment_config: Dictionary of augmentation parameters
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform_list = []
    
    if augment:
        # Stronger augmentation: RandomResizedCrop instead of simple Resize
        # This forces the model to learn from parts of the image
        scale = augment_config.get('crop_scale', (0.8, 1.0)) if augment_config else (0.8, 1.0)
        transform_list.append(transforms.RandomResizedCrop(image_size, scale=scale))
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    if augment:
        # Default config if none provided
        config = augment_config or {}
        
        # KERAS COMPARISON: These replace ImageDataGenerator augmentation params
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=config.get('horizontal_flip', 0.5)),
            
            transforms.RandomGrayscale(p=config.get('grayscale_prob', 0.0)),
            
            transforms.ColorJitter(
                brightness=config.get('color_jitter', 0.3),
                contrast=config.get('color_jitter', 0.3),
                saturation=config.get('color_jitter', 0.3),
                hue=config.get('hue', 0.1)
            ),
            
            transforms.RandomRotation(degrees=config.get('rotation_degrees', 15)),
            
            transforms.RandomAffine(
                degrees=0,
                translate=config.get('translate', (0.1, 0.1)),
                scale=config.get('scale', (0.85, 1.15))
            ),
        ])
        
        
        # Optional blur
        if config.get('blur_enabled', False):
            transform_list.append(
                transforms.GaussianBlur(
                    kernel_size=config.get('blur_kernel', 3),
                    sigma=config.get('blur_sigma', (0.1, 2.0))
                )
            )
    
    transform_list.extend([
        transforms.ToTensor(),  # Converts PIL Image to tensor AND scales [0,255] -> [0,1] (like rescale=1./255)
        transforms.Normalize(  # Like preprocessing_function with ImageNet stats
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    if augment and config.get('gaussian_noise', False):
         transform_list.append(GaussianNoise(std=config.get('noise_std', 0.05)))

    
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
