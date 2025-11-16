"""Data augmentation for driver drowsiness detection"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(image_size=(224, 224)):
    """
    Training augmentations robust to glare, blur, and occlusion
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5
        ),
        
        # Brightness and contrast for lighting robustness
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Simulate glare
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=2,
            src_radius=100,
            p=0.1
        ),
        
        # Simulate blur
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Color jitter
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Simulate occlusion (random erase)
        A.CoarseDropout(
            max_holes=3,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.2
        ),
    ])


def get_val_augmentation(image_size=(224, 224)):
    """
    Validation/test augmentations (no augmentation, just preprocessing)
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        # No augmentation for validation/test
    ])
