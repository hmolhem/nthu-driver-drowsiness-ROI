"""Quick test of dataset and dataloader functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import DrowsinessDataset, create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms

def test_dataset():
    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    # Test single dataset
    train_dataset = DrowsinessDataset(
        csv_path='data/splits/train.csv',
        data_root='datasets',
        transform=get_val_transforms(224)
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    
    # Test getting a sample
    image, label, metadata = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label} ({train_dataset.reverse_label_map[label]})")
    print(f"  Metadata: {metadata}")
    
    # Test class weights
    weights = train_dataset.get_class_weights()
    print(f"\nClass weights: {weights}")
    
    print("\n" + "="*60)
    print("Testing DataLoaders")
    print("="*60)
    
    # Create dataloaders
    loaders = create_dataloaders(
        train_csv='data/splits/train.csv',
        val_csv='data/splits/val.csv',
        test_csv='data/splits/test.csv',
        train_transform=get_train_transforms(224, augment=True),
        val_transform=get_val_transforms(224),
        batch_size=8,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    # Test train loader
    train_loader = loaders['train']
    print(f"\nTrain batches: {len(train_loader)}")
    
    # Get first batch
    images, labels, metadata = next(iter(train_loader))
    print(f"\nFirst batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Subjects: {metadata['subject']}")
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_dataset()
