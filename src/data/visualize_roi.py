import os
import cv2
import pandas as pd
import torch
import numpy as np
from PIL import Image
from roi_transforms import ROICrop
from torchvision import transforms

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_roi(data_root, csv_path, output_dir='roi_visualizations', num_samples=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    
    # Sample random images
    samples = df.sample(n=num_samples)
    
    roi_cropper = ROICrop()
    
    for idx, row in samples.iterrows():
        img_path = os.path.join(data_root, row['filename'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
            
        # Get crops
        crops = roi_cropper(img)
        
        # Visualize
        # Original image
        img.save(os.path.join(output_dir, f"sample_{idx}_original.jpg"))
        
        # Save crops
        for key, tensor in crops.items():
            # Denormalize
            denorm_tensor = denormalize(tensor)
            # Convert to PIL
            ndarr = denorm_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(os.path.join(output_dir, f"sample_{idx}_{key}.jpg"))
            
        print(f"Saved visualization for {row['filename']}")

if __name__ == "__main__":
    # Adjust paths as needed
    DATA_ROOT = 'datasets' 
    CSV_PATH = 'data/splits/train.csv'
    
    # Check if paths exist
    if not os.path.exists(DATA_ROOT):
        # Try finding it relative to project root if running from src/data
        if os.path.exists('../../datasets'):
            DATA_ROOT = '../../datasets'
            CSV_PATH = '../../data/splits/train.csv'
        else:
            print(f"Data root {DATA_ROOT} not found.")
            exit(1)
            
    visualize_roi(DATA_ROOT, CSV_PATH)
