import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.getcwd())
from src.masks.build_pseudo_masks import MaskGenerator

def visualize_mask(image_path, output_path):
    generator = MaskGenerator()
    
    # Generate mask
    print(f"Processing {image_path}...")
    mask_tensor = generator.generate_mask(image_path) # (3, H, W)
    
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        print("Error loading image")
        return
    
    # Resize image to match mask (224, 224)
    img_resized = cv2.resize(img, (224, 224))
    
    # Convert mask to numpy
    mask_np = mask_tensor.numpy()
    
    # Create overlay
    # Channel 0: Left Eye (Blue)
    # Channel 1: Right Eye (Green)
    # Channel 2: Mouth (Red)
    
    overlay = img_resized.copy()
    
    # Apply color overlays
    # Blue for Left Eye
    overlay[mask_np[0] > 0] = [255, 0, 0] 
    # Green for Right Eye
    overlay[mask_np[1] > 0] = [0, 255, 0]
    # Red for Mouth
    overlay[mask_np[2] > 0] = [0, 0, 255]
    
    # Blend
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img_resized, 1 - alpha, 0, img_resized)
    
    # Save
    cv2.imwrite(str(output_path), img_resized)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    
    visualize_mask(args.image_path, args.output_path)
