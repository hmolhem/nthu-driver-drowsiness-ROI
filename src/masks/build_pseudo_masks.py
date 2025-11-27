import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

class MaskGenerator:
    def __init__(self):
        # Load Haar Cascades
        # cv2.data.haarcascades points to the xml files included in opencv-python
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Mouth cascade is not standard in cv2.data, we might need to approximate or use smile
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def generate_mask(self, image_path, target_size=(224, 224)):
        """
        Generate binary masks for eyes and mouth.
        Returns: (3, H, W) tensor -> [left_eye, right_eye, mouth]
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return torch.zeros((3, *target_size))
                
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize for consistency
            img_resized = cv2.resize(img_gray, target_size)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(img_resized, 1.3, 5)
            
            mask = np.zeros((3, target_size[0], target_size[1]), dtype=np.float32)
            
            for (x, y, w, h) in faces:
                # ROI for face
                roi_gray = img_resized[y:y+h, x:x+w]
                
                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    if i >= 2: break # Limit to 2 eyes
                    # Draw eye on mask (channel 0 and 1)
                    # Simple heuristic: left eye vs right eye based on x coordinate
                    center_x = x + ex + ew//2
                    if center_x < target_size[1] // 2:
                        # Left eye (viewer's left, subject's right) -> Channel 0
                        cv2.circle(mask[0], (center_x, y + ey + eh//2), ew//2, 1.0, -1)
                    else:
                        # Right eye -> Channel 1
                        cv2.circle(mask[1], (center_x, y + ey + eh//2), ew//2, 1.0, -1)
                
                # Detect mouth (using smile cascade as proxy or lower face heuristic)
                # Heuristic: Mouth is usually in the lower half of the face
                roi_gray_lower = roi_gray[h//2:, :]
                smiles = self.smile_cascade.detectMultiScale(roi_gray_lower, 1.8, 20)
                
                for (sx, sy, sw, sh) in smiles:
                    # Adjust coordinates to full image
                    mouth_x = x + sx + sw//2
                    mouth_y = y + h//2 + sy + sh//2
                    cv2.ellipse(mask[2], (mouth_x, mouth_y), (sw//2, sh//3), 0, 0, 360, 1.0, -1)
                    break # Take the first strong smile/mouth detection
            
            return torch.from_numpy(mask)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return torch.zeros((3, *target_size))

def process_dataset(data_root, output_root):
    generator = MaskGenerator()
    
    # Walk through dataset
    data_path = Path(data_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))
    
    print(f"Found {len(image_files)} images. Generating masks...")
    
    for img_path in tqdm(image_files):
        # Generate mask
        mask = generator.generate_mask(img_path)
        
        # Save mask
        # Maintain directory structure
        rel_path = img_path.relative_to(data_path)
        save_path = output_path / rel_path.with_suffix('.pt')
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mask, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/archive")
    parser.add_argument("--output_root", type=str, default="data/masks")
    args = parser.parse_args()
    
    process_dataset(args.data_root, args.output_root)
