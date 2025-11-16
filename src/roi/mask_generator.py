"""ROI (Region of Interest) mask generation for eye and mouth regions"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import dlib
from PIL import Image


class ROIMaskGenerator:
    """
    Generate pseudo eye and mouth segmentation masks for driver faces
    
    Uses facial landmarks to identify regions of interest (eyes and mouth)
    which are critical for drowsiness detection.
    """
    
    def __init__(self, predictor_path: Optional[str] = None):
        """
        Args:
            predictor_path: Path to dlib shape predictor model
                          (e.g., shape_predictor_68_face_landmarks.dat)
        """
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize dlib predictor if available
        self.predictor = None
        if predictor_path and Path(predictor_path).exists():
            try:
                self.predictor = dlib.shape_predictor(predictor_path)
                self.face_detector = dlib.get_frontal_face_detector()
            except Exception as e:
                print(f"Warning: Could not load dlib predictor: {e}")
                print("Falling back to simple ROI estimation")
    
    def generate_mask(
        self,
        image: np.ndarray,
        expand_ratio: float = 1.2
    ) -> np.ndarray:
        """
        Generate ROI mask highlighting eye and mouth regions
        
        Args:
            image: Input image as numpy array (H, W, C)
            expand_ratio: Factor to expand ROI regions
        
        Returns:
            Binary mask of same size as image (H, W) with ROI regions set to 255
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.predictor is not None:
            # Use dlib for precise landmark detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector(gray)
            
            if len(faces) > 0:
                face = faces[0]  # Use first detected face
                landmarks = self.predictor(gray, face)
                
                # Extract eye regions (landmarks 36-47)
                left_eye = self._get_eye_region(landmarks, list(range(36, 42)))
                right_eye = self._get_eye_region(landmarks, list(range(42, 48)))
                
                # Extract mouth region (landmarks 48-67)
                mouth = self._get_mouth_region(landmarks, list(range(48, 68)))
                
                # Draw filled regions
                if left_eye is not None:
                    cv2.fillPoly(mask, [left_eye], 255)
                if right_eye is not None:
                    cv2.fillPoly(mask, [right_eye], 255)
                if mouth is not None:
                    cv2.fillPoly(mask, [mouth], 255)
        else:
            # Fallback: Use simple face detection + heuristic ROI
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                x, y, fw, fh = faces[0]  # Use first detected face
                
                # Estimate eye regions (upper 40% of face)
                eye_y_start = y + int(fh * 0.2)
                eye_y_end = y + int(fh * 0.5)
                
                # Left eye (left 45% of face width)
                left_eye_x_start = x + int(fw * 0.1)
                left_eye_x_end = x + int(fw * 0.45)
                mask[eye_y_start:eye_y_end, left_eye_x_start:left_eye_x_end] = 255
                
                # Right eye (right 45% of face width)
                right_eye_x_start = x + int(fw * 0.55)
                right_eye_x_end = x + int(fw * 0.9)
                mask[eye_y_start:eye_y_end, right_eye_x_start:right_eye_x_end] = 255
                
                # Mouth region (lower 30% of face)
                mouth_y_start = y + int(fh * 0.65)
                mouth_y_end = y + int(fh * 0.9)
                mouth_x_start = x + int(fw * 0.25)
                mouth_x_end = x + int(fw * 0.75)
                mask[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = 255
        
        # Apply morphological operations to smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _get_eye_region(
        self,
        landmarks,
        indices: List[int],
        expand: float = 1.3
    ) -> Optional[np.ndarray]:
        """Extract eye region polygon from landmarks"""
        try:
            points = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                             for i in indices])
            
            # Expand region
            center = points.mean(axis=0)
            points = center + (points - center) * expand
            
            return points.astype(np.int32)
        except:
            return None
    
    def _get_mouth_region(
        self,
        landmarks,
        indices: List[int],
        expand: float = 1.2
    ) -> Optional[np.ndarray]:
        """Extract mouth region polygon from landmarks"""
        try:
            points = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                             for i in indices])
            
            # Expand region
            center = points.mean(axis=0)
            points = center + (points - center) * expand
            
            return points.astype(np.int32)
        except:
            return None
    
    def generate_dataset_masks(
        self,
        image_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = ('.jpg', '.png', '.jpeg')
    ):
        """
        Generate ROI masks for all images in a directory
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save generated masks
            extensions: Image file extensions to process
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.rglob(f"*{ext}"))
        
        print(f"Generating ROI masks for {len(image_paths)} images...")
        
        for img_path in image_paths:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate mask
                mask = self.generate_mask(image)
                
                # Save mask maintaining directory structure
                relative_path = img_path.relative_to(image_dir)
                output_path = output_dir / relative_path.with_suffix('.png')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(output_path), mask)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print("ROI mask generation complete!")


def visualize_roi_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Visualize ROI mask overlaid on image
    
    Args:
        image: Original image (H, W, C)
        mask: ROI mask (H, W)
        alpha: Transparency of mask overlay
    
    Returns:
        Visualization image
    """
    # Create colored mask (red for ROI)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask  # Red channel
    
    # Blend with original image
    visualization = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return visualization
