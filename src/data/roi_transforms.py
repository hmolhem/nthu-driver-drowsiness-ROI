import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class ROICrop(object):
    """
    Crop Region of Interest (ROI) from the image: Left Eye, Right Eye, Mouth.
    Uses OpenCV Haar Cascades for detection.
    """
    def __init__(self, output_size=(64, 64)):
        self.output_size = output_size
        
        # Load Haar Cascades
        # Note: We assume cv2.data.haarcascades is available
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Mouth detection is trickier with Haar, often people use smile or split face
        # We will try to estimate mouth position relative to face if specific cascade fails
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.
            
        Returns:
            dict: {'left_eye': tensor, 'right_eye': tensor, 'mouth': tensor}
                  If detection fails, returns black tensors or center crops.
        """
        # Convert to numpy array for OpenCV
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy() * 255
            img_np = img_np.astype(np.uint8)
        elif isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        crops = {}
        
        if len(faces) > 0:
            # Take the largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_np[y:y+h, x:x+w]
            
            # Heuristic for eyes: Top half of the face
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Sort eyes by x position
            eyes = sorted(eyes, key=lambda e: e[0])
            
            if len(eyes) >= 2:
                # Left eye (on the left side of image, which is right eye of person)
                # But for consistency let's just take left-most and right-most
                left_eye_rect = eyes[0]
                right_eye_rect = eyes[-1]
                
                crops['left_eye'] = self._crop_and_resize(roi_color, left_eye_rect)
                crops['right_eye'] = self._crop_and_resize(roi_color, right_eye_rect)
            else:
                # Fallback: Crop upper left and upper right quadrants of face
                h_f, w_f = roi_color.shape[:2]
                crops['left_eye'] = self._crop_and_resize(roi_color, (int(w_f*0.1), int(h_f*0.2), int(w_f*0.35), int(h_f*0.3)))
                crops['right_eye'] = self._crop_and_resize(roi_color, (int(w_f*0.55), int(h_f*0.2), int(w_f*0.35), int(h_f*0.3)))

            # Mouth: Lower third of the face
            # Try smile detection restricted to lower half
            roi_gray_lower = roi_gray[int(h*0.6):, :]
            smiles = self.smile_cascade.detectMultiScale(roi_gray_lower, 1.8, 20)
            
            if len(smiles) > 0:
                (sx, sy, sw, sh) = smiles[0]
                # Adjust y coordinate to be relative to face ROI
                sy += int(h*0.6)
                crops['mouth'] = self._crop_and_resize(roi_color, (sx, sy, sw, sh))
            else:
                # Fallback: Crop bottom center
                h_f, w_f = roi_color.shape[:2]
                crops['mouth'] = self._crop_and_resize(roi_color, (int(w_f*0.25), int(h_f*0.7), int(w_f*0.5), int(h_f*0.25)))
                
        else:
            # No face detected: Return center crops or black images
            # For now, let's return center crops of the whole image as a fallback
            h, w = img_np.shape[:2]
            crops['left_eye'] = self._crop_and_resize(img_np, (int(w*0.2), int(h*0.2), int(w*0.2), int(h*0.2)))
            crops['right_eye'] = self._crop_and_resize(img_np, (int(w*0.6), int(h*0.2), int(w*0.2), int(h*0.2)))
            crops['mouth'] = self._crop_and_resize(img_np, (int(w*0.4), int(h*0.7), int(w*0.2), int(h*0.2)))

        # Convert back to tensor and normalize
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return {k: to_tensor(v) for k, v in crops.items()}

    def _crop_and_resize(self, img, rect):
        (x, y, w, h) = rect
        # Add some padding
        pad_w = int(w * 0.1)
        pad_h = int(h * 0.1)
        
        img_h, img_w = img.shape[:2]
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        crop = img[y1:y2, x1:x2]
        
        # Resize
        crop = cv2.resize(crop, self.output_size)
        return crop

def get_roi_transforms(output_size=(64, 64)):
    return ROICrop(output_size=output_size)
