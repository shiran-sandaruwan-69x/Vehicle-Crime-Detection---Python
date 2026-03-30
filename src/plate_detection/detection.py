import torch
import cv2
from ultralytics import YOLO
from pathlib import Path

class PlateDetector:
    def __init__(self, model_path=None):
        """Initialize plate detector with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use trained model or default
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / 'models' / 'plate_detection' / 'best.pt'
        
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def detect(self, img):
        """Detect license plate with confidence threshold."""
        try:
            with torch.no_grad():
                results = self.model(img, device=self.device, verbose=False, conf=0.5)
                
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                conf = boxes.conf[0].item()
                box = boxes.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, box)
                plate_crop = img[y1:y2, x1:x2]
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return plate_crop, conf
            
            return None, 0.0
        
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, 0.0