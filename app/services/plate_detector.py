"""
Plate Detection Service
"""
import sys
from pathlib import Path

# Add model_scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model_scripts"))

from plate_detection_crop.detection_crop import PlateDetectionCropper
from typing import List
import logging

logger = logging.getLogger(__name__)


class PlateDetectorService:
    """Service for detecting and cropping license plates"""
    
    def __init__(self, model_path: str, confidence: float = 0.5):
        """
        Initialize plate detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold
        """
        self.model_path = model_path
        self.confidence = confidence
        self.detector = None
        self._load_model()
    
    def _load_model(self):
        """Load the detection model"""
        try:
            logger.info(f"Loading plate detection model from {self.model_path}")
            self.detector = PlateDetectionCropper(self.model_path)
            logger.info("Plate detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load plate detection model: {str(e)}")
            raise
    
    def detect(self, image_path: str, confidence_threshold: float = None) -> List[dict]:
        """
        Detect plates and return detection results
        
        Args:
            image_path: Path to input image
            confidence_threshold: Detection confidence threshold (uses self.confidence if None)
            
        Returns:
            List of detections with bbox and confidence
        """
        if self.detector is None:
            raise RuntimeError("Detector not initialized")
        
        try:
            conf = confidence_threshold if confidence_threshold is not None else self.confidence
            
            # Run detection
            results = self.detector.model.predict(image_path, conf=conf, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
            
            logger.info(f"Detected {len(detections)} plates in {image_path}")
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting plates: {str(e)}")
            raise        

    def detect_and_crop(self, image_path: str, output_dir: str) -> List[str]:
        """
        Detect plates and crop them
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save cropped plates
            
        Returns:
            List of paths to cropped plate images
        """
        if self.detector is None:
            raise RuntimeError("Detector not initialized")
        
        try:
            cropped_paths = self.detector.crop_plates(
                image_path, 
                output_dir, 
                confidence=self.confidence
            )
            logger.info(f"Detected {len(cropped_paths)} plates in {image_path}")
            return cropped_paths
        except Exception as e:
            logger.error(f"Error detecting plates: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.detector is not None