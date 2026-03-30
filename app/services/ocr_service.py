"""
OCR Service for text extraction
"""
import sys
from pathlib import Path

# Add model_scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model_scripts"))

from ocr_text_plate.ocr_txt import PlateOCR
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from cropped plates"""
    
    def __init__(self, lang: str = 'en'):
        """
        Initialize OCR service
        
        Args:
            lang: Language for OCR
        """
        self.lang = lang
        self.ocr = None
        self._load_model()
    
    def _load_model(self):
        """Load the OCR model"""
        try:
            logger.info(f"Loading OCR model (language: {self.lang})")
            self.ocr = PlateOCR(lang=self.lang)
            logger.info("OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OCR model: {str(e)}")
            raise
    
    def extract_text(self, image_path: str) -> Dict:
        """
        Extract text from cropped plate image
        
        Args:
            image_path: Path to cropped plate image
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if self.ocr is None:
            raise RuntimeError("OCR not initialized")
        
        try:
            result = self.ocr.extract_text(image_path, debug=False)
            logger.debug(f"Extracted text: {result.get('text', 'N/A')}")
            return result
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.ocr is not None