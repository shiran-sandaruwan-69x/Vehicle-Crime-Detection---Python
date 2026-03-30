import easyocr
import re
from src.utils import logger, load_config, preprocess_image
import numpy as np

CONFIG = load_config()

class TextExtractor:
    def __init__(self):
        # Use 'model' (singular) and correct 'ocr' key
        ocr_config = CONFIG['model']['ocr']
        self.reader = easyocr.Reader(ocr_config['languages'], gpu=ocr_config['gpu'])
        logger.info("EasyOCR reader initialized")

    def extract(self, plate_crop: np.ndarray) -> tuple:
        """Extract text from crop, clean, return string and avg conf."""
        preprocessed = preprocess_image(plate_crop)
        ocr_results = self.reader.readtext(preprocessed)
        
        if not ocr_results:
            logger.warning("No text extracted")
            return "", 0.0
        
        text_parts = [res[1] for res in ocr_results]
        full_text = ''.join(text_parts).upper().replace(' ', '')
        cleaned = re.sub(r'[^A-Z0-9]', '', full_text)
        avg_conf = sum(res[2] for res in ocr_results) / len(ocr_results)
        
        ocr_conf_threshold = CONFIG.get('thresholds', {}).get('ocr_conf', 0.5)
        if avg_conf < ocr_conf_threshold:
            logger.warning(f"Low OCR conf: {avg_conf}")
        
        logger.info(f"Extracted text: {cleaned} with conf: {avg_conf}")
        return cleaned, avg_conf