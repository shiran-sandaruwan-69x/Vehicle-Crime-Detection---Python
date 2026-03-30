"""
Dependency injection for services
"""
from functools import lru_cache
from app.services.plate_detector import PlateDetectorService
from app.services.ocr_service import OCRService
from app.core.config import get_settings


@lru_cache()
def get_plate_detector() -> PlateDetectorService:
    """Get plate detector service instance (singleton)"""
    settings = get_settings()
    return PlateDetectorService(
        model_path=settings.PLATE_DETECTION_MODEL_PATH,
        confidence=settings.DETECTION_CONFIDENCE
    )


@lru_cache()
def get_ocr_service() -> OCRService:
    """Get OCR service instance (singleton)"""
    settings = get_settings()
    return OCRService(lang=settings.OCR_LANGUAGE)