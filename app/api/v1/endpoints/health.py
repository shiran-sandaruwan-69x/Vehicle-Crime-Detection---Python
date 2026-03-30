"""
Health Check Endpoint
"""
from fastapi import APIRouter, Depends
from app.models_api.responses import HealthResponse
from app.core.config import get_settings, Settings
from app.core.dependencies import get_plate_detector, get_ocr_service
from app.services.plate_detector import PlateDetectorService
from app.services.ocr_service import OCRService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and ML models are healthy"
)
async def health_check(
    settings: Settings = Depends(get_settings),
    plate_detector: PlateDetectorService = Depends(get_plate_detector),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """Health check endpoint"""
    
    models_loaded = plate_detector.is_loaded() and ocr_service.is_loaded()
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        version=settings.API_VERSION
    )