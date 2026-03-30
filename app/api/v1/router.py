"""
API v1 Router - Aggregates all endpoint modules
"""
from fastapi import APIRouter
from app.api.v1.endpoints import detection_plate_ep, ocr_text_ep, health, combined_detection_ocr_ep

# Create main API router
api_router = APIRouter()

# Include all endpoint routers with their respective tags
api_router.include_router(
    health.router,
    tags=["Health"]
)

api_router.include_router(
    detection_plate_ep.router,
    tags=["Detection"]
)

api_router.include_router(
    ocr_text_ep.router,
    tags=["Batch Processing"]
)

# NEW: Combined detection + OCR endpoint
api_router.include_router(
    combined_detection_ocr_ep.router,
    tags=["Combined Detection & OCR"]
)