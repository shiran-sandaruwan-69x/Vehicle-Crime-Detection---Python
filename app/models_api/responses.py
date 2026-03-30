"""
Response Models
Response schemas for OCR endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class OCRResult(BaseModel):
    """Individual OCR result for a single plate"""
    text: str = Field(..., description="Extracted license plate text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] of the plate in original image")
    detection_confidence: float = Field(..., ge=0.0, le=1.0, description="Plate detection confidence")
    success: bool = Field(..., description="Whether OCR extraction was successful")
    error: Optional[str] = Field(None, description="Error message if extraction failed")


class OCRResponse(BaseModel):
    """Response containing all OCR results"""
    total_plates: int = Field(..., description="Total number of plates detected")
    successful_extractions: int = Field(..., description="Number of successful text extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")
    results: List[OCRResult] = Field(..., description="List of OCR results for each detected plate")
    message: str = Field(..., description="Summary message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_plates": 2,
                "successful_extractions": 2,
                "failed_extractions": 0,
                "results": [
                    {
                        "text": "ABC1234",
                        "confidence": 0.95,
                        "bbox": [100, 200, 300, 280],
                        "detection_confidence": 0.98,
                        "success": True
                    },
                    {
                        "text": "XYZ5678",
                        "confidence": 0.89,
                        "bbox": [400, 150, 600, 230],
                        "detection_confidence": 0.92,
                        "success": True
                    }
                ],
                "message": "Processed 2 plate(s): 2 successful, 0 failed"
            }
        }

from pydantic import BaseModel, Field
from typing import List, Optional


class PlateDetection(BaseModel):
    """Single plate detection result"""
    plate_number: int = Field(..., description="Plate index in the image")
    text: str = Field(..., description="Extracted plate text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    success: bool = Field(..., description="Whether extraction was successful")
    error: Optional[str] = Field(None, description="Error message if failed")


class DetectionResponse(BaseModel):
    """Response model for single image detection"""
    success: bool = Field(..., description="Overall success status")
    message: str = Field(..., description="Status message")
    detected_plates: List[PlateDetection] = Field(default_factory=list)
    total_plates: int = Field(0, description="Total plates detected")
    successful_extractions: int = Field(0, description="Successfully extracted texts")


class ImageResult(BaseModel):
    """Result for single image in batch"""
    filename: str
    success: bool
    detected_plates: Optional[List[PlateDetection]] = None
    plate_count: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


class BatchDetectionResponse(BaseModel):
    """Response model for batch detection"""
    total_images: int
    total_plates_detected: int
    results: List[ImageResult]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    version: str = Field(..., description="API version")