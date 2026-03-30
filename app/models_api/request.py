"""
Request Models
"""
from pydantic import BaseModel, Field
from typing import Optional


class DetectionRequest(BaseModel):
    """Request model for plate detection (future use for additional params)"""
    confidence_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detection"
    )


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection"""
    confidence_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    max_images: Optional[int] = Field(default=10, ge=1, le=50)