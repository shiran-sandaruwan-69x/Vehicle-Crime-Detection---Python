"""
Batch Detection Endpoint
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import cv2
import numpy as np
import os
import tempfile
from typing import List

from app.models.responses import BatchDetectionResponse, ImageResult, PlateDetection
from app.core.dependencies import get_plate_detector, get_ocr_service
from app.services.plate_detector import PlateDetectorService
from app.services.ocr_service import OCRService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/detect-batch",
    response_model=BatchDetectionResponse,
    summary="Batch Plate Detection",
    description="Upload multiple vehicle images for batch processing"
)
async def detect_batch(
    files: List[UploadFile] = File(..., description="Multiple vehicle images"),
    plate_detector: PlateDetectorService = Depends(get_plate_detector),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    Batch processing: Detect and extract license plates from multiple images.
    """
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 images allowed per batch"
        )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        results = []
        total_plates_detected = 0
        
        for file_idx, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    results.append(ImageResult(
                        filename=file.filename,
                        success=False,
                        error="File must be an image"
                    ))
                    continue
                
                # Save uploaded file
                input_path = os.path.join(temp_dir, f"input_{file_idx}.jpg")
                contents = await file.read()
                
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    results.append(ImageResult(
                        filename=file.filename,
                        success=False,
                        error="Could not decode image"
                    ))
                    continue
                
                cv2.imwrite(input_path, img)
                
                # Detect and crop
                cropped_dir = os.path.join(temp_dir, f"cropped_{file_idx}")
                os.makedirs(cropped_dir, exist_ok=True)
                
                cropped_paths = plate_detector.detect_and_crop(input_path, cropped_dir)
                
                if not cropped_paths:
                    results.append(ImageResult(
                        filename=file.filename,
                        success=False,
                        message="No plates detected"
                    ))
                    continue
                
                # Extract text
                detected_plates = []
                for idx, cropped_path in enumerate(cropped_paths):
                    extraction = ocr_service.extract_text(cropped_path)
                    detected_plates.append(PlateDetection(
                        plate_number=idx + 1,
                        text=extraction.get('text', ''),
                        confidence=extraction.get('confidence', 0.0),
                        success=extraction.get('success', False),
                        error=extraction.get('error') if not extraction.get('success') else None
                    ))
                
                total_plates_detected += len(cropped_paths)
                
                results.append(ImageResult(
                    filename=file.filename,
                    success=True,
                    detected_plates=detected_plates,
                    plate_count=len(cropped_paths)
                ))
            
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append(ImageResult(
                    filename=file.filename,
                    success=False,
                    error=str(e)
                ))
        
        return BatchDetectionResponse(
            total_images=len(files),
            total_plates_detected=total_plates_detected,
            results=results
        )