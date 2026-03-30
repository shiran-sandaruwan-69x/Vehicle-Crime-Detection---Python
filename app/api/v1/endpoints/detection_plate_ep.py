"""
Plate Detection Endpoint
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path

from app.models_api.responses import DetectionResponse, PlateDetection
from app.core.dependencies import get_plate_detector, get_ocr_service
from app.services.plate_detector import PlateDetectorService
from app.services.ocr_service import OCRService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect License Plate",
    description="Upload a vehicle image to detect and extract license plate text"
)
async def detect_plate(
    file: UploadFile = File(..., description="Vehicle image (JPG, JPEG, PNG)"),
    plate_detector: PlateDetectorService = Depends(get_plate_detector),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    Detect and extract license plate text from uploaded vehicle image.
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPG, JPEG, PNG)"
        )
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file
            input_path = os.path.join(temp_dir, "input_image.jpg")
            contents = await file.read()
            
            # Convert to numpy array and save
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not decode image. Please upload a valid image file."
                )
            
            cv2.imwrite(input_path, img)
            
            # Step 1: Detect and crop plates
            cropped_dir = os.path.join(temp_dir, "cropped")
            os.makedirs(cropped_dir, exist_ok=True)
            
            logger.info(f"Processing image: {file.filename}")
            cropped_paths = plate_detector.detect_and_crop(input_path, cropped_dir)
            
            if not cropped_paths:
                return DetectionResponse(
                    success=False,
                    message="No license plates detected in the image",
                    detected_plates=[],
                    total_plates=0,
                    successful_extractions=0
                )
            
            # Step 2: Extract text from each cropped plate
            detected_plates = []
            
            for idx, cropped_path in enumerate(cropped_paths):
                extraction = ocr_service.extract_text(cropped_path)
                
                plate_data = PlateDetection(
                    plate_number=idx + 1,
                    text=extraction.get('text', ''),
                    confidence=extraction.get('confidence', 0.0),
                    success=extraction.get('success', False),
                    error=extraction.get('error') if not extraction.get('success') else None
                )
                
                detected_plates.append(plate_data)
            
            # Prepare response
            successful_extractions = sum(1 for p in detected_plates if p.success)
            
            return DetectionResponse(
                success=successful_extractions > 0,
                message=f"Detected {len(cropped_paths)} plate(s), extracted {successful_extractions} successfully",
                detected_plates=detected_plates,
                total_plates=len(cropped_paths),
                successful_extractions=successful_extractions
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Internal server error: {str(e)}"
            )