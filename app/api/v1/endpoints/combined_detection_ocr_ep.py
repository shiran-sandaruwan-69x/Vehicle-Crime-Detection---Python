"""
Combined Detection + OCR Endpoint
Detects license plates in an image, then extracts text from detected plates
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
import tempfile
import os
import cv2
from pathlib import Path

from app.core.dependencies import get_plate_detector, get_ocr_service
from app.models_api.responses import OCRResponse, OCRResult

router = APIRouter()


@router.post("/detect-and-extract", response_model=OCRResponse, tags=["Combined"])
async def detect_and_extract_text(
    image: UploadFile = File(..., description="Image file containing vehicle license plates"),
    confidence_threshold: float = 0.5,
    plate_detector = Depends(get_plate_detector),
    ocr_service = Depends(get_ocr_service)
):
    """
    Detect license plates in an image and extract text from all detected plates.
    
    **Workflow:**
    1. Detect all license plates in the uploaded image
    2. Crop each detected plate
    3. Run OCR on each cropped plate
    4. Return all extracted texts with confidence scores
    
    **Parameters:**
    - **image**: Image file (JPG, PNG, etc.)
    - **confidence_threshold**: Minimum detection confidence (0.0-1.0)
    
    **Returns:**
    - List of extracted texts with their confidence scores and bounding boxes
    """
    temp_image_path = None
    temp_cropped_dir = None
    
    try:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
            content = await image.read()
            temp_file.write(content)
        
        # Create temp directory for cropped plates
        temp_cropped_dir = tempfile.mkdtemp()
        
        # Step 1: Detect plates
        detections = plate_detector.detect(
            temp_image_path, 
            confidence_threshold=confidence_threshold
        )
        
        if not detections:
            return OCRResponse(
                total_plates=0,
                successful_extractions=0,
                failed_extractions=0,
                results=[],
                message="No license plates detected in the image"
            )
        
        # Step 2: Crop each detected plate
        image_cv = cv2.imread(temp_image_path)
        cropped_paths = []
        
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop the plate region
            cropped_plate = image_cv[y1:y2, x1:x2]
            
            # Save cropped plate
            crop_path = os.path.join(temp_cropped_dir, f"plate_{idx}.jpg")
            cv2.imwrite(crop_path, cropped_plate)
            cropped_paths.append({
                'path': crop_path,
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence']
            })
        
        # Step 3: Extract text from each cropped plate
        results = []
        successful = 0
        failed = 0
        
        for crop_info in cropped_paths:
            ocr_result = ocr_service.extract_text(crop_info['path'])
            
            if ocr_result['success']:
                successful += 1
                results.append(OCRResult(
                    text=ocr_result['text'],
                    confidence=ocr_result['confidence'],
                    bbox=crop_info['bbox'],
                    detection_confidence=crop_info['detection_confidence'],
                    success=True
                ))
            else:
                failed += 1
                results.append(OCRResult(
                    text="",
                    confidence=0.0,
                    bbox=crop_info['bbox'],
                    detection_confidence=crop_info['detection_confidence'],
                    success=False,
                    error=ocr_result.get('error', 'OCR extraction failed')
                ))
        
        return OCRResponse(
            total_plates=len(detections),
            successful_extractions=successful,
            failed_extractions=failed,
            results=results,
            message=f"Processed {len(detections)} plate(s): {successful} successful, {failed} failed"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    
    finally:
        # Cleanup temp files
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        if temp_cropped_dir and os.path.exists(temp_cropped_dir):
            for file in Path(temp_cropped_dir).glob("*"):
                os.remove(file)
            os.rmdir(temp_cropped_dir)


@router.post("/detect-and-extract-batch", tags=["Combined"])
async def detect_and_extract_batch(
    images: List[UploadFile] = File(..., description="Multiple image files"),
    confidence_threshold: float = 0.5,
    plate_detector = Depends(get_plate_detector),
    ocr_service = Depends(get_ocr_service)
):
    """
    Process multiple images: detect plates and extract text from each.
    
    **Returns:**
    - List of results for each image
    """
    all_results = []
    
    for image in images:
        try:
            # Process each image using the single endpoint logic
            result = await detect_and_extract_text(
                image=image,
                confidence_threshold=confidence_threshold,
                plate_detector=plate_detector,
                ocr_service=ocr_service
            )
            all_results.append({
                'filename': image.filename,
                'result': result
            })
        except Exception as e:
            all_results.append({
                'filename': image.filename,
                'error': str(e)
            })
    
    return {
        'total_images': len(images),
        'results': all_results
    }