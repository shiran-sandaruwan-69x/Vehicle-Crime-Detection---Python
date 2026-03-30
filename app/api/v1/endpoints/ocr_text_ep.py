from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import os
import shutil
from pathlib import Path
import tempfile
import sys

# Add model_scripts to path
sys.path.append(str(Path(__file__).resolve().parents[4] / "model_scripts"))

from ocr_text_plate.ocr_txt import PlateOCR

router = APIRouter()

# Initialize OCR model (singleton pattern)
ocr_model = None

def get_ocr_model():
    """Get or initialize OCR model (uses PaddleOCR internally)"""
    global ocr_model
    if ocr_model is None:
        print("Initializing PaddleOCR model...")
        ocr_model = PlateOCR(lang='en')
        print("PaddleOCR model loaded successfully!")
    return ocr_model


@router.post("/extract-text")
async def extract_plate_text(
    file: UploadFile = File(..., description="Cropped plate image file")
):
    """
    Extract text from a cropped vehicle plate image using PaddleOCR.
    
    Args:
        file: Image file (jpg, jpeg, png)
    
    Returns:
        JSON response with extracted text and confidence score
    """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Get OCR model
        ocr = get_ocr_model()
        
        # Extract text
        result = ocr.extract_text(temp_path, debug=False)
        
        # Prepare response
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "text": result['text'],
                    "confidence": result['confidence'],
                    "filename": file.filename
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "error": result.get('error', 'Unknown error'),
                    "filename": file.filename
                }
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {e}")


@router.post("/extract-text-batch")
async def extract_plate_text_batch(
    files: list[UploadFile] = File(..., description="Multiple cropped plate images")
):
    """
    Extract text from multiple cropped vehicle plate images using PaddleOCR.
    
    Args:
        files: List of image files (jpg, jpeg, png)
    
    Returns:
        JSON response with extracted texts for all images
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    
    results = {
        'total_images': len(files),
        'successful': 0,
        'failed': 0,
        'extractions': []
    }
    
    # Get OCR model
    ocr = get_ocr_model()
    
    # Process each file
    temp_paths = []
    
    try:
        for idx, file in enumerate(files, 1):
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext not in allowed_extensions:
                results['extractions'].append({
                    'filename': file.filename,
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': f'Invalid file type: {file_ext}'
                })
                results['failed'] += 1
                continue
            
            temp_path = None
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                    temp_path = temp_file.name
                    temp_paths.append(temp_path)
                
                # Extract text
                result = ocr.extract_text(temp_path, debug=False)
                
                # Add to results
                extraction = {
                    'filename': file.filename,
                    'success': result['success'],
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0.0)
                }
                
                if not result['success']:
                    extraction['error'] = result.get('error', 'Unknown error')
                    results['failed'] += 1
                else:
                    results['successful'] += 1
                
                results['extractions'].append(extraction)
            
            except Exception as e:
                results['extractions'].append({
                    'filename': file.filename,
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': str(e)
                })
                results['failed'] += 1
    
    finally:
        # Clean up all temporary files
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_path}: {e}")
    
    return JSONResponse(
        status_code=200,
        content=results
    )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for OCR service (PaddleOCR).
    """
    try:
        ocr = get_ocr_model()
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "OCR Text Extraction (PaddleOCR)",
                "model_loaded": ocr is not None,
                "backend": "PaddleOCR"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "OCR Text Extraction",
                "error": str(e)
            }
        )


@router.get("/info")
async def ocr_info():
    """
    Get information about the OCR service.
    """
    return JSONResponse(
        content={
            "service": "Vehicle Plate OCR",
            "backend": "PaddleOCR",
            "language": "English",
            "wrapper_class": "PlateOCR",
            "supported_formats": ["jpg", "jpeg", "png"],
            "endpoints": {
                "single": "/extract-text",
                "batch": "/extract-text-batch",
                "health": "/health"
            }
        }
    )