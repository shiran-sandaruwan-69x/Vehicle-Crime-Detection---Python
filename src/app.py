from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from .plate_detection.detection import PlateDetector  # Use relative import
from .ocr_recognition.ocr import TextExtractor  # Relative import
from .utils.utils import logger, load_config  # Relative import

app = FastAPI(title="ANPR Service")
detector = PlateDetector()
extractor = TextExtractor()

@app.post("/anpr")
async def anpr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        plate_crop, det_conf = detector.detect(img)
        if plate_crop is None:
            raise HTTPException(status_code=404, detail="No plate detected")
        
        plate_text, ocr_conf = extractor.extract(plate_crop)
        
        # Optional: Risk scoring (placeholder; integrate ML on VCD patterns)
        risk_score = 0.8 if len(plate_text) > 5 else 0.0  # Example
        
        return {
            "plate_text": plate_text,
            "detection_conf": det_conf,
            "ocr_conf": ocr_conf,
            "risk_score": risk_score
        }
    except Exception as e:
        logger.error(f"Error in ANPR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn src.app:app --host 0.0.0.0 --port 8000