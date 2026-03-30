"""
Main FastAPI Application
Industry-standard structure with proper separation of concerns
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import get_settings
from app.api.v1 import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 50)
    logger.info("Starting Vehicle License Plate Recognition API")
    logger.info(f"Version: {settings.API_VERSION}")
    logger.info("=" * 50)
    
    # Create temp directory if not exists
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temp directory: {settings.TEMP_DIR}")
    
    logger.info("API startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return JSONResponse(
        content={
            "service": settings.API_TITLE,
            "version": settings.API_VERSION,
            "docs": "/docs",
            "health": f"{settings.API_PREFIX}/health",
            "endpoints": {
                "detect_single": f"{settings.API_PREFIX}/detect",
                "detect_batch": f"{settings.API_PREFIX}/detect-batch",
                "ocr_extract": f"{settings.API_PREFIX}/ocr/extract-text",
                "ocr_batch": f"{settings.API_PREFIX}/ocr/extract-text-batch"
            }
        }
    )


# Include API v1 router
app.include_router(
    api_router,
    prefix=settings.API_PREFIX
)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",  # Changed from "app.main:app"
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )