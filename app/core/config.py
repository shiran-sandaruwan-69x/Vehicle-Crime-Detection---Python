"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # API Metadata
    API_TITLE: str = "Vehicle License Plate Recognition API"
    API_DESCRIPTION: str = "Upload vehicle images to detect and extract license plate text"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    RELOAD: bool = True
    
    # Models
    PLATE_DETECTION_MODEL_PATH: str = "D:\ICBT Campus\my final project crime\python model\OCR_Vehicle_Plate_Detector\models\plate_detection\best.pt"
    OCR_LANGUAGE: str = "en"
    DETECTION_CONFIDENCE: float = 0.5
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()