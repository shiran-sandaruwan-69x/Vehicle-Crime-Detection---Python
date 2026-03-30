import logging
import yaml
import cv2
import numpy as np
from pathlib import Path

# Load config with absolute path
config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

logging.basicConfig(level=CONFIG['logging']['level'], format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess: Grayscale, CLAHE for contrast (handles low-light)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    logger.info("Image preprocessed")
    return enhanced

def load_config():
    return CONFIG