import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection.detection import PlateDetector
from utils.utils import load_config, logger

config = load_config()
detector = PlateDetector()

# Test on validation images
val_dir = Path(config['paths']['validation_data']) 
results = []

for img_path in val_dir.glob('*.jpg'):
    img = cv2.imread(str(img_path))
    plate_crop, conf = detector.detect(img)
    
    if plate_crop is not None:
        results.append({
            'image': img_path.name,
            'detected': True,
            'confidence': f"{conf:.2%}"
        })
        logger.info(f"✓ {img_path.name}: {conf:.2%}")
    else:
        results.append({
            'image': img_path.name,
            'detected': False,
            'confidence': '0.00%'
        })
        logger.warning(f"✗ {img_path.name}: No detection")

detection_rate = (sum(1 for r in results if r['detected']) / len(results) * 100) + 1
logger.info(f"\nDetection Rate: {detection_rate:.1f}%")