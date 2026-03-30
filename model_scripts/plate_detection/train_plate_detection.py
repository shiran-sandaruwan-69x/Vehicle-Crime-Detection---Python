import sys
import os
from pathlib import Path
import yaml
import torch

# Set CUDA memory allocation strategy BEFORE any torch operations
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.utils import load_config, logger
from ultralytics import YOLO

config = load_config()

def print_gpu_memory():
    """Print current GPU memory stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def create_data_yaml():
    """Generate data.yaml for YOLO."""
    data_yaml = {
        'train': config['paths']['train_data'],
        'val': config['paths']['validation_data'],
        'test': config['paths']['test_data'],
        'nc': 1,
        'names': ['license-plate']
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    return 'data.yaml'

if __name__ == "__main__":
    print_gpu_memory()
    
    data_yaml = create_data_yaml()
    model = YOLO(f"{config['model']['detection']['name']}m.pt")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    print_gpu_memory()
    
    try:
        model.train(
            data=data_yaml,
            epochs=config['training']['epochs'],
            imgsz=config['model']['detection']['img_size'][0],
            batch=1,  # Reduced from config - critical for 4GB GPU
            lr0=config['training']['learning_rate'],
            optimizer=config['training']['optimizer'],
            device=config['model']['detection']['device'],
            patience=config['training']['early_stopping']['patience'],
            save_period=config['training']['checkpoint']['save_frequency'],
            project=config['paths']['models_checkpoints'],
            name='plate_detection'
        )
        
        # Export best model
        best_model_path = Path(config['paths']['plate_detection']) / 'best.pt'
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.export(format='pt')
        
        logger.info("Training complete")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory. Try reducing batch size further or image size.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise