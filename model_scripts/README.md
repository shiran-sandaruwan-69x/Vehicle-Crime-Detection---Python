# Scripts Directory

This directory contains utility scripts for training, testing, and data processing.

## Common Scripts

- `train.py`: Script to train detection and OCR models
- `evaluate.py`: Script to evaluate model performance
- `predict.py`: Script to run inference on new images
- `preprocess_data.py`: Script to preprocess raw data
- `download_dataset.py`: Script to download public datasets

## Usage Examples

```bash
# Train a model
python scripts/train.py --config config/training_config.yaml

# Evaluate model
python scripts/evaluate.py --model models/plate_detection/best.pt --data data/test/

# Run prediction
python scripts/predict.py --image path/to/image.jpg --output output/
```
