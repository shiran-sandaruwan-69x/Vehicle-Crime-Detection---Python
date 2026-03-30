# Configuration Directory

This directory contains configuration files for models, training, and application settings.

## Configuration Files

- `config.yaml`: Main application configuration
- `model_config.yaml`: Model architecture and hyperparameters
- `training_config.yaml`: Training parameters (learning rate, batch size, epochs)
- `data_config.yaml`: Data paths and preprocessing settings

## Example Structure

```yaml
# config.yaml
model:
  detection_model: "yolov5"
  ocr_model: "tesseract"

preprocessing:
  resize: [640, 640]
  normalize: true

paths:
  data_dir: "data/"
  models_dir: "models/"
  output_dir: "output/"
```
