# OCR Vehicle Plate Detector

An intelligent system for detecting vehicle license plates and extracting text using computer vision and OCR technologies.

## 🚀 Features

- **License Plate Detection**: Automatically detect license plates in images using state-of-the-art object detection models
- **OCR Text Extraction**: Extract text from detected license plates with high accuracy
- **Multiple Model Support**: Support for YOLO, Faster R-CNN, Tesseract, EasyOCR, and more
- **Easy Configuration**: YAML-based configuration for models, training, and inference
- **Comprehensive Pipeline**: End-to-end pipeline from image preprocessing to text extraction

## 📁 Project Structure

```
OCR_Vehicle_Plate_Detector/
├── src/
│   └── ocr_plate_detector/          # Main source code package
│       ├── __init__.py
│       ├── preprocessing/           # Image preprocessing modules
│       ├── detection/               # License plate detection models
│       ├── recognition/             # OCR and text recognition
│       └── utils/                   # Utility functions
│
├── data/                            # Dataset directory
│   ├── raw/                         # Original unprocessed images
│   ├── processed/                   # Preprocessed images
│   ├── train/                       # Training dataset
│   ├── test/                        # Testing dataset
│   └── validation/                  # Validation dataset
│
├── models/                          # Saved models and weights
│   ├── plate_detection/             # Detection model weights
│   ├── ocr/                         # OCR model weights
│   └── checkpoints/                 # Training checkpoints
│
├── notebooks/                       # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_plate_detection.ipynb
│   ├── 04_ocr_recognition.ipynb
│   └── 05_full_pipeline.ipynb
│
├── config/                          # Configuration files
│   └── config.yaml                  # Main configuration file
│
├── scripts/                         # Utility scripts
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   ├── predict.py                   # Inference script
│   └── preprocess_data.py           # Data preprocessing script
│
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
│
├── docs/                            # Documentation
│   └── README.md
│
├── output/                          # Output files
│   ├── images/                      # Processed images with detections
│   └── reports/                     # Performance reports
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation script
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kasrsu/OCR_Vehicle_Plate_Detector.git
cd OCR_Vehicle_Plate_Detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## 📖 Usage

### Quick Start

```python
from ocr_plate_detector import PlateDetector

# Initialize detector
detector = PlateDetector(config_path="config/config.yaml")

# Detect and recognize license plate
result = detector.process_image("path/to/vehicle/image.jpg")
print(f"Detected plate: {result['text']}")
```

### Training a Model

```bash
python scripts/train.py --config config/config.yaml
```

### Running Inference

```bash
python scripts/predict.py --image path/to/image.jpg --output output/
```

### Evaluating Model Performance

```bash
python scripts/evaluate.py --model models/plate_detection/best.pt --data data/test/
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model selection (YOLO, Faster R-CNN, etc.)
- Training parameters (learning rate, batch size, epochs)
- Preprocessing settings
- OCR engine selection (Tesseract, EasyOCR)
- Input/output paths

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/ocr_plate_detector tests/

# Run specific test module
pytest tests/unit/test_preprocessing.py
```

## 📊 Supported Models

### Detection Models
- YOLOv8 (Recommended)
- YOLOv5
- Faster R-CNN

### OCR Engines
- EasyOCR (Recommended)
- Tesseract OCR
- Custom CRNN models
