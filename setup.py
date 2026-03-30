"""Setup script for OCR Vehicle Plate Detector package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ocr-plate-detector",
    version="0.1.0",
    author="kasrsu",
    description="OCR system for vehicle license plate detection and text extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kasrsu/OCR_Vehicle_Plate_Detector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pytesseract>=0.3.8",
        "easyocr>=1.4.0",
        "ultralytics>=8.0.0",
        "scikit-image>=0.18.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "tensorflow": [
            "tensorflow>=2.6.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ocr, license-plate, vehicle-detection, computer-vision, deep-learning",
)
