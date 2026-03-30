#!/usr/bin/env python3
"""
Inference script for license plate detection and OCR.

Usage:
    python predict.py --image path/to/image.jpg --output output/
    python predict.py --image path/to/image.jpg --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run license plate detection and OCR on an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with bounding boxes",
    )
    return parser.parse_args()


def main():
    """Main function for running inference."""
    args = parse_args()
    
    print(f"Input image: {args.image}")
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output}")
    
    # TODO: Implement actual inference pipeline
    # from ocr_plate_detector import PlateDetector
    # detector = PlateDetector(config_path=args.config)
    # result = detector.process_image(args.image)
    # detector.save_results(result, args.output, visualize=args.visualize)
    
    print("\nInference pipeline not yet implemented.")
    print("This is a template script for the project structure.")


if __name__ == "__main__":
    main()
