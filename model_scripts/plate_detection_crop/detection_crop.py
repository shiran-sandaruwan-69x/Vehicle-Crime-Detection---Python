import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class PlateDetectionCropper:
    def __init__(self, model_path):
        """
        Initialize the detector with a trained YOLO model.
        
        Args:
            model_path (str): Path to the trained model weights
        """
        self.model = YOLO(model_path)
        self.model.to('cpu')  # Use GPU if available
    
    def crop_plates(self, image_path, output_dir, confidence=0.5):
        """
        Detect and crop plates from an image.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save cropped plates
            confidence (float): Confidence threshold for detection
            
        Returns:
            list: Paths to saved cropped images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return []
        
        # Run detection
        results = self.model.predict(image_path, conf=confidence)
        
        cropped_paths = []
        
        # Process detections
        for idx, result in enumerate(results):
            boxes = result.boxes
            
            for box_idx, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Add padding if desired
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Crop the plate
                cropped_plate = image[y1:y2, x1:x2]
                
                # Save cropped plate
                image_name = Path(image_path).stem
                output_path = os.path.join(
                    output_dir, 
                    f"{image_name}_plate_{box_idx}.jpg"
                )
                cv2.imwrite(output_path, cropped_plate)
                cropped_paths.append(output_path)
                print(f"Saved: {output_path}")
        
        return cropped_paths
    
    def crop_plates_batch(self, input_dir, output_dir, confidence=0.5, extensions=['jpg', 'jpeg', 'png']):
        """
        Detect and crop plates from all images in a directory.
        
        Args:
            input_dir (str): Directory containing images
            output_dir (str): Directory to save cropped plates
            confidence (float): Confidence threshold for detection
            extensions (list): Image file extensions to process
            
        Returns:
            dict: Summary of processed images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        summary = {
            'total_images': 0,
            'total_plates': 0,
            'failed': []
        }
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_dir).glob(f'*.{ext}'))
            image_files.extend(Path(input_dir).glob(f'*.{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to process...")
        
        for image_path in image_files:
            summary['total_images'] += 1
            try:
                cropped = self.crop_plates(str(image_path), output_dir, confidence)
                summary['total_plates'] += len(cropped)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                summary['failed'].append(str(image_path))
        
        return summary


# Example usage
if __name__ == "__main__":
    # Paths
    model_path = "/home/kasr/Storage/Shiran_Works/OCR_Vehicle_Plate_Detector/models/plate_detection/best.pt"
    input_dir = "/home/kasr/Storage/Shiran_Works/OCR_Vehicle_Plate_Detector/data/raw"  # Your input data
    output_dir = "/home/kasr/Storage/Shiran_Works/OCR_Vehicle_Plate_Detector/data/cropped_images"
    
    # Initialize cropper
    cropper = PlateDetectionCropper(model_path)
    
    # Process single image
    # cropper.crop_plates("path/to/image.jpg", output_dir)
    
    # Process batch
    summary = cropper.crop_plates_batch(input_dir, output_dir, confidence=0.5)
    
    print("\n" + "="*50)
    print("Processing Summary:")
    print(f"Total images processed: {summary['total_images']}")
    print(f"Total plates detected: {summary['total_plates']}")
    print(f"Failed: {len(summary['failed'])}")
    if summary['failed']:
        print(f"Failed files: {summary['failed']}")