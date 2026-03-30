import cv2
import os
from pathlib import Path
import easyocr
import json

class PlateOCR:
    def __init__(self, lang='en'):
        """
        Initialize EasyOCR for plate text extraction.
        
        Args:
            lang (str): Language for OCR (e.g., 'en' for English)
        """
        # EasyOCR uses language codes like ['en'] as a list
        lang_list = [lang] if isinstance(lang, str) else lang
        self.ocr = easyocr.Reader(lang_list, gpu=False)
        print(f"EasyOCR initialized (Language: {lang_list})")
    
    def extract_text(self, image_path, debug=False):
        """
        Extract text from a single cropped plate image.
        
        Args:
            image_path (str): Path to cropped plate image
            debug (bool): Print debug information
            
        Returns:
            dict: Extracted text with confidence score
        """
        try:
            # Read image using cv2 to ensure it's valid
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'image': image_path,
                    'text': '',
                    'confidence': 0.0,
                    'success': False,
                    'error': 'Could not read image'
                }
            
            # Convert to grayscale for better OCR results
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Run OCR - EasyOCR returns list of (bbox, text, confidence)
            result = self.ocr.readtext(gray)
            
            # Debug: Print the raw result structure
            if debug:
                print(f"\n=== DEBUG: Raw result structure ===")
                print(f"Result type: {type(result)}")
                print(f"Result length: {len(result) if result else 0}")
                if result and len(result) > 0:
                    print(f"First item type: {type(result[0])}")
                    print(f"First item: {result[0]}")
                print(f"===================================\n")
            
            if not result or result is None or len(result) == 0:
                return {
                    'image': image_path,
                    'text': '',
                    'confidence': 0.0,
                    'success': False,
                    'error': 'No result from OCR'
                }
            
            # Extract text and confidence
            plate_text = ''
            confidence = 0.8
            
            # EasyOCR returns [(bbox, text, conf), ...]
            for detection in result:
                bbox, text, conf = detection
                plate_text += str(text)
                confidence = max(confidence, float(conf))
                
                if debug:
                    print(f"Extracted: '{text}' (conf: {conf})")
            
            if plate_text.strip():
                return {
                    'image': image_path,
                    'text': plate_text.strip(),
                    'confidence': round(confidence, 4),
                    'success': True
                }
            else:
                return {
                    'image': image_path,
                    'text': '',
                    'confidence': 0.0,
                    'success': False,
                    'error': 'No text detected in result'
                }
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'image': image_path,
                'text': '',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def extract_text_batch(self, input_dir, output_dir=None, extensions=['jpg', 'jpeg', 'png']):
        """
        Extract text from all cropped plate images in a directory.
        
        Args:
            input_dir (str): Directory containing cropped plate images
            output_dir (str): Directory to save results (optional)
            extensions (list): Image file extensions to process
            
        Returns:
            dict: Summary with all extracted texts
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'extractions': []
        }
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_dir).glob(f'*.{ext}'))
            image_files.extend(Path(input_dir).glob(f'*.{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
        print(f"Found {len(image_files)} cropped plate images to process...")
        
        # Enable debug for first few images
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
            
            debug = idx <= 3  # Debug first 3 images
            extraction = self.extract_text(str(image_path), debug=debug)
            results['extractions'].append(extraction)
            results['total_images'] += 1
            
            if extraction['success']:
                results['successful'] += 1
                print(f"  ✓ Text: {extraction['text']} (Confidence: {extraction['confidence']})")
            else:
                results['failed'] += 1
                print(f"  ✗ Failed - {extraction.get('error', 'Unknown error')}")
        
        # Save results to JSON if output directory specified
        if output_dir:
            results_file = os.path.join(output_dir, 'ocr_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
        
        return results
    
    def extract_and_save_txt(self, input_dir, output_dir, extensions=['jpg', 'jpeg', 'png']):
        """
        Extract text and save individual .txt files for each plate.
        
        Args:
            input_dir (str): Directory containing cropped plate images
            output_dir (str): Directory to save extracted text files
            extensions (list): Image file extensions to process
            
        Returns:
            dict: Summary of extraction
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'files': []
        }
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_dir).glob(f'*.{ext}'))
            image_files.extend(Path(input_dir).glob(f'*.{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))
        print(f"Found {len(image_files)} cropped plate images to process...")
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
            
            debug = idx <= 3  # Debug first 3 images
            extraction = self.extract_text(str(image_path), debug=debug)
            results['total_images'] += 1
            
            if extraction['success']:
                results['successful'] += 1
                
                # Save text to file
                txt_filename = image_path.stem + '.txt'
                txt_path = os.path.join(output_dir, txt_filename)
                
                with open(txt_path, 'w') as f:
                    f.write(f"{extraction['text']}\n")
                    f.write(f"Confidence: {extraction['confidence']}")
                
                results['files'].append({
                    'image': image_path.name,
                    'text_file': txt_filename,
                    'text': extraction['text'],
                    'confidence': extraction['confidence']
                })
                print(f"  ✓ Saved: {txt_filename}")
            else:
                results['failed'] += 1
                print(f"  ✗ Failed - {extraction.get('error', 'Unknown error')}")
        
        # Save summary
        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Paths
    cropped_images_dir = "/home/kasr/Storage/Shiran_Works/OCR_Vehicle_Plate_Detector/data/cropped_images"
    ocr_output_dir = "/home/kasr/Storage/Shiran_Works/OCR_Vehicle_Plate_Detector/data/ocr_results"
    
    # Initialize OCR
    ocr = PlateOCR(lang='en')
    
    # Option 1: Extract text and save JSON results
    # results = ocr.extract_text_batch(cropped_images_dir, ocr_output_dir)
    
    # Option 2: Extract text and save individual .txt files
    results = ocr.extract_and_save_txt(cropped_images_dir, ocr_output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("OCR Processing Summary:")
    print(f"Total images processed: {results['total_images']}")
    print(f"Successfully extracted: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {(results['successful']/results['total_images']*100):.2f}%" if results['total_images'] > 0 else "N/A")