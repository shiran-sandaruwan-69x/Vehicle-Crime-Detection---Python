import os
import shutil
import xml.etree.ElementTree as ET
import yaml
from sklearn.model_selection import train_test_split
import logging

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=config['logging']['level'], format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_voc_to_yolo(xml_path, img_width, img_height):
    """Parse XML and return YOLO format string."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls != 'license-plate':
            continue
        cls_id = 0  # Single class
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Normalize
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    return None

def process_data():
    """Process raw data: Convert XML to YOLO, split dataset."""

    # Use the already-loaded top-level config instead of reopening the file
    # Use 'paths' instead of 'data'
    raw_dir = config['paths']['raw_data']
    processed_dir = config['paths']['processed_data']
    train_dir = config['paths']['train_data']
    test_dir = config['paths']['test_data']
    validation_dir = config['paths']['validation_data']
    # Ensure processed_dir exists before writing labels/images
    os.makedirs(processed_dir, exist_ok=True)

    image_files = []
    for file in os.listdir(raw_dir):
        if file.endswith('.xml'):
            xml_path = os.path.join(raw_dir, file)
            img_name = file.replace('.xml', '.jpg')
            img_path = os.path.join(raw_dir, img_name)
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found for {file}")
                continue
            
            # Get image size from XML (or use PIL if needed)
            tree = ET.parse(xml_path)
            size = tree.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            yolo_label = convert_voc_to_yolo(xml_path, img_width, img_height)
            if yolo_label:
                txt_name = img_name.replace('.jpg', '.txt')
                txt_path = os.path.join(processed_dir, txt_name)
                with open(txt_path, 'w') as f:
                    f.write(yolo_label)
                
                # Copy image
                shutil.copy(img_path, os.path.join(processed_dir, img_name))
                image_files.append(img_name)
            else:
                logger.warning(f"No valid label in {file}")
    
    # Split dataset
    train_ratio = config['dataset']['train_split']
    val_ratio = config['dataset']['val_split']
    test_ratio = config['dataset']['test_split']

    train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=val_ratio/(val_ratio + test_ratio), random_state=42)
    
    def copy_split(files, split_dir):
        os.makedirs(split_dir, exist_ok=True)
        for img in files:
            shutil.copy(os.path.join(processed_dir, img), os.path.join(split_dir, img))
            txt = img.replace('.jpg', '.txt')
            shutil.copy(os.path.join(processed_dir, txt), os.path.join(split_dir, txt))
    
    copy_split(train_files, train_dir)
    copy_split(val_files, validation_dir)
    copy_split(test_files, test_dir)
    
    logger.info("Data processing and splitting complete")

if __name__ == "__main__":
    process_data()