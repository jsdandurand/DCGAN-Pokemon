import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def convert_background_to_black(image_path, output_path, threshold=240):
    """
    Convert white/light backgrounds to black.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        threshold: RGB value threshold to identify background (default 240)
    """
    # Open image and convert to RGBA
    img = Image.open(image_path).convert('RGBA')
    data = np.array(img)
    
    # Create alpha mask for non-background pixels
    # Background is identified as pixels with high RGB values
    rgb = data[:, :, :3]
    is_light = np.all(rgb > threshold, axis=2)
    
    # Create output array with black background
    output_data = data.copy()
    output_data[is_light] = [0, 0, 0, 255]  # Set background pixels to black
    
    # Convert back to PIL Image and save
    output_img = Image.fromarray(output_data)
    
    # Convert to RGB (remove alpha channel) and save
    output_img = output_img.convert('RGB')
    output_img.save(output_path)

def process_dataset(input_dir, output_dir, threshold=240):
    """
    Process all images in the dataset.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save processed images
        threshold: RGB value threshold to identify background
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    input_dir = Path(input_dir)
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        output_path = output_dir / img_path.name
        try:
            convert_background_to_black(img_path, output_path, threshold)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert dataset images to have black backgrounds")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed images")
    parser.add_argument("--threshold", type=int, default=240, help="RGB threshold for background detection (0-255)")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.threshold)
    print("Processing complete!") 