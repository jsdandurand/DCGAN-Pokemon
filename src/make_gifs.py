import os
import imageio
import argparse
from pathlib import Path
import re

def get_epoch_number(filename):
    """Extract epoch number from filename like 'epoch_123.png'"""
    match = re.search(r'epoch_(\d+)\.png', filename.name)
    return int(match.group(1)) if match else 0

def create_gifs(run_name):
    """
    Create GIFs from saved training images for a given run.
    
    Args:
        run_name: Name of the run directory containing the saved images
    """
    run_path = Path(run_name)
    if not run_path.exists():
        raise ValueError(f"Run directory {run_name} does not exist")

    # Create GIF from epoch images
    epoch_images_dir = run_path / "epoch_images"
    if epoch_images_dir.exists():
        epoch_images = []
        # Sort files by epoch number
        image_files = sorted(epoch_images_dir.glob('epoch_*.png'), key=get_epoch_number)
        for img_path in image_files:
            try:
                epoch_images.append(imageio.imread(img_path))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        if epoch_images:
            imageio.mimsave(run_path / 'training_progress.gif', epoch_images, fps=5)
            print(f"Created training progress GIF with {len(epoch_images)} frames")
        else:
            print("No epoch images found")
    else:
        print("Epoch images directory not found")

    # Create GIF from fixed noise images
    fixed_noise_dir = run_path / "fixed_noise"
    if fixed_noise_dir.exists():
        fixed_images = []
        # Sort files by epoch number
        image_files = sorted(fixed_noise_dir.glob('epoch_*.png'), key=get_epoch_number)
        for img_path in image_files:
            try:
                fixed_images.append(imageio.imread(img_path))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        if fixed_images:
            imageio.mimsave(run_path / 'fixed_noise_progress.gif', fixed_images, fps=5)
            print(f"Created fixed noise progress GIF with {len(fixed_images)} frames")
        else:
            print("No fixed noise images found")
    else:
        print("Fixed noise directory not found")

def main():
    parser = argparse.ArgumentParser(description='Create GIFs from saved training images')
    parser.add_argument('run_name', type=str, help='Name of the run directory containing the saved images')
    args = parser.parse_args()
    
    try:
        create_gifs(args.run_name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 