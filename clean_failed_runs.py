#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def is_failed_run(run_path):
    """
    Check if a run is considered failed. A run is failed if:
    1. It has no images in either fixed_noise or epoch_images directories, or
    2. It has less than 5 images in either directory (indicating early stop)
    
    Returns True if the run is considered failed.
    """
    # Common image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif'}
    has_fixed_noise = False
    has_epoch_images = False
    fixed_noise_count = 0
    epoch_images_count = 0
    
    # Check fixed_noise directory
    fixed_noise_dir = run_path / 'fixed_noise'
    if fixed_noise_dir.exists():
        for file in fixed_noise_dir.iterdir():
            if any(file.name.lower().endswith(ext) for ext in image_extensions):
                has_fixed_noise = True
                fixed_noise_count += 1
    
    # Check epoch_images directory
    epoch_images_dir = run_path / 'epoch_images'
    if epoch_images_dir.exists():
        for file in epoch_images_dir.iterdir():
            if any(file.name.lower().endswith(ext) for ext in image_extensions):
                has_epoch_images = True
                epoch_images_count += 1
    
    # A run is failed if it has no images in either directory or has too few images
    return not (has_fixed_noise and has_epoch_images) or fixed_noise_count < 5 or epoch_images_count < 5

def clean_failed_runs(models_dir):
    """
    Delete all run directories that:
    1. Don't contain images in both fixed_noise and epoch_images directories, or
    2. Have less than 5 images in either directory (indicating early stop)
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Directory {models_dir} does not exist!")
        return

    deleted_count = 0
    total_runs = 0
    failed_no_images = 0
    failed_early_stop = 0

    # Go through each run directory
    for run_dir in models_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        total_runs += 1
        
        # Check fixed_noise directory
        fixed_noise_dir = run_dir / 'fixed_noise'
        has_fixed_noise = False
        fixed_noise_count = 0
        if fixed_noise_dir.exists():
            for file in fixed_noise_dir.iterdir():
                if file.name.lower().endswith('.png'):
                    has_fixed_noise = True
                    fixed_noise_count += 1
        
        # Check epoch_images directory
        epoch_images_dir = run_dir / 'epoch_images'
        has_epoch_images = False
        epoch_images_count = 0
        if epoch_images_dir.exists():
            for file in epoch_images_dir.iterdir():
                if file.name.lower().endswith('.png'):
                    has_epoch_images = True
                    epoch_images_count += 1
        
        # Delete if either condition is not met
        if not (has_fixed_noise and has_epoch_images) or fixed_noise_count < 5 or epoch_images_count < 5:
            reason = []
            if not has_fixed_noise:
                reason.append("no fixed_noise images")
            if not has_epoch_images:
                reason.append("no epoch_images")
            if fixed_noise_count < 10:
                reason.append(f"only {fixed_noise_count} fixed_noise images")
            if epoch_images_count < 10:
                reason.append(f"only {epoch_images_count} epoch_images")
            
            print(f"Deleting failed run: {run_dir.name} (Reason: {', '.join(reason)})")
            try:
                shutil.rmtree(run_dir)
                deleted_count += 1
                if not (has_fixed_noise and has_epoch_images):
                    failed_no_images += 1
                else:
                    failed_early_stop += 1
            except Exception as e:
                print(f"Error deleting {run_dir}: {e}")

    print(f"\nCleaning complete!")
    print(f"Total runs found: {total_runs}")
    print(f"Failed runs deleted: {deleted_count}")
    print(f"  - No images: {failed_no_images}")
    print(f"  - Early stop: {failed_early_stop}")

if __name__ == "__main__":
    models_dir = "logs/models"
    clean_failed_runs(models_dir) 