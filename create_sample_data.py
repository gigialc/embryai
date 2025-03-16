#!/usr/bin/env python3
"""
Create Sample Embryo Data

This script generates synthetic embryo images for testing the federated learning demo.
The images are random noise, labeled as either survival (1) or non-survival (0).
"""

import os
import random
import numpy as np
from PIL import Image

def create_synthetic_dataset(num_images=100, output_folder="embryo_data"):
    """Create a synthetic dataset of embryo images for testing."""
    print(f"Creating {num_images} synthetic embryo images in {output_folder}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Image dimensions
    image_size = (224, 224)
    
    # Generate images
    for i in range(num_images):
        # Determine label (50% survival, 50% non-survival)
        label = 1 if i < num_images // 2 else 0
        
        # Create random noise image
        # Add some structure to make it slightly more realistic
        img_array = np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # Add some circular structure for embryo-like appearance
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        radius = min(center_x, center_y) - 20
        
        # Create a circular mask
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        
        # Apply mask to brighten center area
        brightness_factor = 1.5 if label == 1 else 1.2  # Surviving embryos are brighter
        img_array[mask] = np.minimum(255, (img_array[mask] * brightness_factor).astype(np.uint8))
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        
        # Save image with appropriate filename
        filename = f"{output_folder}/embryo_{i:04d}_label_{label}.png"
        img.save(filename)
        
        if (i + 1) % 10 == 0:
            print(f"Created {i + 1}/{num_images} images")
    
    print(f"Successfully created {num_images} synthetic embryo images!")
    print(f"  - {num_images // 2} survival images (label 1)")
    print(f"  - {num_images // 2} non-survival images (label 0)")
    print(f"\nYou can now run the federated learning demo with these sample images.")

if __name__ == "__main__":
    # Create 100 synthetic embryo images
    create_synthetic_dataset(num_images=100) 