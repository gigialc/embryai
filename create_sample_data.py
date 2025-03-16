#!/usr/bin/env python3
"""
Create Sample Embryo Data

This script generates synthetic embryo images for testing the federated learning system.
It creates a balanced dataset with both viable and non-viable embryo images.
"""

import os
import numpy as np
from PIL import Image
import random

# Import utilities
from embryo_fl_utils import DATA_FOLDER

def create_synthetic_dataset(num_images=100):
    """Create a synthetic dataset of embryo images for testing."""
    print(f"Creating synthetic dataset with {num_images} images...")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Image parameters
    image_size = (224, 224)
    
    # Create both viable (label 1) and non-viable (label 0) embryos
    num_viable = num_images // 2
    num_non_viable = num_images - num_viable
    
    # Create viable embryo images (more structured, brighter)
    for i in range(num_viable):
        # Create base image with random noise
        img_array = np.random.randint(100, 200, (image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # Add some structure (a bright circular pattern - simulating a healthy embryo)
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        radius = random.randint(50, 80)
        
        # Draw a bright circular region
        for x in range(image_size[0]):
            for y in range(image_size[1]):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    # Make the central region brighter
                    brightness_factor = 1.5 * (1 - dist/radius)
                    img_array[x, y] = np.minimum(255, img_array[x, y] * brightness_factor).astype(np.uint8)
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_array)
        filename = f"{DATA_FOLDER}/embryo_{i:04d}_label_1.png"
        img.save(filename)
    
    # Create non-viable embryo images (more random, darker)
    for i in range(num_non_viable):
        # Create base image with random noise (darker)
        img_array = np.random.randint(50, 150, (image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # Add some irregular structures (simulating fragmentation)
        num_fragments = random.randint(5, 15)
        for _ in range(num_fragments):
            frag_x = random.randint(0, image_size[0]-30)
            frag_y = random.randint(0, image_size[1]-30)
            frag_size = random.randint(10, 30)
            
            # Make a small region darker
            img_array[frag_x:frag_x+frag_size, frag_y:frag_y+frag_size] = (
                img_array[frag_x:frag_x+frag_size, frag_y:frag_y+frag_size] * 0.7
            ).astype(np.uint8)
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_array)
        filename = f"{DATA_FOLDER}/embryo_{i:04d}_label_0.png"
        img.save(filename)
    
    print(f"Created {num_viable} viable embryo images (label 1)")
    print(f"Created {num_non_viable} non-viable embryo images (label 0)")
    print(f"Total {num_images} images saved to {DATA_FOLDER}/")
    return True

if __name__ == "__main__":
    # Delete existing data if it exists
    if os.path.exists(DATA_FOLDER):
        import shutil
        shutil.rmtree(DATA_FOLDER)
        print(f"Removed existing {DATA_FOLDER} directory")
    
    # Create fresh synthetic data
    create_synthetic_dataset(100) 