#!/usr/bin/env python3
"""
Setup script for EmbryoML Federated Learning Demo

This script installs all the required dependencies for the federated learning demo.
Run this script before attempting to run the federated_embryo_demo.py script.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages for the federated learning demo."""
    print("Installing required packages...")
    
    # List of required packages
    requirements = [
        "torch",
        "torchvision",
        "flwr",  # Flower federated learning framework
        "pillow",
        "numpy"
    ]
    
    # Install each package using pip
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")
    print("You can now run the federated learning demo with:")
    print("  - Server: python federated_embryo_demo.py server --port=8080")
    print("  - Client: python federated_embryo_demo.py client --server_address=<server_ip>:8080 --client_id=<id>")

def create_data_directory():
    """Create the data directory if it doesn't exist."""
    if not os.path.exists("embryo_data"):
        os.makedirs("embryo_data", exist_ok=True)
        print("\nCreated embryo_data directory.")
        print("Please add your embryo images to this directory before running the demo.")
        print("Images should follow the naming convention *label_0*.png or *label_1*.png")
        print("where 1 indicates survival and 0 indicates non-survival.")

if __name__ == "__main__":
    print("=== EmbryoML Federated Learning Setup ===")
    install_requirements()
    create_data_directory()
    print("\nSetup complete! You're ready to run the federated learning demo.") 