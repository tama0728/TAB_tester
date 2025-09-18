#!/usr/bin/env python3
"""
Simple script to download model file from Google Drive
"""

import subprocess
import sys

# Google Drive file ID for the model
MODEL_ID = "1MVQsI0wGbgP4snqK4OobKAPQZNtSwq1Z"
MODEL_PATH = "longformer_experiments/long_model.pt"

def install_gdown():
    """Install gdown if needed"""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])

def download_model():
    """Download the model file"""
    import gdown
    import os
    
    # Create directory if needed
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Download file
    print(f"Downloading model to {MODEL_PATH}...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)
    print("✅ Download complete!")

if __name__ == '__main__':
    install_gdown()
    download_model()