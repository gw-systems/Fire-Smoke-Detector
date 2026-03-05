import os
import urllib.request
from pathlib import Path

# Important: Update this base URL to the actual Release URL once you create it!
# For example, if your release tag is v1.0:
# BASE_URL = "https://github.com/gw-systems/Fire-Smoke-Detector/releases/download/v1.0/"
BASE_URL = "https://github.com/gw-systems/Fire-Smoke-Detector/releases/download/v1.0/"

MODELS = {
    "3.pt": BASE_URL + "3.pt",
    "5.pt": BASE_URL + "5.pt",
    "6.pt": BASE_URL + "6.pt"
}

def download_models(download_dir="Models"):
    """
    Downloads the large YOLOv8 .pt models from the GitHub Release
    into the specified directory.
    """
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    for model_name, url in MODELS.items():
        dst_path = os.path.join(download_dir, model_name)
        
        # Skip if the file already exists and has a size > 0
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            print(f"[{model_name}] already exists at {dst_path}, skipping download.")
            continue
            
        print(f"Downloading {model_name} from {url} ...")
        
        try:
            # Download the file
            urllib.request.urlretrieve(url, dst_path)
            print(f"Successfully downloaded {model_name} to {dst_path}")
        except Exception as e:
            print(f"Failed to download {model_name}. Error: {e}")
            # Clean up the file if it's partially downloaded
            if os.path.exists(dst_path):
                os.remove(dst_path)
                
if __name__ == "__main__":
    download_models()
