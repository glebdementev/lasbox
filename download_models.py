"""Module for downloading model files from remote server."""
import os
import requests
from pathlib import Path
from typing import Optional, Callable


# Server path for models
MODEL_SERVER_PATH = "https://github.com/NRCan/TreeAIBox/releases/download/v1.0/"


def download_model(model_name: str, local_path: Path, progress_callback: Optional[Callable[[int], None]] = None) -> bool:
    """
    Download a model file from the server to local storage.
    
    Args:
        model_name: Name of the model (without .pth extension)
        local_path: Path where the model should be saved
        progress_callback: Optional callback function(percent: int) -> None for progress updates
    
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Due to GitHub rule, all brackets in released model names were reformatted
        url = f"{MODEL_SERVER_PATH}{model_name}.pth".replace("(", "_").replace(")", "")
        
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file first, in case download is interrupted
        temp_path = local_path.with_suffix(local_path.suffix + ".temp")
        
        print(f"Downloading {model_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        last_printed_percent = -1
        
        with open(temp_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    # Update progress
                    if total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        if progress_callback:
                            progress_callback(percent)
                        elif percent >= last_printed_percent + 25:  # Print every 25%
                            print(f"  Progress: {percent}%")
                            last_printed_percent = percent
        
        # Rename the temp file to the target file once download is complete
        if local_path.exists():
            local_path.unlink()
        temp_path.rename(local_path)
        
        print(f"Successfully downloaded {model_name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download model {model_name}: {str(e)}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        return False
    except Exception as e:
        print(f"Unexpected error downloading model {model_name}: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False


def ensure_model_exists(model_name: str, local_path: Path, progress_callback: Optional[Callable[[int], None]] = None) -> bool:
    """
    Ensure a model file exists locally, downloading it if necessary.
    
    Args:
        model_name: Name of the model (without .pth extension)
        local_path: Path where the model should be located
        progress_callback: Optional callback function(percent: int) -> None for progress updates
    
    Returns:
        True if model exists (or was successfully downloaded), False otherwise
    """
    if local_path.exists():
        print(f"Model {model_name} already exists at {local_path}")
        return True
    
    print(f"Model {model_name} not found at {local_path}")
    return download_model(model_name, local_path, progress_callback)

