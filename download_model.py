"""
Script to download the DeepSeek-OCR model from HuggingFace Hub.

This script downloads the pre-trained DeepSeek-OCR model and saves it
to a local directory for training and inference.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(
    model_name: str = "unsloth/DeepSeek-OCR",
    local_dir: str = "deepseek_ocr",
    force_download: bool = False
) -> Optional[str]:
    """
    Download DeepSeek-OCR model from HuggingFace Hub.

    Args:
        model_name: Name of the model on HuggingFace Hub
        local_dir: Directory to save the downloaded model
        force_download: Whether to force re-download even if model exists

    Returns:
        Path to downloaded model directory, or None if download failed

    Raises:
        Exception: If download fails
    """
    local_path = Path(local_dir)

    # Check if model already exists
    if local_path.exists() and not force_download:
        logger.info(f"Model already exists at {local_dir}. Skipping download.")
        logger.info("Use force_download=True to re-download.")
        return str(local_path)

    try:
        logger.info(f"Downloading model '{model_name}' to '{local_dir}'...")
        result = snapshot_download(
            model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Model successfully downloaded to {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("Please check your internet connection and HuggingFace credentials.")
        raise


if __name__ == "__main__":
    try:
        download_model()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
