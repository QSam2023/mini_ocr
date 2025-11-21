"""
Inference script for DeepSeek-OCR model.

This script loads a trained model and performs OCR inference on a single image.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from unsloth import FastVisionModel
from transformers import AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str = "./deepseek_ocr") -> tuple[Any, Any]:
    """
    Load the DeepSeek-OCR model and tokenizer.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        FileNotFoundError: If model directory doesn't exist
        Exception: If model loading fails
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please run 'python download_model.py' first."
        )

    try:
        logger.info(f"Loading model from {model_path}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )
        logger.info("Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_inference(
    model: Any,
    tokenizer: Any,
    image_path: str,
    prompt: str = "<image>\nFree OCR. ",
    output_path: str = "output",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    save_results: bool = True,
) -> Optional[str]:
    """
    Run OCR inference on an image.

    Args:
        model: Loaded OCR model
        tokenizer: Model tokenizer
        image_path: Path to input image
        prompt: OCR prompt template
        output_path: Directory to save results
        base_size: Base size for image processing
        image_size: Target image size
        crop_mode: Whether to use dynamic cropping
        save_results: Whether to save intermediate results

    Returns:
        Extracted text from the image, or None if inference failed

    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If inference fails
    """
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        logger.info(f"Running inference on {image_path}...")
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            test_compress=False,
        )

        logger.info("Inference completed successfully")
        return result

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main() -> None:
    """Main function to run OCR inference from command line."""
    parser = argparse.ArgumentParser(
        description="Run OCR inference on an image using DeepSeek-OCR"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="your_image.jpg",
        help="Path to input image file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./deepseek_ocr",
        help="Path to model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\nFree OCR. ",
        help="OCR prompt template"
    )

    args = parser.parse_args()

    try:
        # Load model
        model, tokenizer = load_model(args.model)

        # Run inference
        result = run_inference(
            model,
            tokenizer,
            image_path=args.image,
            prompt=args.prompt,
            output_path=args.output,
        )

        # Print result
        print("\n" + "=" * 50)
        print("OCR Result:")
        print("=" * 50)
        print(result)
        print("=" * 50)

    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
