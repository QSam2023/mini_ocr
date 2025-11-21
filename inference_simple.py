"""
Simple inference script without Unsloth (fallback version).

This script uses pure transformers for inference when Unsloth has compatibility issues.
"""

import argparse
import logging
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str = "./deepseek_ocr"):
    """
    Load the DeepSeek-OCR model and tokenizer using transformers.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please run 'python download_model.py' first."
        )

    try:
        logger.info(f"Loading model from {model_path}...")

        # Load with transformers directly
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )

        logger.info("Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_inference(
    model,
    tokenizer,
    image_path: str,
    prompt: str = "<image>\nFree OCR. "
):
    """
    Run OCR inference on an image.

    Args:
        model: Loaded OCR model
        tokenizer: Model tokenizer
        image_path: Path to input image
        prompt: OCR prompt template

    Returns:
        Extracted text from the image
    """
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        logger.info(f"Running inference on {image_path}...")

        # Load image
        image = Image.open(image_path)

        # Check if model has generate method (for causal LM models)
        if hasattr(model, 'generate'):
            # Prepare inputs
            inputs = tokenizer(
                prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            # Generate
            outputs = model.generate(**inputs, max_new_tokens=512)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Otherwise try the infer method if available
        elif hasattr(model, 'infer'):
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path="output",
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
            )
        else:
            raise AttributeError("Model doesn't have 'generate' or 'infer' method")

        logger.info("Inference completed successfully")
        return result

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main():
    """Main function to run OCR inference from command line."""
    parser = argparse.ArgumentParser(
        description="Run OCR inference using pure transformers (no Unsloth)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./deepseek_ocr",
        help="Path to model directory"
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
