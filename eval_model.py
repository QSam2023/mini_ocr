"""
Model evaluation script using Character Error Rate (CER) metric.

This script evaluates the OCR model performance on a test dataset
using the CER (Character Error Rate) metric.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from jiwer import cer
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


def evaluate_sample(
    model: Any,
    tokenizer: Any,
    sample: Dict[str, Any],
    sample_idx: int,
    output_dir: str = "eval_output"
) -> Dict[str, Any]:
    """
    Evaluate model on a single sample.

    Args:
        model: Loaded OCR model
        tokenizer: Model tokenizer
        sample: Dataset sample containing image and text
        sample_idx: Index of the sample (for logging)
        output_dir: Directory to save evaluation outputs

    Returns:
        Dictionary containing prediction, ground truth, and CER score

    Raises:
        Exception: If evaluation fails
    """
    try:
        # Save image temporarily for inference
        temp_image_path = f"eval_{sample_idx}.jpg"
        sample["image_path"].save(temp_image_path)
        logger.info(f"Evaluating sample {sample_idx}...")

        # Get ground truth
        ground_truth = sample["text"]

        # Run inference
        result = model.infer(
            tokenizer,
            prompt="<image>\nFree OCR. ",
            image_file=temp_image_path,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
        )

        # Extract prediction
        prediction = result[-1] if isinstance(result, list) else result

        # Calculate CER
        cer_score = cer(ground_truth, prediction)

        # Clean up temporary file
        Path(temp_image_path).unlink(missing_ok=True)

        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "cer": cer_score,
            "sample_idx": sample_idx
        }

    except Exception as e:
        logger.error(f"Evaluation failed for sample {sample_idx}: {e}")
        raise


def main() -> None:
    """Main function to run model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-OCR model using CER metric"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./deepseek_ocr",
        help="Path to model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hezarai/parsynth-ocr-200k",
        help="Dataset name on HuggingFace"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:2000]",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=1523,
        help="Index of sample to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_output",
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    try:
        # Load model
        model, tokenizer = load_model(args.model)

        # Load dataset
        logger.info(f"Loading dataset '{args.dataset}' split '{args.split}'...")
        dataset = load_dataset(args.dataset, split=args.split)
        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Validate sample index
        if args.sample_idx >= len(dataset):
            raise ValueError(
                f"Sample index {args.sample_idx} out of range. "
                f"Dataset has {len(dataset)} samples."
            )

        # Evaluate sample
        sample = dataset[args.sample_idx]
        results = evaluate_sample(
            model,
            tokenizer,
            sample,
            args.sample_idx,
            args.output
        )

        # Print results
        print("\n" + "=" * 60)
        print(f"Evaluation Results (Sample {results['sample_idx']})")
        print("=" * 60)
        print(f"\nGround Truth:\n{results['ground_truth']}")
        print(f"\nPrediction:\n{results['prediction']}")
        print(f"\nCharacter Error Rate (CER): {results['cer']:.4f}")
        print("=" * 60)

        # Log results
        logger.info(f"Evaluation completed. CER: {results['cer']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
