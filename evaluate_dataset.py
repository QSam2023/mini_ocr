"""
Dataset evaluation script for OCR model.

This script evaluates the OCR model on custom datasets with JSON annotations.
Each JSON file contains images, prompts, and ground truth results.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

from transformers import AutoModel, AutoTokenizer
from jiwer import cer, wer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str = "./deepseek_ocr") -> Tuple[Any, Any]:
    """
    Load the DeepSeek-OCR model and tokenizer.

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


def load_annotation_file(json_path: str) -> Dict[str, Any]:
    """
    Load annotation JSON file.

    Args:
        json_path: Path to JSON annotation file

    Returns:
        Parsed JSON data
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {data.get('total_images', 0)} annotations from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load annotation file {json_path}: {e}")
        raise


def run_ocr_inference(
    model: Any,
    tokenizer: Any,
    image_path: str,
    prompt: str,
    output_dir: str = "eval_output"
) -> str:
    """
    Run OCR inference on a single image.

    Args:
        model: OCR model
        tokenizer: Tokenizer
        image_path: Path to image file
        prompt: OCR prompt from annotation
        output_dir: Output directory

    Returns:
        OCR prediction result
    """
    try:
        result = model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False,
        )

        # Extract text from result
        if isinstance(result, list):
            prediction = result[-1] if result else ""
        else:
            prediction = str(result)

        return prediction

    except Exception as e:
        logger.warning(f"Inference failed for {image_path}: {e}")
        return ""


def normalize_text(text: Any) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Text or dict/list to normalize

    Returns:
        Normalized string
    """
    if isinstance(text, dict):
        # Convert dict to JSON string
        return json.dumps(text, ensure_ascii=False, sort_keys=True)
    elif isinstance(text, list):
        # Convert list to string
        return " ".join(str(item) for item in text)
    else:
        return str(text)


def calculate_metrics(ground_truth: str, prediction: str) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        ground_truth: Ground truth text
        prediction: Predicted text

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    try:
        # Character Error Rate
        metrics['cer'] = cer(ground_truth, prediction) if ground_truth and prediction else 1.0

        # Word Error Rate
        metrics['wer'] = wer(ground_truth, prediction) if ground_truth and prediction else 1.0

        # Exact match
        metrics['exact_match'] = 1.0 if ground_truth == prediction else 0.0

        # Length ratio
        gt_len = len(ground_truth)
        pred_len = len(prediction)
        metrics['length_ratio'] = pred_len / gt_len if gt_len > 0 else 0.0

    except Exception as e:
        logger.warning(f"Failed to calculate metrics: {e}")
        metrics = {'cer': 1.0, 'wer': 1.0, 'exact_match': 0.0, 'length_ratio': 0.0}

    return metrics


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    annotation_file: str,
    data_root: str,
    output_dir: str = "evaluation_results",
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.

    Args:
        model: OCR model
        tokenizer: Tokenizer
        annotation_file: Path to annotation JSON file
        data_root: Root directory for image files
        output_dir: Output directory for results
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        Evaluation results dictionary
    """
    # Load annotations
    annotations = load_annotation_file(annotation_file)
    results_list = annotations.get('results', [])

    if max_samples:
        results_list = results_list[:max_samples]

    logger.info(f"Evaluating {len(results_list)} samples...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluation results
    all_metrics = []
    detailed_results = []

    # Process each sample
    for idx, sample in enumerate(tqdm(results_list, desc="Evaluating")):
        image_name = sample.get('image_name', '')
        image_rel_path = sample.get('image_path', '')
        prompt = sample.get('prompt', '<image>\nFree OCR. ')
        ground_truth = sample.get('result', {})

        # Construct full image path
        image_path = Path(data_root) / image_rel_path

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        # Run inference
        prediction = run_ocr_inference(
            model, tokenizer, str(image_path), prompt, output_dir
        )

        # Normalize texts for comparison
        gt_text = normalize_text(ground_truth)
        pred_text = normalize_text(prediction)

        # Calculate metrics
        metrics = calculate_metrics(gt_text, pred_text)
        all_metrics.append(metrics)

        # Store detailed result
        detailed_result = {
            'index': idx,
            'image_name': image_name,
            'image_path': str(image_path),
            'prompt': prompt,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'metrics': metrics
        }
        detailed_results.append(detailed_result)

        # Log progress
        if (idx + 1) % 10 == 0:
            avg_cer = sum(m['cer'] for m in all_metrics) / len(all_metrics)
            logger.info(f"Processed {idx + 1}/{len(results_list)}, Avg CER: {avg_cer:.4f}")

    # Calculate average metrics
    if all_metrics:
        avg_metrics = {
            'avg_cer': sum(m['cer'] for m in all_metrics) / len(all_metrics),
            'avg_wer': sum(m['wer'] for m in all_metrics) / len(all_metrics),
            'avg_exact_match': sum(m['exact_match'] for m in all_metrics) / len(all_metrics),
            'avg_length_ratio': sum(m['length_ratio'] for m in all_metrics) / len(all_metrics),
        }
    else:
        avg_metrics = {'avg_cer': 0, 'avg_wer': 0, 'avg_exact_match': 0, 'avg_length_ratio': 0}

    # Compile final results
    evaluation_results = {
        'dataset': str(annotation_file),
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(results_list),
        'evaluated_samples': len(all_metrics),
        'average_metrics': avg_metrics,
        'detailed_results': detailed_results
    }

    return evaluation_results


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results
        output_path: Output file path
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_summary(results: Dict[str, Any]):
    """
    Print evaluation summary.

    Args:
        results: Evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {results['dataset']}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Evaluated Samples: {results['evaluated_samples']}")
    print(f"Timestamp: {results['timestamp']}")
    print("\n" + "-" * 70)
    print("Average Metrics:")
    print("-" * 70)

    avg_metrics = results['average_metrics']
    print(f"Character Error Rate (CER):  {avg_metrics['avg_cer']:.4f}")
    print(f"Word Error Rate (WER):       {avg_metrics['avg_wer']:.4f}")
    print(f"Exact Match Rate:            {avg_metrics['avg_exact_match']:.4f}")
    print(f"Average Length Ratio:        {avg_metrics['avg_length_ratio']:.4f}")
    print("=" * 70 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate OCR model on custom dataset with JSON annotations"
    )
    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="Path to annotation JSON file (e.g., ocr_data/stamp_data_example/stamp_01/stamp_ocr_01.json)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="ocr_data",
        help="Root directory for image files"
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
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to save detailed results JSON (default: <output>/results_<timestamp>.json)"
    )

    args = parser.parse_args()

    try:
        # Load model
        model, tokenizer = load_model(args.model)

        # Run evaluation
        results = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            annotation_file=args.annotation,
            data_root=args.data_root,
            output_dir=args.output,
            max_samples=args.max_samples
        )

        # Print summary
        print_summary(results)

        # Save detailed results
        if args.save_json:
            output_json = args.save_json
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json = f"{args.output}/results_{timestamp}.json"

        save_results(results, output_json)

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
