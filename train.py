"""
Training script for fine-tuning DeepSeek-OCR model using LoRA.

This script fine-tunes the DeepSeek-OCR model on a custom OCR dataset
using parameter-efficient LoRA (Low-Rank Adaptation) method.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoModel, Trainer, TrainingArguments
from unsloth import FastVisionModel, is_bf16_supported

from DeepSeekOCRDataCollator import DeepSeekOCRDataCollator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str = "./deepseek_ocr") -> tuple[Any, Any]:
    """
    Load the DeepSeek-OCR base model and tokenizer.

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
        logger.info(f"Loading base model from {model_path}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )
        logger.info("Base model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def convert_to_conversation(
    sample: Dict[str, Any],
    instruction: str = "<image>\nFree OCR. "
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert a dataset sample to conversation format.

    Args:
        sample: Dataset sample with 'image_path' and 'text' fields
        instruction: OCR instruction prompt

    Returns:
        Formatted conversation dictionary
    """
    return {
        "messages": [
            {
                "role": "<|User|>",
                "content": instruction,
                "images": [sample["image_path"]],
            },
            {
                "role": "<|Assistant|>",
                "content": sample["text"],
            },
        ]
    }


def prepare_dataset(
    dataset_name: str = "hezarai/parsynth-ocr-200k",
    split: str = "train[:1000]",
    instruction: str = "<image>\nFree OCR. "
) -> List[Dict[str, Any]]:
    """
    Load and prepare the training dataset.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split specification
        instruction: OCR instruction prompt

    Returns:
        List of formatted conversation samples

    Raises:
        Exception: If dataset loading fails
    """
    try:
        logger.info(f"Loading dataset '{dataset_name}' split '{split}'...")
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Rename column if needed
        if "image_path" in dataset.column_names and "image" not in dataset.column_names:
            dataset = dataset.rename_column("image_path", "image")

        # Convert to conversation format
        logger.info("Converting dataset to conversation format...")
        converted_dataset = [
            convert_to_conversation(sample, instruction)
            for sample in dataset
        ]
        logger.info(f"Converted {len(converted_dataset)} samples")

        return converted_dataset

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise


def setup_lora_model(
    model: Any,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    random_state: int = 3407
) -> Any:
    """
    Configure model with LoRA for parameter-efficient fine-tuning.

    Args:
        model: Base model to apply LoRA to
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate for LoRA layers
        random_state: Random seed for reproducibility

    Returns:
        Model with LoRA configuration applied

    Raises:
        Exception: If LoRA setup fails
    """
    try:
        logger.info("Setting up LoRA configuration...")
        model = FastVisionModel.get_peft_model(
            model,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=random_state,
        )
        logger.info(f"LoRA configured with rank={lora_rank}, alpha={lora_alpha}")
        return model

    except Exception as e:
        logger.error(f"Failed to setup LoRA: {e}")
        raise


def main() -> None:
    """Main function to run training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek-OCR model with LoRA"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./deepseek_ocr",
        help="Path to base model directory"
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
        default="train[:1000]",
        help="Dataset split to use for training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lora_model",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank parameter"
    )

    args = parser.parse_args()

    try:
        # Load base model
        model, tokenizer = load_model(args.model_path)

        # Prepare dataset
        instruction = "<image>\nFree OCR. "
        train_dataset = prepare_dataset(
            dataset_name=args.dataset,
            split=args.split,
            instruction=instruction
        )

        # Setup LoRA
        model = setup_lora_model(model, lora_rank=args.lora_rank)

        # Prepare model for training
        FastVisionModel.for_training(model)

        # Create data collator
        logger.info("Initializing data collator...")
        data_collator = DeepSeekOCRDataCollator(
            tokenizer=tokenizer,
            model=model,
            image_size=640,
            base_size=1024,
            crop_mode=True,
            train_on_responses_only=True,
        )

        # Create trainer
        logger.info("Setting up trainer...")
        training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            args=training_args,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")

        # Save model
        logger.info(f"Saving model to {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
