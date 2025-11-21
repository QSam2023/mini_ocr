# Mini OCR

A lightweight OCR (Optical Character Recognition) project based on DeepSeek-OCR and Unsloth for efficient fine-tuning and inference.

## Features

- Fine-tune DeepSeek-OCR model with LoRA for parameter-efficient training
- Support for dynamic image preprocessing and cropping
- Fast inference with optimized model loading
- Character Error Rate (CER) evaluation metrics
- Training on custom OCR datasets

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for training

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd mini_ocr
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the base model

```bash
python download_model.py
```

This will download the DeepSeek-OCR model to `./deepseek_ocr/` directory.

## Usage

### Training

Train the model on the Parsynth OCR dataset:

```bash
python train.py
```

**Training parameters:**
- Dataset: hezarai/parsynth-ocr-200k (first 1000 samples)
- Batch size: 2
- Gradient accumulation: 4
- Learning rate: 2e-4
- Max steps: 60
- LoRA rank: 16

The trained LoRA weights will be saved to `./lora_model/`.

### Inference

Run inference on a single image:

```bash
python inference.py --image path/to/your/image.jpg
```

Optional parameters:
```bash
python inference.py --image photo.jpg --model ./deepseek_ocr --output results
```

### Evaluation

#### Single Sample Evaluation

Evaluate the model on a single sample using Character Error Rate (CER):

```bash
python eval_model.py --sample-idx 100
```

#### Dataset Evaluation

Evaluate the model on your custom dataset with JSON annotations:

```bash
# Evaluate on stamp data
python evaluate_dataset.py --annotation ocr_data/stamp_data_example/stamp_01/stamp_ocr_01.json

# Evaluate on table data
python evaluate_dataset.py --annotation ocr_data/table_data_example/table_01/table_ocr_01.json

# Limit to first 10 samples
python evaluate_dataset.py --annotation ocr_data/stamp_data_example/stamp_01/stamp_ocr_01.json --max-samples 10

# Save results to custom location
python evaluate_dataset.py --annotation ocr_data/stamp_data_example/stamp_01/stamp_ocr_01.json --save-json my_results.json
```

The evaluation script will:
- Use the prompts specified in the JSON file
- Compare predictions against ground truth annotations
- Calculate CER, WER, and exact match metrics
- Generate a detailed report with all results

## Project Structure

```
mini_ocr/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration file
├── config_loader.py               # Configuration loader utility
├── .gitignore                     # Git ignore rules
├── download_model.py              # Model download script
├── DeepSeekOCRDataCollator.py     # Custom data collator for OCR
├── train.py                       # Training script
├── inference.py                   # Inference script
├── eval_model.py                  # Single sample evaluation
├── evaluate_dataset.py            # Dataset evaluation script
└── ocr_data/                      # Custom OCR datasets
    ├── stamp_data_example/        # Stamp/seal detection data
    └── table_data_example/        # Table extraction data
```

## Model Architecture

This project uses:
- **Base Model**: DeepSeek-OCR (vision-language model)
- **Inference**: Pure transformers (for compatibility)
- **Training Framework**: Unsloth for efficient LoRA fine-tuning
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Optimization**: 8-bit AdamW optimizer

**Note**: Inference scripts use pure transformers without Unsloth to avoid compatibility issues. Training still uses Unsloth for speed benefits.

## Data Format

The training data should follow this format:

```python
{
    "image_path": "path/to/image.jpg",
    "text": "extracted text content"
}
```

## Performance

The data collator includes several optimizations:
- Dynamic image preprocessing
- Adaptive cropping based on image dimensions
- Efficient token masking for training
- Support for multiple image scales

## Known Limitations

- Currently trains on a small subset (1000 samples) by default
- Requires GPU for efficient training
- No multi-GPU support yet
- Limited error handling in inference scripts

## TODO

- [ ] Add command-line argument parsing
- [ ] Support for batch inference
- [ ] Add comprehensive unit tests
- [ ] Implement validation during training
- [ ] Add more evaluation metrics (accuracy, precision, recall)
- [ ] Docker support for easy deployment
- [ ] Web API for inference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- DeepSeek-OCR team for the base model
- Unsloth for efficient fine-tuning framework
- HuggingFace for datasets and model hosting
