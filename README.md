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
python inference.py
```

Edit `inference.py` to specify your image path:
```python
image_file = "your_image.jpg"
```

### Evaluation

Evaluate the model using Character Error Rate (CER):

```bash
python eval_model.py
```

This will evaluate the model on a sample from the test dataset.

## Project Structure

```
mini_ocr/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── env_setup.sh                  # Alternative setup script
├── .gitignore                    # Git ignore rules
├── download_model.py             # Model download script
├── DeepSeekOCRDataCollator.py    # Custom data collator for OCR
├── train.py                      # Training script
├── inference.py                  # Inference script
└── eval_model.py                 # Evaluation script
```

## Model Architecture

This project uses:
- **Base Model**: DeepSeek-OCR (vision-language model)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: Unsloth for efficient training
- **Optimization**: 8-bit AdamW optimizer

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
