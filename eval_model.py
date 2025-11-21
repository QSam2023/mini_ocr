from datasets import load_dataset
from jiwer import cer
from unsloth import FastVisionModel
from transformers import AutoModel

model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",
)
dataset = load_dataset("hezarai/parsynth-ocr-200k", split="train[:2000]")

# 使用样本 1523 进行评估
sample = dataset[1523]
sample["image_path"].save("eval.jpg")

gt = sample["text"]

res = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file="eval.jpg",
    output_path="eval_output",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=False,
)

pred = res[-1]
print("Prediction:", pred)
print("Ground truth:", gt)

print("CER:", cer(gt, pred))
