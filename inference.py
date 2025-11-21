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

prompt = "<image>\nFree OCR. "
image_file = "your_image.jpg"

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path="output",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=False,
)

print(res)
