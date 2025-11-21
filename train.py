from datasets import load_dataset
from unsloth import FastVisionModel
from transformers import AutoModel

from deepseek_ocr.modeling_deepseekocr import (
    format_messages, text_encode,
    BasicImageTransform, dynamic_preprocess,
)
from DeepSeekOCRDataCollator import DeepSeekOCRDataCollator  # 若保存为独立文件

model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",
)
instruction = "<image>\nFree OCR. "

def convert_to_conversation(sample):
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

dataset = load_dataset("hezarai/parsynth-ocr-200k", split="train[:1000]")
dataset = dataset.rename_column("image_path", "image")
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

model = FastVisionModel.get_peft_model(
    model,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

from transformers import Trainer, TrainingArguments
from unsloth import is_bf16_supported

FastVisionModel.for_training(model)

data_collator = DeepSeekOCRDataCollator(
    tokenizer=tokenizer,
    model=model,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    train_on_responses_only=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
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
    ),
)

trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
