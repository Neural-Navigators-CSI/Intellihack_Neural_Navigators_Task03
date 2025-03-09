from datasets import load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import Trainer
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_from_disk

# train_dataset = load_from_disk("Data/train-dataset")
# test_dataset = load_from_disk("Data/test-dataset")


def format_prompt(example):
    return {"text": f"### Question: {example['question']}\n### Answer: {example['answer']} <|seperator|> "}

# train_dataset = train_dataset.map(format_prompt)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Qwen 2.5 3B with quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank of LoRA adapters
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()

print("Model with LoRA adapters loaded successfully!")

# def preprocess_function(examples):
#     return {
#         'text': [f"Question: {q} Answer: {a}" for q, a in zip(examples['question'], examples['answer'])]
#     }
#
# train_dataset = train_dataset.map(preprocess_function, batched=True)
#
# # Define the training arguments
# training_args = TrainingArguments(
#     output_dir="./qwen_finetuned",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     learning_rate=3e-4,
#     max_steps=50,
#     warmup_steps=5,
#     fp16=True,
#     logging_steps=10,
#     save_steps=25,
#     optim="adamw_8bit",
#     disable_tqdm=False,
# )
#
# # Initialize the trainer
# trainer = SFTTrainer(
#     model=model,
#     processing_class=tokenizer,
#     train_dataset=train_dataset,
#     args=training_args,
# )
# print("Starting fine-tuning...")
# trainer.train()

model.save_pretrained("./qwen_finetuned")
tokenizer.save_pretrained("./qwen_finetuned")