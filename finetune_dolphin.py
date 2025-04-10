import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

def create_model(model_name):
    # Load model in 16-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 16-bit precision
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def prepare_model_for_training(model):
    # Configure LoRA with increased rank for better language adaptation
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    return model

def main():
    # Model name
    model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
    
    # Create model
    model, tokenizer = create_model(model_name)
    
    # Prepare model for training
    model = prepare_model_for_training(model)
    
    # Load your dataset (replace with your dataset)
    # For example:
    # dataset = load_dataset("your_dataset_name")
    # dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512))
    
    # Training arguments with 16-bit precision
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Increased batch size for 16-bit
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-4,
        fp16=True,  # Enable mixed precision
        bf16=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        optim="adamw_torch",
        max_grad_norm=1.0,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # Replace with your dataset
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    main() 