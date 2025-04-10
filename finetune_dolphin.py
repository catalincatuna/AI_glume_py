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
from tqdm import tqdm

class DownloadProgress(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n

    def update(self, n):
        super().update(n)
        self._current += n

def create_model(model_name):
    print(f"Downloading model {model_name}...")
    print("This may take a while depending on your internet connection.")
    print("Model size: ~8GB (16GB with 16-bit precision)")
    
    # Create progress bar
    progress_bar = DownloadProgress(
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="Downloading model"
    )
    
    # Load model in 16-bit precision with progress tracking
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=False,
        resume_download=True,
    )
    
    progress_bar.close()
    print("\nModel downloaded successfully!")
    print("Downloading tokenizer...")
    
    # Reset progress bar for tokenizer
    progress_bar = DownloadProgress(
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="Downloading tokenizer"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        resume_download=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    progress_bar.close()
    print("Tokenizer downloaded successfully!")
    print("\nModel and tokenizer are ready for training.")
    
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
    
    # Load and prepare Romanian dataset
    print("Loading and preparing Romanian dataset...")
    with open("input_glume.txt", "r", encoding="utf-8") as f:
        texts = f.read().split("\n")
    
    # Create dataset with Romanian-specific preprocessing
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize with Romanian-specific settings
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_special_tokens_mask=True
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments optimized for Romanian
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,  # Increased epochs for better language learning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=200,  # Increased warmup for better adaptation
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        learning_rate=1e-4,  # Slightly lower learning rate for better stability
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        save_total_limit=2,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving the fine-tuned model...")
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Training completed and model saved!")

if __name__ == "__main__":
    main() 