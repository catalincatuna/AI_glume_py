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
from transformers.utils.hub import cached_file

class DownloadProgress(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n

    def update(self, n):
        super().update(n)
        self._current += n

def create_model(model_name):
    # Check if model is already cached
    try:
        model_path = cached_file(model_name, "config.json")
        print(f"Model found in cache at: {model_path}")
        print("Using cached version. No download needed.")
    except:
        print("Model not found in cache. Starting download...")
    
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
    print("\nModel loaded successfully!")
    print("Loading tokenizer...")
    
    # Reset progress bar for tokenizer
    progress_bar = DownloadProgress(
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="Loading tokenizer"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        resume_download=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    progress_bar.close()
    print("Tokenizer loaded successfully!")
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

def load_text_files(directory):
    """Load all .txt files from a directory and combine their contents."""
    texts = []
    print(f"Loading text files from {directory}...")
    
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} text files.")
    
    for file_name in txt_files:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty content
                    texts.append(content)
            print(f"Loaded {file_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
    
    return texts

def main():
    # Model name
    model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
    
    # Create model
    model, tokenizer = create_model(model_name)
    
    # Prepare model for training
    model = prepare_model_for_training(model)
    
    # Load and prepare Romanian dataset from multiple files
    print("Loading and preparing Romanian dataset...")
    input_directory = "./input_data"  # Directory containing your .txt files
    texts = load_text_files(input_directory)
    
    if not texts:
        raise ValueError("No text data found in the input directory!")
    
    print(f"Total number of text samples: {len(texts)}")
    
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
    
    # Training arguments optimized for Romanian with increased epochs
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,  # Increased from 5 to 10 epochs
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=300,  # Increased warmup for more epochs
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        learning_rate=1e-4,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        save_total_limit=3,  # Increased to save more checkpoints
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        # Added early stopping to prevent overfitting
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
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