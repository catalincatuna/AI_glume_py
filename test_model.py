from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_finetuned_model(model_path):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Path to your fine-tuned model
    model_path = "./finetuned_model"
    
    # Load the model
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(model_path)
    
    # Romanian-specific test prompts
    test_prompts = [
        # Translation tests
        "Tradu în română: Hello, how are you?",
        "Tradu în română: The weather is beautiful today.",
        "Tradu în română: I would like to learn more about artificial intelligence.",
        
        # Grammar tests
        "Completează propoziția: Mă duc la magazin să cumpăr...",
        "Scrie o propoziție folosind verbul 'a merge' la timpul trecut:",
        "Formează pluralul pentru 'carte':",
        
        # Cultural context
        "Spune-mi despre tradițiile de Paște din România:",
        "Ce înseamnă 'dragoste' în cultura românească?",
        "Descrie un obicei tradițional românesc:",
        
        # Complex language structures
        "Scrie o scurtă poveste despre un copil care vizitează Castelul Peleș:",
        "Explică conceptul de 'dor' în limba română:",
        "Descrie diferența dintre 'a merge' și 'a se duce':"
    ]
    
    # Generate responses
    print("\nTesting the model on Romanian language capabilities...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main() 