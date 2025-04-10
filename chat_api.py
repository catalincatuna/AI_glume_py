from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "cognitivecomputations/dolphin-2.9-llama3-8b"

# Initialize the model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Initialize conversation chain with memory
def create_conversation_chain(llm):
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation

# Global conversation chain
conversation_chain = None

@app.on_event("startup")
async def startup_event():
    global conversation_chain
    model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
    llm = load_model(model_name)
    conversation_chain = create_conversation_chain(llm)

@app.post("/chat")
async def chat(request: ChatRequest):
    global conversation_chain
    
    if not conversation_chain:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Format the conversation history
        conversation_history = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in request.messages
        ])
        
        # Get response from the model
        response = conversation_chain.predict(input=conversation_history)
        
        return {
            "response": response,
            "model": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 