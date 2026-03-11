from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from llama_cpp import Llama

MODEL_URL = "https://huggingface.co/RichardErkhov/HuggingFaceTB_-_SmolLM-135M-Instruct-gguf/resolve/main/SmolLM-135M-Instruct.IQ4_XS.gguf"
MODEL_PATH = "SmolLM-135M-Instruct.IQ4_XS.gguf"

# Auto-download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(MODEL_PATH, "wb") as f:
        for data in response.iter_content(1024 * 1024):
            f.write(data)
    print("Model downloaded!")

# Load the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=1
)

# FastAPI setup
app = FastAPI()

class Response(BaseModel):
    Generated: str

@app.get("/chat", response_model=Response)
def chat(q: str):
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    
    output = llm(
        prompt,
        max_tokens=150,
        temperature=0.4,
        stop=["<|im_end|>"]
    )
    
    text = output["choices"][0]["text"].strip()
    return {"Generated": text}

@app.get("/")
def home():
    return {"message": "Chat-API live! Visit /docs for more info."}
