from fastapi import FastAPI
import llama
import os
import requests

MODEL_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/blob/main/smollm2-360m-instruct-q8_0.gguf"
MODEL_PATH = "smollm2-360m-instruct-q8_0.gguf"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
def chat(inputs):
    prompt=inputs
    res=llama.addprompt(prompt)
    return res
app=FastAPI()
@app.get("/")
def home():
    return {"message":"Scraper API"}
@app.post("/chat")
def chatting(text:str):
    return {"message":chat(text)}
@app.get("/model")
def model():
    return {"model":"smollm","params":"360m","quantization":"Q8_0"}
