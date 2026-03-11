from fastapi import FastAPI
import llama
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
