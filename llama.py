import os
import requests
from llama_cpp import Llama

MODEL_URL = "https://huggingface.co/DevQuasar/HuggingFaceTB.SmolLM2-135M-Instruct-GGUF/resolve/main/HuggingFaceTB.SmolLM2-135M-Instruct.Q4_K_M.gguf"
MODEL_PATH = "model.gguf"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

download_model()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=2
)

def addprompt(prompt):
    res = llm(prompt, max_tokens=100)
    return res["choices"][0]["text"]
