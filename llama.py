import os
import requests
from llama_cpp import Llama

MODEL_URL = "https://huggingface.co/DevQuasar/HuggingFaceTB.SmolLM2-135M-Instruct-GGUF/resolve/main/HuggingFaceTB.SmolLM2-135M-Instruct.Q4_K_M.gguf"
MODEL_PATH = "model.gguf"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... this may take a moment.")
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status() # Check if download link is valid
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")

download_model()

# Initialize LLM with safety check
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512, # SmolLM2 supports larger context
        n_threads=1,
        verbose=False # Keeps the console clean
    )
except Exception as e:
    print(f"Failed to load model: {e}")

def addprompt(prompt):
    # Using create_chat_completion is better for "Instruct" models
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return output["choices"][0]["message"]["content"]

# Test it
# print(addprompt("Explain photosynthesis in one sentence."))
