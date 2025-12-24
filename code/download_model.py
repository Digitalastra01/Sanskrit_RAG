"""
Script to download the quantized TinyLlama model from Hugging Face.
"""
import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LOCAL_DIR = "models"

def download_model():
    """
    Downloads the GGUF model to the local directory if it doesn't exist.
    """
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
    
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=LOCAL_DIR)
    print(f"Model downloaded to {model_path}")

if __name__ == "__main__":
    download_model()
