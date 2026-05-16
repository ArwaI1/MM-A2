import os
import ast
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import snapshot_download
from src import DEVICE

# Model files that confirm a folder is a valid cached model
_MODEL_EXTENSIONS = ('.bin', '.safetensors', '.pt', '.gguf')
_CONFIG_FILES     = ('config.json', 'tokenizer_config.json', 'preprocessor_config.json',
                     'adapter_config.json', 'special_tokens_map.json')

def _resolve_token(token: str | None) -> str | None:
    """Return token: explicit arg → env var → HF credential store → None."""
    if token:
        return token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

def ensure_model_downloaded(repo_id: str, local_path: str, token: str = None) -> str:
    os.makedirs(local_path, exist_ok=True)

    existing = os.listdir(local_path)
    has_weights = any(f.endswith(_MODEL_EXTENSIONS) for f in existing)
    has_config  = any(f in _CONFIG_FILES for f in existing)
    is_cached   = has_weights and has_config

    if not is_cached:
        resolved_token = _resolve_token(token)
        if resolved_token is None:
            raise RuntimeError(
                f"\n\nNo HuggingFace token found — cannot download '{repo_id}'.\n"
                f"This is a gated model that requires authentication. Steps to fix:\n"
                f"  1. Accept the license at: https://huggingface.co/{repo_id}\n"
                f"  2. Create a token at:      https://huggingface.co/settings/tokens\n"
                f"  3. In PowerShell before running:\n"
                f"       $env:HF_TOKEN = 'hf_your_token_here'\n"
                f"     Or log in once with:    huggingface-cli login\n"
            )
        print(f"Downloading '{repo_id}' to '{local_path}' (happens once)...")
        snapshot_download(repo_id=repo_id, local_dir=local_path, token=resolved_token)
        print(f"'{repo_id}' downloaded successfully!")
    else:
        print(f"Fast-loading '{repo_id}' from local folder...")

    return local_path

def index_is_valid(path, min_rows=10):
    if not os.path.exists(path):
        return False
    try:
        if path.endswith(".pt"):
            data = torch.load(path, weights_only=False, map_location=DEVICE)
            return (isinstance(data, torch.Tensor) and data.shape[0] >= min_rows)
        elif path.endswith(".csv"):
            return len(pd.read_csv(path)) >= min_rows
    except Exception:
        return False
    return False

def image_to_histogram(image: Image.Image, bins=64) -> torch.Tensor:
    img_gray = image.convert("L").resize((224, 224))
    hist     = np.histogram(np.array(img_gray), bins=bins, range=(0, 256))[0]
    hist     = hist.astype(np.float32)
    hist     = hist / (hist.sum() + 1e-8)
    return torch.tensor(hist).unsqueeze(0)

def clean_report(raw: str) -> str:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return "\n\n".join(str(r).strip() for r in parsed)
    except Exception:
        pass
    raw = raw.strip()
    if raw.startswith("['") and raw.endswith("']"):
        raw = raw[2:-2]
    return raw.replace("', '", "\n\n").strip()

def truncate_to_sentences(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result    = " ".join(sentences[:max_sentences]).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result