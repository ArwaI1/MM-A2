
import os
import torch

# Force Hugging Face to use D: drive for all hidden dependencies
os.environ["HF_HOME"] = r"D:\huggingface_cache"

# 1. LOCAL CONFIGURATION
MED_MODEL_ID  = "google/medgemma-1.5-4b-it"
COL_MODEL_ID  = "vidore/colpali-v1.2"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

INDEX_SIZE = 100   # Max rows to process for now
DEVICE = "cpu"
DTYPE  = torch.float32

# 2. DYNAMIC PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
INDEX_DIR   = os.path.join(DATA_DIR, "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# Data paths
LOCAL_CSV    = os.path.join(RAW_DIR, "mimic_cxr_aug_train.csv")
LOCAL_IMAGES = os.path.join(RAW_DIR, "images")

# Local cache paths for Indexes
INDEX_PATH      = os.path.join(INDEX_DIR, "mimic_colpali_index.pt")
QA_DATA_PATH    = os.path.join(INDEX_DIR, "mimic_qa_dataset.csv")
CLIP_INDEX_PATH = os.path.join(INDEX_DIR, "mimic_clip_index.pt")
HIST_INDEX_PATH = os.path.join(INDEX_DIR, "mimic_hist_index.pt")

# Local cache paths for Models
MEDGEMMA_LOCAL = os.path.join(MODELS_DIR, "medgemma")
CLIP_LOCAL     = os.path.join(MODELS_DIR, "clip")
COLPALI_LOCAL  = os.path.join(MODELS_DIR, "colpali")

# Ensure directories exist
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)