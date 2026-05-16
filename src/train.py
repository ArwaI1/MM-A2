import os
import gc
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from colpali_engine.models import ColPali, ColPaliProcessor

from src import *
from src.data_processing import index_is_valid, image_to_histogram, ensure_model_downloaded

def get_clip_embedding(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        emb     = model.visual_projection(outputs.pooler_output)
        emb     = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float()

def build_colpali_index():
    print(f"\n---> Building ColPali index ({INDEX_SIZE} cases)...")
    gc.collect()

    if not os.path.exists(LOCAL_CSV):
        raise FileNotFoundError(f"Missing base CSV file at: {LOCAL_CSV}. Place it in data/raw/")
    
    df = pd.read_csv(LOCAL_CSV).head(INDEX_SIZE)
    path_lookup = {
        os.path.splitext(f)[0]: os.path.join(LOCAL_IMAGES, f)
        for f in os.listdir(LOCAL_IMAGES) if f.endswith(('.jpg', '.png'))
    }

    print("Loading ColPali for indexing...")
    colpali_path = ensure_model_downloaded(COL_MODEL_ID, COLPALI_LOCAL)
    c_model = ColPali.from_pretrained(colpali_path, torch_dtype=DTYPE, device_map="cpu")
    c_proc  = ColPaliProcessor.from_pretrained(colpali_path)

    embs, qa_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Vectorizing"):
        raw_val = str(row['image'])
        try:
            img_list    = ast.literal_eval(raw_val)
            target_path = img_list[0] if isinstance(img_list, list) else raw_val
        except Exception:
            target_path = raw_val

        img_id    = os.path.splitext(os.path.basename(str(target_path)))[0]
        full_path = path_lookup.get(img_id)
        if not full_path:
            continue

        try:
            img   = Image.open(full_path).convert("RGB")
            batch = c_proc.process_images([img])
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                emb = c_model(**batch).mean(dim=1)
            embs.append(emb.cpu().float())
            qa_list.append({"path": full_path, "report": row['text']})
        except Exception as e:
            pass

    if not embs:
        raise ValueError("No embeddings generated. Check your images directory.")

    full_idx = torch.cat(embs, dim=0)
    qa_df    = pd.DataFrame(qa_list)
    torch.save(full_idx, INDEX_PATH)
    qa_df.to_csv(QA_DATA_PATH, index=False)

    print("Building histogram index...")
    hist_list = [image_to_histogram(Image.open(p).convert("RGB")) for p in qa_df["path"].tolist()]
    hist_idx = torch.cat(hist_list, dim=0)
    torch.save(hist_idx, HIST_INDEX_PATH)

    del c_model, c_proc
    gc.collect()
    print(f"✅ ColPali index ready: {full_idx.shape}")
    return full_idx, qa_df, hist_idx

if __name__ == "__main__":
    print("==========================================================")
    print(" 🫁 Data Pipeline: Index Builder (Local CPU)")
    print("==========================================================")
    
    # 1. COLPALI & HISTOGRAMS
    needs_rebuild = (not index_is_valid(INDEX_PATH) or not index_is_valid(QA_DATA_PATH))
    if needs_rebuild:
        v_index, qa_db, hist_index = build_colpali_index()
    else:
        qa_db = pd.read_csv(QA_DATA_PATH)
        print("✅ ColPali & Histogram Indexes already exist and are valid.")

    # 2. CLIP
    print("\n---> Checking CLIP index...")
    clip_valid = index_is_valid(CLIP_INDEX_PATH)
    if not clip_valid:
        print("Loading CLIP Model for Indexing...")
        clip_path  = ensure_model_downloaded(CLIP_MODEL_ID, CLIP_LOCAL)
        clip_proc  = CLIPProcessor.from_pretrained(clip_path)
        clip_model = CLIPModel.from_pretrained(clip_path).to(DEVICE)

        print(f"Building CLIP index for {len(qa_db)} cases...")
        clip_embs = []
        for img_path in tqdm(qa_db["path"].tolist(), desc="Encoding"):
            img = Image.open(img_path).convert("RGB")
            clip_embs.append(get_clip_embedding(img, clip_proc, clip_model))
        
        clip_index = torch.cat(clip_embs, dim=0)
        torch.save(clip_index, CLIP_INDEX_PATH)
        print("✅ CLIP index ready.")
    else:
        print("✅ CLIP Index already exists and is valid.")
        
    print("\n🎉 Training / Indexing complete! You can now run model.py")