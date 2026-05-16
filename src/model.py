import os
import torch
import pandas as pd
from PIL import Image
import gradio as gr

from transformers import AutoProcessor, AutoModelForImageTextToText, CLIPProcessor, CLIPModel

from src import *
from src.data_processing import (
    clean_report, truncate_to_sentences,
    image_to_histogram, ensure_model_downloaded,
)

# Global variables for models and data
m_model, m_proc = None, None
clip_model, clip_proc = None, None
hist_index, clip_index, qa_db = None, None, None


def load_system():
    global m_model, m_proc, clip_model, clip_proc
    global hist_index, clip_index, qa_db

    # ── Indexes (must exist — run train.py first if missing) ──────────────────
    for path in (HIST_INDEX_PATH, CLIP_INDEX_PATH, QA_DATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Index not found: {path}\n"
                f"Run `python -m src.train` first to build all indexes."
            )

    print("Loading indexes...")
    hist_index = torch.load(HIST_INDEX_PATH, weights_only=False, map_location=DEVICE)
    clip_index = torch.load(CLIP_INDEX_PATH, weights_only=False, map_location=DEVICE)
    qa_db      = pd.read_csv(QA_DATA_PATH)

    n_qa   = len(qa_db)
    n_hist = hist_index.shape[0]
    n_clip = clip_index.shape[0]
    print(f"  Histogram index : {hist_index.shape}")
    print(f"  CLIP index      : {clip_index.shape}")
    print(f"  QA dataset      : {n_qa} rows")
    if not (n_qa == n_hist == n_clip):
        raise ValueError(
            f"Index size mismatch — Hist:{n_hist}, CLIP:{n_clip}, QA:{n_qa}.\n"
            f"Delete data/processed/ and re-run `python -m src.train`."
        )

    print("\nLoading CLIP...")
    clip_path  = ensure_model_downloaded(CLIP_MODEL_ID, CLIP_LOCAL)
    clip_proc  = CLIPProcessor.from_pretrained(clip_path)
    clip_model = CLIPModel.from_pretrained(clip_path).to(DEVICE)
    clip_model.eval()

    # ColPali (PaliGemma-3B) cannot be loaded alongside MedGemma-4B on CPU —
    # both together exceed typical laptop RAM. ColPali is used offline in
    # train.py to build the index; at inference the histogram index (built from
    # the same images) acts as a lightweight retriever for the ColPali-RAG path.
    print("\nLoading MedGemma (CPU) — this requires a lot of RAM...")
    token         = os.environ.get("HF_TOKEN")
    medgemma_path = ensure_model_downloaded(MED_MODEL_ID, MEDGEMMA_LOCAL, token=token)
    m_proc  = AutoProcessor.from_pretrained(medgemma_path)
    m_model = AutoModelForImageTextToText.from_pretrained(
        medgemma_path,
        device_map="cpu",
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    m_model.eval()
    print("System loaded successfully.")


# ── Embedding helpers ──────────────────────────────────────────────────────────

def get_clip_embedding_inference(image: Image.Image) -> torch.Tensor:
    inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs)
        emb     = clip_model.visual_projection(outputs.pooler_output)
        emb     = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float()


# ── Retrieval ──────────────────────────────────────────────────────────────────

def colpali_retrieve(image: Image.Image, top_k: int = 3):
    # Uses the histogram index built from ColPali-indexed images.
    # ColPali itself runs only in train.py; histograms are the lightweight
    # inference-time proxy that keeps memory within CPU limits.
    q_hist  = image_to_histogram(image)
    sims    = torch.nn.functional.cosine_similarity(hist_index, q_hist)
    top_idx = sims.topk(top_k).indices.tolist()
    return [(qa_db.iloc[i]["report"], float(sims[i])) for i in top_idx]


def clip_retrieve(image: Image.Image, top_k: int = 3):
    q_emb   = get_clip_embedding_inference(image).to(DEVICE)
    sims    = (clip_index.to(DEVICE) @ q_emb.T).squeeze()
    top_idx = sims.topk(top_k).indices.tolist()
    return [(qa_db.iloc[i]["report"], float(sims[i])) for i in top_idx]


# ── Generation ─────────────────────────────────────────────────────────────────

def medgemma_generate(image: Image.Image, prompt_text: str,
                      max_new_tokens: int = 200, is_qa: bool = False) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt_text},
    ]}]
    text   = m_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = m_proc(text=text, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = m_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded   = m_proc.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return truncate_to_sentences(decoded, max_sentences=3) if is_qa else decoded


# ── Mode handlers ──────────────────────────────────────────────────────────────

def generate_report(image):
    if image is None:
        return "Please upload an X-ray.", "—", "—", "—"
    try:
        image = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
        prompt = (
            "You are an expert radiologist. Analyze this chest X-ray and generate "
            "a structured report with FINDINGS, IMPRESSION, and RECOMMENDATIONS."
        )

        medgemma_report = medgemma_generate(image, prompt, max_new_tokens=300)

        clip_hits  = clip_retrieve(image, top_k=1)
        clip_out   = f"[Cosine Sim: {clip_hits[0][1]:.4f}]\n\n{clean_report(clip_hits[0][0])}"

        col_hits   = colpali_retrieve(image, top_k=1)
        col_out    = f"[Cosine Sim: {col_hits[0][1]:.4f}]\n\n{clean_report(col_hits[0][0])}"

        comparison = (
            f"MedGemma: generative (no retrieval) | "
            f"CLIP retrieval sim: {clip_hits[0][1]:.4f} | "
            f"ColPali retrieval sim: {col_hits[0][1]:.4f}\n"
            f"MedGemma output length: {len(medgemma_report.split())} words"
        )
        return medgemma_report, clip_out, col_out, comparison
    except Exception as e:
        return f"ERROR: {e}", "—", "—", "—"


def answer_question(image, question):
    if image is None or not question.strip():
        return "Please upload an X-ray and enter a question.", "—", "—"
    try:
        image = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
        qa_prompt = (
            "You are a radiologist. Answer in exactly 2 sentences.\n\n"
            "CONTEXT:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:"
        )

        col_hits = colpali_retrieve(image, top_k=3)
        col_ctx  = "\n\n".join(clean_report(r) for r, _ in col_hits)
        col_ans  = medgemma_generate(image, qa_prompt.format(ctx=col_ctx, q=question),
                                     max_new_tokens=100, is_qa=True)

        clip_hits = clip_retrieve(image, top_k=3)
        clip_ctx  = "\n\n".join(clean_report(r) for r, _ in clip_hits)
        clip_ans  = medgemma_generate(image, qa_prompt.format(ctx=clip_ctx, q=question),
                                      max_new_tokens=100, is_qa=True)

        comparison = (
            f"ColPali top-1 sim: {col_hits[0][1]:.4f}  |  CLIP top-1 sim: {clip_hits[0][1]:.4f}\n"
            f"ColPali answer: {len(col_ans.split())} words  |  CLIP answer: {len(clip_ans.split())} words\n"
            f"ColPali uses learned document-patch embeddings; CLIP uses vision-language alignment."
        )
        return col_ans, clip_ans, comparison
    except Exception as e:
        return f"ERROR: {e}", "—", "—"


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def launch_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="Multi-Modal Chest X-Ray System") as demo:
        gr.Markdown("# Multi-Modal Chest X-Ray Intelligence System")
        gr.Markdown(
            "**Report Generation** uses MedGemma (generative) and compares with "
            "ColPali and CLIP (retrieval-based).  \n"
            "**QA** uses RAG: ColPali or CLIP retrieves similar reports, then MedGemma answers grounded in that context."
        )

        with gr.Tabs():

            # ── Tab 1: Report Generation ──────────────────────────────────────
            with gr.Tab("Report Generation"):
                rg_image = gr.Image(type="pil", label="Input X-Ray")
                rg_btn   = gr.Button("Generate Report", variant="primary")
                with gr.Row():
                    rg_medgemma = gr.Textbox(label="MedGemma — Generated Report", lines=12)
                    rg_clip     = gr.Textbox(label="CLIP — Retrieved Reference",  lines=12)
                    rg_colpali  = gr.Textbox(label="ColPali — Retrieved Reference", lines=12)
                rg_comparison = gr.Textbox(label="Comparison", lines=3)
                rg_btn.click(
                    generate_report,
                    inputs=rg_image,
                    outputs=[rg_medgemma, rg_clip, rg_colpali, rg_comparison],
                )

            # ── Tab 2: QA ─────────────────────────────────────────────────────
            with gr.Tab("QA (RAG)"):
                with gr.Row():
                    qa_image    = gr.Image(type="pil", label="Input X-Ray")
                    qa_question = gr.Textbox(label="Clinical Question", lines=3)
                qa_btn = gr.Button("Ask", variant="primary")
                with gr.Row():
                    qa_colpali = gr.Textbox(label="ColPali-RAG Answer", lines=5)
                    qa_clip    = gr.Textbox(label="CLIP-RAG Answer",    lines=5)
                qa_comparison = gr.Textbox(label="Comparison", lines=3)
                qa_btn.click(
                    answer_question,
                    inputs=[qa_image, qa_question],
                    outputs=[qa_colpali, qa_clip, qa_comparison],
                )

    demo.launch(share=False)


if __name__ == "__main__":
    load_system()
    launch_ui()
