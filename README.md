# Multi-Modal Chest X-Ray Intelligence System

A CPU-local multi-modal AI system for chest X-ray analysis. Combines a generative vision-language model (MedGemma) with two retrieval-based approaches (CLIP and ColPali) to generate radiology reports and answer clinical questions via RAG (Retrieval-Augmented Generation).

Built on the MIMIC-CXR dataset for DSAI 413 — Assignment 2.

---

## Features

- **Report Generation** — MedGemma generates a structured radiology report (Findings / Impression / Recommendations). CLIP and ColPali each retrieve the most similar past report from the indexed database. All three outputs are shown side-by-side with cosine similarity scores.
- **QA (RAG)** — The user asks a clinical question about an uploaded X-ray. ColPali and CLIP independently retrieve the top-3 most relevant past reports. MedGemma answers the question grounded in each set of retrieved context. Both answers are compared.
- **Gradio web interface** — runs locally in your browser, no deployment needed.

---

## Project Structure

```
mimic_project/
├── src/
│   ├── __init__.py          # Central config: model IDs, file paths, device settings
│   ├── data_processing.py   # Utilities: report cleaning, histograms, model download
│   ├── train.py             # Offline index builder — run once before the app
│   └── model.py             # Inference engine + Gradio UI — the running application
├── data/
│   ├── raw/
│   │   ├── mimic_cxr_aug_train.csv   # MIMIC-CXR metadata and reports
│   │   └── images/                   # Chest X-ray JPG files
│   └── processed/                    # Generated indexes (created by train.py)
├── models/                           # Cached model weights (downloaded on first run)
├── requirements.txt
└── REPORT.md
```

---

## Models

| Model | ID | Role |
|---|---|---|
| **MedGemma** | `google/medgemma-1.5-4b-it` | 4B VLM — generates reports and RAG answers |
| **CLIP** | `openai/clip-vit-base-patch32` | 512-dim vision embeddings for image retrieval |
| **ColPali** | `vidore/colpali-v1.2` | Patch-level document retrieval (offline indexing only) |

MedGemma and CLIP are gated models on Hugging Face and require a token and license acceptance. ColPali runs only during `train.py` — it is too large (~6 GB) to co-load with MedGemma (~8 GB) on a CPU-only machine. At inference, a lightweight **histogram index** (64-bin grayscale pixel histogram, cosine similarity) acts as a proxy retriever for the ColPali-RAG path.

---

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face

MedGemma and CLIP are gated models. You must:
1. Accept the license for each model on Hugging Face.
2. Create an access token at `https://huggingface.co/settings/tokens`.
3. Set the token before running:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Or log in once with:

```powershell
huggingface-cli login
```

### 4. Place the MIMIC-CXR data

Put the training CSV at `data/raw/mimic_cxr_aug_train.csv` and the X-ray images at `data/raw/images/`.

---

## Usage

### Step 1 — Build indexes (run once)

```powershell
python -m src.train
```

This reads 100 MIMIC-CXR cases and produces three search indexes in `data/processed/`:

| Output file | Shape | Description |
|---|---|---|
| `mimic_colpali_index.pt` | `[N, 128]` | ColPali mean-pooled patch embeddings |
| `mimic_hist_index.pt` | `[N, 64]` | Grayscale histogram vectors (inference-time proxy) |
| `mimic_clip_index.pt` | `[N, 512]` | CLIP vision embeddings |
| `mimic_qa_dataset.csv` | N rows | Image paths and doctor reports |

### Step 2 — Launch the app

```powershell
python -m src.model
```

Opens a Gradio interface in your browser with two tabs:

- **Report Generation** — upload an X-ray, click "Generate Report"
- **QA (RAG)** — upload an X-ray, type a clinical question, click "Ask"

---

## Architecture

```
[MIMIC-CXR — 100 images + reports]
              |
              v
  ┌─────────────────────────────┐
  │  train.py  (runs once)      │
  │  ColPali  → colpali_idx.pt  │
  │  CLIP     → clip_idx.pt     │
  │  Histogram→ hist_idx.pt     │
  │  Reports  → qa_dataset.csv  │
  └──────────────┬──────────────┘
                 │
                 v
  ┌──────────────────────────────────────┐
  │  model.py  (per query)               │
  │                                      │
  │  Report tab:                         │
  │    MedGemma  → generate report       │
  │    CLIP      → retrieve top-1 match  │
  │    Histogram → retrieve top-1 match  │
  │                                      │
  │  QA tab:                             │
  │    Histogram → top-3 → RAG context   │
  │    CLIP      → top-3 → RAG context   │
  │    MedGemma answers with each        │
  └──────────────┬───────────────────────┘
                 │
                 v
        [Gradio Web Interface]
```

---

## Requirements

- Python 3.10+
- CPU-only (no GPU required); expect several minutes per query for MedGemma
- ~16 GB RAM recommended (MedGemma alone uses ~8 GB)
- ~20 GB disk space for all three model caches

Key packages: `torch`, `transformers`, `colpali-engine`, `gradio`, `accelerate`, `huggingface_hub`

See `requirements.txt` for pinned versions.

---

## Dataset

**MIMIC-CXR** — a de-identified medical dataset of chest X-rays from hospital patients.

| Field | Description |
|---|---|
| `subject_id` | Anonymised patient identifier |
| `image` | Paths to X-ray JPG files |
| `view` | AP, PA, or Lateral projection |
| `text` | Original radiology reports written by doctors |
| `text_augment` | Paraphrased versions (data augmentation) |

100 cases are indexed by default (`INDEX_SIZE = 100` in `src/__init__.py`) to keep the system runnable on CPU.
