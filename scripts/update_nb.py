import json
import os

nb_path = "e:/CODE/SC/Papers/SycophancyRAGBench/sycophancy_rag_bench/Run_SycophancyRAGBench.ipynb"

nb = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True)
    })

add_md("""# 🚀 SycophancyRAGBench: Master One-Cell Execution
### *A Benchmark and Empirical Study of Retrieval-Augmented Generation & Sycophancy in LLMs*

> **Publication Target:** Top-Tier AI Conferences (ACL / EMNLP / NeurIPS / ICLR)
> 
> **Key Capabilities:**
> - 🏢 **Multi-Family Contemporary Models:** Qwen 2.5 (Alibaba), Gemma 3 (Google), and Llama 3.2 (Meta)
> - 🔍 **Multi-Paradigm Retrieval:** Sparse (BM25), Dense (BGE/MiniLM), Hybrid (RRF)
> - ⚖️ **Rigorous Experimental Controls:** Oracle Evidence (upper bound) & Irrelevant Distractors (noise floor)
> - 🔬 **Causal Mechanism Analysis:** Evidence-User Conflict Matrix, Faithfulness vs. Capitulation
> - 💾 **Auto-Persistence:** Live sync to Google Drive on Colab; Instant downloadable ZIP on Kaggle!
""")

add_md("""## ⚡ Quickstart Instructions

| Platform | Steps |
| :--- | :--- |
| **Google Colab** | 1. Upload `sycophancy_rag_bench.zip` to Colab or clone repository.<br>2. Set Runtime to **GPU** (T4 or A100).<br>3. Run the cell below. Enter your Hugging Face Token when prompted (for Gemma-3 and Llama-3.2).<br>4. Everything automatically saves directly to Google Drive (`/content/drive/MyDrive/SycophancyRAGBench`). |
| **Kaggle** | 1. Upload `sycophancy_rag_bench.zip` to notebook or add as input dataset.<br>2. Turn on **GPU accelerator** (T4 x 2 or P100) and **Internet ON**.<br>3. Run the cell below. Enter your Hugging Face Token when prompted.<br>4. All results are automatically zipped into `/kaggle/working/sycophancy_rag_bench_outputs.zip` for 1-click download! |
""")

one_cell_code = """# ==============================================================================
# 🚀 ONE-CELL UNIVERSAL RUNNER (Google Colab & Kaggle Notebooks)
# ==============================================================================
import os, sys, shutil

# 1. Detect environment and locate package directory
if os.path.exists('/content'):
    print('[Platform] Detected Google Colab.')
    if os.path.exists('/content/sycophancy_rag_bench.zip'):
        print('[Setup] Extracting sycophancy_rag_bench.zip...')
        !unzip -q -o /content/sycophancy_rag_bench.zip -d /content/
elif os.path.exists('/kaggle/working'):
    print('[Platform] Detected Kaggle Notebooks.')
    if os.path.exists('/kaggle/input/sycophancy-rag-bench/sycophancy_rag_bench.zip'):
        !unzip -q -o /kaggle/input/sycophancy-rag-bench/sycophancy_rag_bench.zip -d /kaggle/working/
    elif os.path.exists('/kaggle/working/sycophancy_rag_bench.zip'):
        !unzip -q -o /kaggle/working/sycophancy_rag_bench.zip -d /kaggle/working/

pkg_candidates = [
    os.getcwd(),
    os.path.join(os.getcwd(), 'sycophancy_rag_bench'),
    '/content/sycophancy_rag_bench',
    '/kaggle/working/sycophancy_rag_bench'
]
PKG_DIR = next((p for p in pkg_candidates if os.path.exists(os.path.join(p, 'run_all.py'))), os.getcwd())
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
os.chdir(PKG_DIR)
print(f'[Setup] Working directory set to: {PKG_DIR}')

# 2. Install dependencies (quietly)
print('[Setup] Verifying and installing dependencies...')
!pip install -q -U "transformers>=4.51.0" accelerate bitsandbytes sentence-transformers datasets statsmodels scikit-learn matplotlib tqdm

# 3. Benchmark Run Configuration (Adjust as desired!)
MODELS_TO_RUN = ['qwen2.5-3b', 'gemma-3-4b', 'llama-3.2-3b']
SETTINGS_TO_RUN = ['vanilla', 'rag_bm25', 'rag_dense', 'rag_hybrid', 'control_oracle', 'control_irrelevant']

# 🔑 Hugging Face Authentication Token:
# Paste your token here (e.g. "hf_..."), or leave as None to be prompted interactively in output:
HF_TOKEN = None

# 📊 Dataset Scale Configuration:
# - Set TAKE_ALL_DATA = False for standard balanced conference benchmark (300 claims / 1800 prompts)
# - Set TAKE_ALL_DATA = True to extract 100% of ALL data from FEVER & TruthfulQA!
TAKE_ALL_DATA = False

# 🔄 Fresh Run Configuration (No Cache):
# - Set FRESH_RUN = True to wipe all previous outputs, caches, and indexes for a 100% fresh run!
# - Set FRESH_RUN = False to resume an interrupted session without re-computing existing outputs
FRESH_RUN = True

# 4. Execute the End-to-End Pipeline
from run_all import run_pipeline

class ConfigArgs:
    models = MODELS_TO_RUN
    settings = SETTINGS_TO_RUN
    fresh = FRESH_RUN
    all_data = TAKE_ALL_DATA
    sample_limit = SAMPLE_LIMIT
    batch_size = 8
    hf_token = HF_TOKEN
    skip_retrieval = False
    skip_inference = False
    skip_eval = False
    skip_stats = False
    zip_on_finish = True

print('\\n' + '='*60)
print('STARTING SYCOPHANCYRAGBENCH BENCHMARK EXECUTION...')
print(f'Models:   {MODELS_TO_RUN}')
print(f'Settings: {SETTINGS_TO_RUN}')
print(f'Fresh:    {FRESH_RUN} (No cache, ground-zero run)')
print(f'All Data: {TAKE_ALL_DATA}')
print(f'Sample:   {SAMPLE_LIMIT if SAMPLE_LIMIT else "FULL"}')
print('='*60 + '\\n')

run_pipeline(ConfigArgs())

# 5. Display Generated Publication Figures Inline
from IPython.display import Image, display
from configs import config as cfg

fig1 = os.path.join(cfg.FIG_DIR, 'sycophancy_by_model_and_retriever.png')
fig2 = os.path.join(cfg.FIG_DIR, 'evidence_user_conflict_breakdown.png')

if os.path.exists(fig1):
    print('\\n--- Generated Figure 1: Sycophancy by Model & Retriever ---')
    display(Image(fig1))

if os.path.exists(fig2):
    print('\\n--- Generated Figure 2: Causal Evidence-User Conflict Breakdown ---')
    display(Image(fig2))
"""

add_code(one_cell_code)

add_md("""## 📊 Inspecting & Exporting Outputs

All generated experimental data is automatically organized:
- `generations/`: Raw model responses (`<model>__<setting>.jsonl`)
- `retrieval/`: Retrieved passages and Recall/MRR metrics (`retrieval_<method>.jsonl`)
- `eval/`: LLM Judge ratings & causal conflict breakdown (`judged.jsonl`)
- `stats/`: Paired McNemar significance tests, Bootstrap CIs, Cohen Kappa scores
- `stats/figures/`: Publication-ready 300 DPI PNG and vector PDF charts
- `stats/tables/`: Formatted LaTeX tables ready to paste directly into your LaTeX paper!

**Download Locations:**
- **On Kaggle:** Download `sycophancy_rag_bench_outputs.zip` from `/kaggle/working` in the right panel under *Output*.
- **On Colab:** Check your Google Drive folder: `/content/drive/MyDrive/SycophancyRAGBench`.
""")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Run_SycophancyRAGBench.ipynb successfully.")
