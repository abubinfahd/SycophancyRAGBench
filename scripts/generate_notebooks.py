"""
Generates clean, modular step-by-step notebooks for interactive exploration and inspection in Colab/Kaggle.
"""

import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

# ------------------------------------------------------------------------------
# Notebook 00: Setup & Data
# ------------------------------------------------------------------------------
nb00_cells = [
    md_cell("# 00 · Setup & Dataset Construction — *SycophancyRAGBench*\n\nBuilds the 300 factual claims (200 FEVER refuted + 100 TruthfulQA misconceptions) and generates 1,800 factorial belief-prompt variants across 6 framing styles."),
    code_cell("""# Bootstrap environment (Colab / Kaggle / Local)
import sys, os
from pathlib import Path

# Add project root to path
pkg_root = os.path.abspath("..") if os.path.exists("../configs") else os.path.abspath(".")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from configs import config as cfg
from src import env, utils as U
from src.data_loader import load_or_build_benchmark, print_data_summary

env.setup_workspace()
U.set_seed(cfg.SEED)
print("Bootstrap complete. Environment:", env.get_runtime_name())
"""),
    code_cell("""# Load or build benchmark claims & prompts
claims, prompts = load_or_build_benchmark(force_rebuild=False)
print_data_summary(prompts)
"""),
    code_cell("""# Inspect sample claim and belief-prompt variants
import pandas as pd
df = pd.DataFrame(prompts)
sample_id = claims[0]["claim_id"]
print("FALSE CLAIM:", claims[0]["false_claim"], "\\n")
for p in prompts:
    if p["claim_id"] == sample_id:
        print(f"[{p['style'].upper():<9}] {p['prompt_text']}")
""")
]
with open(os.path.join(NOTEBOOKS_DIR, "00_setup_and_data.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_nb(nb00_cells), f, indent=2)

# ------------------------------------------------------------------------------
# Notebook 01: Retrieval & Controls
# ------------------------------------------------------------------------------
nb01_cells = [
    md_cell("# 01 · Multi-Paradigm Retrieval & Controls — *SycophancyRAGBench*\n\nCompares Sparse (BM25), Dense (BGE), and Hybrid (RRF) retrieval alongside Oracle Evidence and Irrelevant Distractor controls."),
    code_cell("""import sys, os
pkg_root = os.path.abspath("..") if os.path.exists("../configs") else os.path.abspath(".")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from configs import config as cfg
from src import env, utils as U
from src.data_loader import load_or_build_benchmark
from src.retrieval.manager import run_retrieval_pipeline

env.setup_workspace()
claims, prompts = load_or_build_benchmark()
print(f"Loaded {len(claims)} claims.")
"""),
    code_cell("""# Run all retrieval paradigms & experimental controls
retrievers = ["bm25", "dense", "hybrid", "oracle", "irrelevant"]
results_paths = run_retrieval_pipeline(claims, retrievers_to_run=retrievers, k=cfg.TOP_K)
"""),
    code_cell("""# Inspect comparative retrieval quality (Recall@k & MRR@10)
metrics = U.load_json(cfg.RETRIEVAL_METRICS)
import pandas as pd
table = []
for m, vals in metrics.items():
    if "recall_at_k" in vals:
        row = {"Retriever": m.upper(), "MRR@10": vals.get("mrr_at_10")}
        row.update(vals["recall_at_k"])
        table.append(row)

df_retr = pd.DataFrame(table)
print("=== RETRIEVAL COMPARISON (FEVER GOLD EVIDENCE) ===")
print(df_retr.to_string(index=False))
""")
]
with open(os.path.join(NOTEBOOKS_DIR, "01_retrieval_and_controls.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_nb(nb01_cells), f, indent=2)

# ------------------------------------------------------------------------------
# Notebook 02: Inference
# ------------------------------------------------------------------------------
nb02_cells = [
    md_cell("# 02 · Multi-Model Inference — *SycophancyRAGBench*\n\nRuns batched inference across model families (Qwen 2.5, Gemma 3, Llama 3.2) under Vanilla, RAG (Sparse/Dense/Hybrid), and Control conditions."),
    code_cell("""import sys, os
pkg_root = os.path.abspath("..") if os.path.exists("../configs") else os.path.abspath(".")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from configs import config as cfg
from src import env
from run_all import run_pipeline

env.setup_workspace()
env.authenticate_hf()

class InferenceArgs:
    models = ["qwen2.5-3b", "gemma-3-4b", "llama-3.2-3b"]
    settings = ["vanilla", "rag_bm25", "rag_dense", "rag_hybrid", "control_oracle", "control_irrelevant"]
    sample_limit = None
    batch_size = 8
    hf_token = None
    skip_retrieval = True
    skip_inference = False
    skip_eval = True
    skip_stats = True
    zip_on_finish = False

run_pipeline(InferenceArgs())
""")
]
with open(os.path.join(NOTEBOOKS_DIR, "02_inference.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_nb(nb02_cells), f, indent=2)

# ------------------------------------------------------------------------------
# Notebook 03: Evaluation & Causal Analysis
# ------------------------------------------------------------------------------
nb03_cells = [
    md_cell("# 03 · Evaluation & Causal Conflict Analysis — *SycophancyRAGBench*\n\nEvaluates generated outputs using Qwen2.5-7B-Instruct judge and performs fine-grained causal Evidence-User conflict resolution breakdown."),
    code_cell("""import sys, os
pkg_root = os.path.abspath("..") if os.path.exists("../configs") else os.path.abspath(".")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from configs import config as cfg
from src import env
from run_all import run_pipeline

env.setup_workspace()

class EvalArgs:
    models = ["qwen2.5-3b", "gemma-3-4b", "llama-3.2-3b"]
    settings = ["vanilla", "rag_bm25", "rag_dense", "rag_hybrid", "control_oracle", "control_irrelevant"]
    sample_limit = None
    batch_size = 8
    hf_token = None
    skip_retrieval = True
    skip_inference = True
    skip_eval = False
    skip_stats = False
    zip_on_finish = True

run_pipeline(EvalArgs())
""")
]
with open(os.path.join(NOTEBOOKS_DIR, "03_evaluation_and_causal.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_nb(nb03_cells), f, indent=2)

# ------------------------------------------------------------------------------
# Notebook 04: Stats & Publication Figures
# ------------------------------------------------------------------------------
nb04_cells = [
    md_cell("# 04 · Statistical Significance & Publication Artifacts — *SycophancyRAGBench*\n\nComputes paired McNemar tests with Benjamini-Hochberg FDR adjustments, Bootstrap 95% CIs, Cohen's Kappa, and generates publication-grade 300 DPI figures and LaTeX tables."),
    code_cell("""import sys, os
pkg_root = os.path.abspath("..") if os.path.exists("../configs") else os.path.abspath(".")
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from configs import config as cfg
from src import env, utils as U
import pandas as pd
from src.analysis.plots import generate_all_publication_artifacts
from src.analysis.stats import paired_mcnemar_test, apply_fdr_correction, compute_all_bootstrap_cis, compute_cohen_kappas
from IPython.display import Image, display

env.setup_workspace()
judged = U.read_jsonl(cfg.JUDGED_PATH)
df = pd.DataFrame(judged)
print(f"Loaded {len(df)} judged records.")

# Compute McNemar significance
models = sorted(df['model'].unique())
settings = sorted(df['setting'].unique())
mcnemar_rows = []
for m in models:
    for s in settings:
        if s != "vanilla":
            mcnemar_rows.append(paired_mcnemar_test(df, m, "j_endorses", "vanilla", s, belief_only=True))

mcnemar_rows = apply_fdr_correction(mcnemar_rows)
generate_all_publication_artifacts(df, mcnemar_rows)

fig1 = os.path.join(cfg.FIG_DIR, "sycophancy_by_model_and_retriever.png")
if os.path.exists(fig1):
    display(Image(fig1))
""")
]
with open(os.path.join(NOTEBOOKS_DIR, "04_stats_and_publication.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_nb(nb04_cells), f, indent=2)

print("Modular notebooks 00 - 04 created successfully.")
