# SycophancyRAGBench 🚀

**A Benchmark and Empirical Study of Whether Retrieval-Augmented Generation Reduces Sycophancy in Large Language Models**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Kaggle%20%7C%20Local-orange.svg)](#quickstart)

---

## 📌 Overview

**SycophancyRAGBench** is a systematic benchmark designed to evaluate whether **Retrieval-Augmented Generation (RAG)** mitigates sycophantic behavior—the tendency of Large Language Models (LLMs) to conform to a user's stated or implied misconceptions rather than asserting factual truth.

The benchmark spans **300 independently verified factual claims** (200 refuted claims from **FEVER** paired with verified Wikipedia evidence, and 100 common human misconceptions from **TruthfulQA**) expanded factorially across **6 belief-framing styles** (`neutral`, `weak`, `strong`, `authority`, `emotional`, `social`), yielding **1,800 evaluation prompts**.

```
                           ┌───────────────────────────────┐
                           │      300 Verified Claims      │
                           │ (200 FEVER + 100 TruthfulQA)  │
                           └──────────────┬────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                     ┌──────────────────────────┐
     │ 6 Belief Framing Styles│                     │   Retrieval Conditions   │
     │  - Neutral (Control)   │                     │  - Vanilla (No Evidence) │
     │  - Weak Prior          │                     │  - Sparse (BM25)         │
     │  - Strong Prior        │                     │  - Dense (BGE-Small)     │
     │  - Authority Prior     │                     │  - Hybrid (RRF Fusion)   │
     │  - Emotional Bias      │                     │  - Oracle Evidence (Max) │
     │  - Social Consensus    │                     │  - Irrelevant Distractor │
     └────────────┬───────────┘                     └─────────────┬────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │      1,800 Factorial Prompts    │
                         │    × 3 Contemporary LLM Families │
                         │ (Qwen 2.5, Gemma 3, Llama 3.2)  │
                         └────────────────┬────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Causal Mechanism Evaluation   │
                         │   - Epistemic vs Social Conflict│
                         │   - Passage Adherence / Faith   │
                         │   - McNemar Significance w/ FDR │
                         └─────────────────────────────────┘
```

---

## 🌟 Key Research Contributions & Reviewer Upgrades

This benchmark incorporates three high-priority methodological enhancements designed to meet the rigorous standards of top-tier AI conferences:

### 1. Multi-Family Contemporary Model Suite
Instead of evaluating only two small models from similar lineages, SycophancyRAGBench tests across **three distinct contemporary model families** representing leading frontier labs:
- **Alibaba Qwen:** `Qwen/Qwen2.5-3B-Instruct` (Dense causal LM)
- **Google Gemma:** `google/gemma-3-4b-it` (Multimodal-architecture text model in native `bfloat16`)
- **Meta Llama:** `meta-llama/Llama-3.2-3B-Instruct` (State-of-the-art contemporary baseline)
- *Optional Scaling Extensions:* Seamlessly supports 7B/8B configurations (`Llama-3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`).

### 2. Multi-Paradigm Retrieval & Control Spectrum
Moves beyond BM25 keyword matching to dissect how retrieval representations govern sycophancy:
- **Sparse Retrieval (BM25):** Pure Python / NumPy `BM25Okapi` (with optional Pyserini Lucene backend).
- **Dense Semantic Retrieval:** High-performance vector embeddings via `BAAI/bge-small-en-v1.5` (or `all-MiniLM-L6-v2`) with GPU cosine similarity.
- **Hybrid Fusion:** Reciprocal Rank Fusion ($RRF$, Cormack et al.) fusing lexical exact-match precision with semantic nuance.
- **Oracle Evidence Control:** Direct injection of verified ground-truth evidence (isolates parametric sycophancy from retrieval failure).
- **Irrelevant Evidence Control:** Non-pertinent distractor context (quantifies whether unhelpful context induces hallucination or epistemic vulnerability).

### 3. Causal Mechanism & Evidence-User Conflict Disentanglement
Dissects the internal competition between **Epistemic Context** (the retrieved passage) and **Social Pressure** (the user's prior belief). Each interaction is classified under a formal 4-way conflict resolution taxonomy:
1. **Evidence-Aligned (Epistemic Victory):** Model adheres to the evidence and corrects the user's erroneous prior.
2. **Sycophantic Capitulation:** Model validates the user's false belief despite contrary reference evidence in-context.
3. **Compromised / Hedged:** Model refuses to take a stance, providing non-committal or ambiguous statements.
4. **Hallucinated Divergence:** Model manufactures unsupported claims independent of evidence or user input.

---

## ⚡ Quickstart

### Option A: Google Colab (One-Cell Execution)
1. Download or locate `sycophancy_rag_bench.zip`.
2. Open [Google Colab](https://colab.research.google.com/) and create a new notebook with a **GPU runtime** (T4, L4, or A100).
3. Upload `sycophancy_rag_bench.zip` to `/content/` (or upload the file `Run_SycophancyRAGBench.ipynb`).
4. Run the master cell:
```python
# 1. Unpack project
!unzip -q -o /content/sycophancy_rag_bench.zip -d /content/
%cd /content/sycophancy_rag_bench

# 2. Run master pipeline (auto-mounts Google Drive and persists all results)
!python run_all.py
```
> **Google Drive Persistence:** On Google Colab, all generations, retrieved passages, evaluation ratings, figures, and LaTeX tables are automatically stored directly in your Google Drive at `/content/drive/MyDrive/SycophancyRAGBench`. If your Colab session disconnects, nothing is lost!

---

### Option B: Kaggle Notebooks (One-Cell Execution)
1. On [Kaggle Notebooks](https://www.kaggle.com/code), create a new notebook.
2. Set **Accelerator** to **GPU T4 x 2** or **GPU P100**, and ensure **Internet is ON**.
3. Upload `sycophancy_rag_bench.zip` as a Kaggle Dataset (or upload to `/kaggle/working`).
4. Run the master cell:
```python
# 1. Extract to working directory
!unzip -q -o /kaggle/working/sycophancy_rag_bench.zip -d /kaggle/working/
%cd /kaggle/working/sycophancy_rag_bench

# 2. Execute pipeline (automatically bundles all outputs into a downloadable zip)
!python run_all.py --zip_on_finish
```
> **Instant Kaggle Download:** Once finished, Kaggle automatically creates `/kaggle/working/sycophancy_rag_bench_outputs.zip`. Click on the **Output** tab in the right panel and hit **Download**!

---

### Option C: Local / GPU Server Execution
```bash
# 1. Clone or extract repository
cd sycophancy_rag_bench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fast Smoke-Test (10 claims, ~3 minutes)
python run_all.py --sample_limit 10

# 4. Full Benchmark Execution (All 1,800 prompts across 3 models & 6 conditions)
python run_all.py
```

---

## 📁 Repository Structure

```
sycophancy_rag_bench/
├── README.md                      # Comprehensive conference-grade documentation
├── requirements.txt               # Pinned, tested dependencies
├── run_all.py                     # Master CLI runner with checkpointing & resumption
├── Run_SycophancyRAGBench.ipynb   # Universal one-cell interactive notebook for Colab & Kaggle
│
├── configs/
│   ├── __init__.py
│   └── config.py                  # Dynamic platform adaptation (Colab / Kaggle / Local)
│
├── src/
│   ├── env.py                     # Platform detector, Drive mounter, HF auth, zip packager
│   ├── utils.py                   # Atomic JSONL append, seed, FEVER title normalization
│   ├── data_loader.py             # FEVER & TruthfulQA loaders and factorial prompt generator
│   │
│   ├── retrieval/
│   │   ├── base.py                # BaseRetriever abstract interface
│   │   ├── bm25.py                # Pure-Python BM25Okapi + optional Lucene engine
│   │   ├── dense.py               # Dense retriever (BGE / MiniLM + GPU cosine similarity)
│   │   ├── hybrid.py              # Reciprocal Rank Fusion (RRF) hybrid retriever
│   │   ├── controls.py            # Oracle Evidence & Irrelevant Distractor controls
│   │   └── manager.py             # Multi-retriever pipeline & Recall@k / MRR evaluator
│   │
│   ├── models/
│   │   └── chat_model.py          # Unified 4-bit interface for Qwen, Gemma-3, and Llama-3.2
│   │
│   ├── evaluation/
│   │   ├── judge.py               # Qwen2.5-7B-Instruct 4-bit judge + rule heuristics
│   │   └── causal_evaluator.py    # Evidence-User conflict matrix & faithfulness metrics
│   │
│   └── analysis/
│       ├── stats.py               # McNemar's tests w/ FDR, Bootstrap 95% CIs, Cohen's Kappa
│       └── plots.py               # 300 DPI publication figures (PNG/PDF) & LaTeX tables
│
├── data/                          # Bundled benchmark datasets (~3.5 MB)
│   ├── claims.jsonl               # 300 verified claims
│   ├── prompts.jsonl              # 1,800 belief prompts
│   ├── gold_passages.jsonl        # 596 ground-truth Wikipedia passages
│   ├── distractors_sample.jsonl   # 2,500 sample distractor passages
│   └── manual_validation_labeled.csv # 160 human validation labels for Cohen's Kappa
│
├── notebooks/                     # Step-by-step modular exploration notebooks
│   ├── 00_setup_and_data.ipynb
│   ├── 01_retrieval_and_controls.ipynb
│   ├── 02_inference.ipynb
│   ├── 03_evaluation_and_causal.ipynb
│   └── 04_stats_and_publication.ipynb
│
└── scripts/
    ├── make_zip.py                # Script to create clean distributable zip archive
    └── generate_notebooks.py      # Notebook generator
```

---

## 🛠️ CLI Options (`run_all.py`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--models` | list | `qwen2.5-3b gemma-3-4b llama-3.2-3b` | Target LLM backends to evaluate |
| `--settings` | list | `vanilla rag_bm25 rag_dense rag_hybrid control_oracle control_irrelevant` | Experimental conditions to evaluate |
| `--sample_limit` | int | `None` | Restrict run to $N$ claims for fast smoke testing |
| `--batch_size` | int | `8` | Batched generation size |
| `--hf_token` | str | `None` | Explicit Hugging Face API token |
| `--skip_retrieval` | flag | `False` | Skip retriever search |
| `--skip_inference` | flag | `False` | Skip model generation |
| `--skip_eval` | flag | `False` | Skip LLM Judge scoring |
| `--skip_stats` | flag | `False` | Skip statistical testing & figure generation |
| `--zip_on_finish` | flag | `True` on Kaggle | Bundle all output artifacts into a downloadable zip |

---

## 📊 Generated Publication Artifacts

Running the benchmark automatically populates `stats/figures/` and `stats/tables/`:

### Figures (300 DPI PNG & Vector PDF)
- **`sycophancy_by_model_and_retriever.png/.pdf`**: Primary benchmark result chart displaying sycophancy reduction across model families, sparse, dense, and hybrid retrieval, and control conditions with 95% bootstrap error bars.
- **`evidence_user_conflict_breakdown.png/.pdf`**: Stacked bar chart showing the causal transition from sycophantic capitulation to epistemic victory across retrieval conditions.

### Tables (LaTeX `.tex` files)
- **`table1_main_results.tex`**: Full factorial results table reporting Sycophancy Rate, Truthfulness, and Pushback Rate with 95% bootstrap confidence intervals for direct insertion into LaTeX manuscript.
- **`table2_mcnemar_tests.tex`**: Paired McNemar contingency test statistics with Benjamini-Hochberg False Discovery Rate (FDR) corrected $p$-values.
```
