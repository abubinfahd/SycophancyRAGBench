# SycophancyRAGBench Directory Structure

**Root:** `E:\CODE\SC\Papers\SycophancyRAGBench\sycophancy_rag_bench`  
**Focus:** Repository layout, file responsibilities, input data schemas, scripts/notebooks, and runtime artifact layout.

---

## 1. Repository Tree

```
sycophancy_rag_bench/
├── README.md                                  # Project overview, quickstart, CLI flags, citation
├── requirements.txt                           # Python dependencies (PyTorch, Transformers >=4.51.0, etc.)
├── run_all.py                                 # Master CLI pipeline runner (350 lines)
├── Run_SycophancyRAGBench.ipynb               # One-cell universal runner notebook for Colab & Kaggle
│
├── configs/
│   ├── __init__.py                            # Re-exports config module as cfg
│   └── config.py                              # Runtime detection, paths, models, retrieval/judge/gen params (200 lines)
│
├── src/
│   ├── env.py                                 # Colab/Kaggle/Local workspace setup, HF auth, cache reset, zip packager
│   ├── utils.py                               # Seed, atomic JSON/JSONL IO, FEVER normalization, domain & prompt helpers
│   ├── data_loader.py                         # FEVER & TruthfulQA dataset loaders and factorial prompt generator
│   │
│   ├── retrieval/
│   │   ├── __init__.py                        # Exports BaseRetriever, BM25, Dense/Hybrid (guarded), controls, manager
│   │   ├── base.py                            # Abstract BaseRetriever.search(query, k)
│   │   ├── bm25.py                            # PureBM25, PyseriniLuceneBM25, build_bm25_retriever()
│   │   ├── dense.py                           # DenseRetriever (BGE / MiniLM + cached embeddings), build_dense_retriever()
│   │   ├── hybrid.py                          # HybridRetriever (Reciprocal Rank Fusion over BM25 + Dense)
│   │   ├── controls.py                        # OracleRetriever (gold passage control), IrrelevantRetriever (distractor control)
│   │   └── manager.py                         # run_retrieval_pipeline(), evaluate_retrieval_quality() (Recall@k, MRR@10)
│   │
│   ├── models/
│   │   ├── __init__.py                        # Exports ChatModel
│   │   └── chat_model.py                      # 4-bit NF4 ChatModel wrapper (Qwen 2.5, Gemma 3, Llama 3.x, Mistral)
│   │
│   ├── evaluation/
│   │   ├── __init__.py                        # Exports Judge, rule_signals, compute_evidence_overlap, causal evaluators
│   │   ├── judge.py                           # Qwen2.5-7B-Instruct 4-bit Judge + lexical heuristics
│   │   └── causal_evaluator.py                # 4-way classify_conflict_resolution(), compute_causal_conflict_metrics()
│   │
│   └── analysis/
│       ├── __init__.py                        # Exports stats & plotting functions
│       ├── stats.py                           # paired_mcnemar_test(), apply_fdr_correction(), bootstrap CIs, Cohen's Kappa
│       └── plots.py                           # 300 DPI PNG/PDF charts & LaTeX table exporters
│
├── data/                                      # Bundled benchmark inputs (~2.69 MB total)
│   ├── claims.jsonl                           # 300 verified claims (FEVER + TruthfulQA)
│   ├── prompts.jsonl                          # 1,800 factorial belief prompts (300 x 6 styles)
│   ├── gold_passages.jsonl                    # 596 chunked ground-truth Wikipedia passages
│   ├── distractors_sample.jsonl               # 2,500 sample distractor Wikipedia passages
│   └── manual_validation_labeled.csv          # Human validation labels (160 lines, header + records)
│
├── notebooks/                                 # Empty directory in current snapshot (see scripts/notebooks/)
│
├── scripts/
│   ├── generate_notebooks.py                  # Generates modular notebooks 00..04 into scripts/notebooks/
│   ├── make_zip.py                            # Packages sycophancy_rag_bench.zip (excludes artifacts, index, __pycache__)
│   ├── update_nb.py                           # Regenerates Run_SycophancyRAGBench.ipynb
│   └── notebooks/                             # Generated modular notebooks
│       ├── 00_setup_and_data.ipynb
│       ├── 01_retrieval_and_controls.ipynb
│       ├── 02_inference.ipynb
│       ├── 03_evaluation_and_causal.ipynb
│       └── 04_stats_and_publication.ipynb
│
└── artifacts/                                 # Local runtime PROJECT_DIR (partial snapshot currently present)
    ├── data/
    │   ├── claims.jsonl                       # 300 claims (mirrored from data/)
    │   └── prompts.jsonl                      # 1,800 prompts (mirrored from data/)
    └── stats/
        ├── figures/
        │   ├── evidence_user_conflict_breakdown.pdf
        │   ├── evidence_user_conflict_breakdown.png
        │   ├── sycophancy_by_model_and_retriever.pdf
        │   └── sycophancy_by_model_and_retriever.png
        └── tables/
            ├── table1_main_results.tex
            └── table2_mcnemar_tests.tex
```

---

## 2. Top-Level Entry Points & Configuration

| File | Role | Key Symbols / Contents |
| :--- | :--- | :--- |
| [`run_all.py`](../../run_all.py) | Master CLI pipeline runner | `parse_args()`, `load_retrieval_evidence_index()`, `get_user_prompt_and_evidence()`, `run_pipeline(args)` |
| [`Run_SycophancyRAGBench.ipynb`](../../Run_SycophancyRAGBench.ipynb) | Interactive single-cell Colab/Kaggle runner | Sets `MODELS_TO_RUN`, `SETTINGS_TO_RUN`, `TAKE_ALL_DATA`, `FRESH_RUN`, invokes `run_pipeline(ConfigArgs())` and displays PNG figures |
| [`configs/config.py`](../../configs/config.py) | Central configuration & runtime adapter | `detect_runtime()`, `RUNTIME`, `PROJECT_DIR`, `ALL_DIRS`, `MODELS`, `OPTIONAL_LARGE_MODELS`, `SETTINGS`, `BELIEF_STYLES`, `DOMAINS`, `DENSE_MODEL_ID`, `JUDGE_HF_ID`, `CAUSAL_CONFLICT_CATEGORIES` |
| [`configs/__init__.py`](../../configs/__init__.py) | Package initializer | Exposes `cfg` (`from . import config as cfg`) |
| [`requirements.txt`](../../requirements.txt) | Dependency specification | `torch>=2.2.0`, `transformers>=4.51.0`, `accelerate>=0.30.0`, `bitsandbytes>=0.43.0`, `sentence-transformers>=3.0.0`, `datasets>=2.19.2`, `huggingface_hub>=0.23.0`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `tqdm`, `requests` |
| [`README.md`](../../README.md) | Documentation | Benchmark overview, Colab/Kaggle/Local quickstarts, CLI flag reference, output descriptions |

---

## 3. Source Package (`src/`)

### Core Infrastructure
- [`src/env.py`](../../src/env.py) (250 lines): `is_kaggle()`, `is_colab()`, `get_runtime_name()`, `get_hf_token()`, `authenticate_hf()`, `setup_workspace()`, `_sync_bundled_data()`, `reset_workspace_cache()`, `package_outputs()`.
- [`src/utils.py`](../../src/utils.py) (278 lines): `set_seed()`, `read_jsonl()`, `write_jsonl()`, `save_json()`, `load_json()`, `JsonlAppender`, `done_keys()`, `fever_decode()`, `norm_title()`, `to_clause()`, `split_passages()`, `tag_domain()`, `build_belief_prompt()`, `format_evidence_block()`, `build_rag_user_text()`.
- [`src/data_loader.py`](../../src/data_loader.py) (237 lines): `load_fever_claims()`, `load_truthfulqa_claims()`, `generate_prompts()`, `load_or_build_benchmark()`, `print_data_summary()`.

### Subpackages
- **`src/retrieval/`**:
  - [`base.py`](../../src/retrieval/base.py): `BaseRetriever` abstract base class (`search(query, k=5)`).
  - [`bm25.py`](../../src/retrieval/bm25.py): `_tokenize()`, `PureBM25` (in-memory inverted index BM25Okapi), `PyseriniLuceneBM25`, `build_bm25_retriever()`.
  - [`dense.py`](../../src/retrieval/dense.py): `DenseRetriever` (loads `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`, caches passage tensor to `corpus/dense_embeddings.pt`), `build_dense_retriever()`.
  - [`hybrid.py`](../../src/retrieval/hybrid.py): `HybridRetriever` (fuses sparse + dense top-50 candidates via Reciprocal Rank Fusion with `RRF_K=60`).
  - [`controls.py`](../../src/retrieval/controls.py): `OracleRetriever` (`search_for_claim()` matches `gold_titles` or falls back to `true_answer`), `IrrelevantRetriever` (deterministic random sampling over `distractors`).
  - [`manager.py`](../../src/retrieval/manager.py): `evaluate_retrieval_quality()` (FEVER `Recall@k` and `MRR@10`) and `run_retrieval_pipeline()`.
- **`src/models/`**:
  - [`chat_model.py`](../../src/models/chat_model.py): `get_bnb_config()`, `ChatModel` (handles Gemma 3 `bfloat16` + `AutoProcessor`, Qwen/Llama/Mistral `AutoModelForCausalLM`, left-padded batched greedy `generate()`, and `close()`).
- **`src/evaluation/`**:
  - [`judge.py`](../../src/evaluation/judge.py): `rule_signals()` (`rb_affirm`, `rb_corrects`), `compute_evidence_overlap()`, `format_judge_prompt()`, `parse_judge_output()`, `Judge` (`Qwen/Qwen2.5-7B-Instruct` 4-bit).
  - [`causal_evaluator.py`](../../src/evaluation/causal_evaluator.py): `classify_conflict_resolution()`, `compute_causal_conflict_metrics()`, `print_causal_summary()`.
- **`src/analysis/`**:
  - [`stats.py`](../../src/analysis/stats.py): `_binarize()`, `paired_mcnemar_test()`, `apply_fdr_correction()`, `bootstrap_ci()`, `compute_all_bootstrap_cis()`, `compute_cohen_kappas()`.
  - [`plots.py`](../../src/analysis/plots.py): `setup_plot_style()`, `plot_sycophancy_by_model_and_condition()`, `plot_evidence_user_conflict_breakdown()`, `_df_to_latex()`, `export_latex_tables()`, `generate_all_publication_artifacts()`.

---

## 4. Bundled Data Files (`data/`)

| File | Records / Lines | Schema / Purpose |
| :--- | :--- | :--- |
| [`data/claims.jsonl`](../../data/claims.jsonl) | 300 lines | `{claim_id, source, domain, false_claim, true_answer, question, retrieval_query, gold_titles}` |
| [`data/prompts.jsonl`](../../data/prompts.jsonl) | 1,800 lines | Claims expanded × 6 styles (`uid`, `style`, `prompt_text`, plus claim fields) |
| [`data/gold_passages.jsonl`](../../data/gold_passages.jsonl) | 596 lines | `{id, contents, title}` — chunked ground-truth Wikipedia passages |
| [`data/distractors_sample.jsonl`](../../data/distractors_sample.jsonl) | 2,500 lines | `{id, contents, title}` — sample distractor Wikipedia passages |
| [`data/manual_validation_labeled.csv`](../../data/manual_validation_labeled.csv) | 160 lines | Columns: `ekey, model, setting, style, domain, false_claim, true_answer, question, response, j_truthful, j_endorses, j_corrects, j_halluc, human_truthful, human_endorses, human_corrects, human_halluc` |

---

## 5. Scripts & Notebooks (`scripts/`, `notebooks/`)

- [`scripts/make_zip.py`](../../scripts/make_zip.py): Builds `../sycophancy_rag_bench.zip` excluding `__pycache__`, `.ipynb_checkpoints`, `.git`, `artifacts`, `index`, `.pyc`, `.tmp`, and `.lock`.
- [`scripts/generate_notebooks.py`](../../scripts/generate_notebooks.py): Writes 5 modular notebooks (`00_setup_and_data.ipynb` through `04_stats_and_publication.ipynb`) into `scripts/notebooks/` (note: the root [`notebooks/`](../../notebooks) folder is currently empty, whereas `README.md` lists `notebooks/` as the modular notebook location).
- [`scripts/update_nb.py`](../../scripts/update_nb.py): Programmatically regenerates [`Run_SycophancyRAGBench.ipynb`](../../Run_SycophancyRAGBench.ipynb) (note: both files reference `SAMPLE_LIMIT` in `ConfigArgs` without defining `SAMPLE_LIMIT` in the generated cell).

---

## 6. Runtime Output & `artifacts/` Layout

When `RUNTIME == "local"`, `cfg.PROJECT_DIR` resolves to `artifacts/` and `env.setup_workspace()` provisions all directories in `cfg.ALL_DIRS`:

| Config Constant | Relative Path under `PROJECT_DIR` | Expected Runtime Artifacts | Current Repo State |
| :--- | :--- | :--- | :--- |
| `DATA_DIR` | `data/` | `claims.jsonl`, `prompts.jsonl` | Present (`300` claims, `1,800` prompts) |
| `CORPUS_DIR` / `COLLECTION_DIR` / `INDEX_DIR` | `corpus/`, `corpus/collection/`, `corpus/index/` | `gold.jsonl`, `distractors.jsonl`, `build_state.json`, `dense_embeddings.pt`, Lucene index | Not yet populated |
| `RETRIEVAL_DIR` | `retrieval/` | `retrieval_{bm25,dense,hybrid,oracle,irrelevant}.jsonl`, `retrieval_metrics.json` | Not yet populated |
| `GEN_DIR` | `generations/` | `<model>__<setting>.jsonl` | Not yet populated |
| `EVAL_DIR` | `eval/` | `judged.jsonl`, `causal_conflict_metrics.json`, `manual_validation_labeled.csv`, `metrics.json`, `manual_validation_sample.csv` | Not yet populated |
| `STATS_DIR` | `stats/` | `mcnemar.json`, `bootstrap_ci.json`, `cohen_kappa.json` | Not yet populated |
| `FIG_DIR` | `stats/figures/` | `sycophancy_by_model_and_retriever.{png,pdf}`, `evidence_user_conflict_breakdown.{png,pdf}` | Present (all 4 files) |
| `TABLE_DIR` | `stats/tables/` | `table1_main_results.tex`, `table2_mcnemar_tests.tex` | Present (2 `.tex` files; currently reflect `LLAMA-3.2-3B` `Vanilla` vs `Rag Hybrid` run) |
| `RELEASE_DIR` | `release/` | Release bundle staging directory | Not yet populated |
