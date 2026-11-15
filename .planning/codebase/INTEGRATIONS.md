# External Integrations & Runtime Platforms

**Repository:** `sycophancy_rag_bench` (`SycophancyRAGBench`)

---

## 1. Runtime Environment Detection & Storage Backends

Environment detection and storage routing are implemented in `configs/config.py` (`detect_runtime()`) and `src/env.py` (`is_kaggle()`, `is_colab()`, `setup_workspace()`).

| Platform | Detection Condition | Workspace Root (`PROJECT_DIR`) | Output Archive (`OUTPUT_ZIP_PATH`) | Persistence Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Kaggle Notebooks** | `"KAGGLE_KERNEL_RUN_TYPE" in os.environ` or `/kaggle/working` exists | `/kaggle/working/SycophancyRAGBench` | `/kaggle/working/sycophancy_rag_bench_outputs.zip` | Auto-zipped on finish (`should_zip = args.zip_on_finish or env.is_kaggle()` in `run_all.py`) |
| **Google Colab** | `"google.colab" in sys.modules` or `/content` exists (checked after Kaggle) | `/content/drive/MyDrive/SycophancyRAGBench` (fallback: `/content/SycophancyRAGBench`) | `/content/sycophancy_rag_bench_outputs.zip` | Mounts Google Drive via `google.colab.drive.mount("/content/drive")` |
| **Local / Server** | Default fallback | `<repo_root>/artifacts` | `<repo_root>/sycophancy_rag_bench_outputs.zip` | Local filesystem; optional zip via `--zip_on_finish` |

### Platform-Specific Behaviors (`src/env.py`)

- **Google Colab Drive Mount & Fallback:** `setup_workspace(mount_google_drive=True)` mounts `/content/drive` if `/content/drive/MyDrive` is absent. If mounting fails, `PROJECT_DIR` and all 11 subdirectory constants (`DATA_DIR`, `CORPUS_DIR`, `COLLECTION_DIR`, `INDEX_DIR`, `RETRIEVAL_DIR`, `GEN_DIR`, `EVAL_DIR`, `STATS_DIR`, `FIG_DIR`, `TABLE_DIR`, `RELEASE_DIR`) are dynamically rewritten from `DRIVE_ROOT` to `/content/SycophancyRAGBench`.
- **Bundled Data Synchronization:** `_sync_bundled_data()` copies pre-bundled files from `data/` into the active `PROJECT_DIR` when missing:
  - `data/claims.jsonl` (300 lines) -> `<DATA_DIR>/claims.jsonl`
  - `data/prompts.jsonl` (1,800 lines) -> `<DATA_DIR>/prompts.jsonl`
  - `data/gold_passages.jsonl` (596 lines) -> `<COLLECTION_DIR>/gold.jsonl`
  - `data/distractors_sample.jsonl` (2,500 lines) -> `<COLLECTION_DIR>/distractors.jsonl`
  - `data/manual_validation_labeled.csv` (160 lines) -> `<EVAL_DIR>/manual_validation_labeled.csv`
- **Kaggle Input & Zip Extraction (`Run_SycophancyRAGBench.ipynb`):** Checks `/kaggle/input/sycophancy-rag-bench/sycophancy_rag_bench.zip` or `/kaggle/working/sycophancy_rag_bench.zip` on Kaggle, and `/content/sycophancy_rag_bench.zip` on Colab, extracting with `!unzip -q -o`.
- **Output Packaging (`package_outputs()`):** Writes deflated ZIP archive via `zipfile.ZipFile` (`exclude_large_index=True` skips directories whose path contains `"index"`).

---

## 2. Hugging Face Hub Authentication & Model Registry

### Authentication Chain (`src/env.py:get_hf_token` & `authenticate_hf`)

Tokens are resolved in the following order before calling `huggingface_hub.login(token=tok, add_to_git_credential=False)`:
1. Explicit CLI/API argument (`--hf_token` in `run_all.py` / `HF_TOKEN` in `Run_SycophancyRAGBench.ipynb`).
2. Environment variables: `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.
3. Google Colab Secrets: `google.colab.userdata.get("HF_TOKEN")`.
4. Kaggle Secrets: `kaggle_secrets.UserSecretsClient().get_secret(...)` checking keys `["HF_TOKEN", "hf_token", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF", "huggingface"]`.
5. Interactive prompt (`getpass.getpass` with fallback to `input`).

If no token is provided, non-gated models still download while gated models fail unless cached.

### Hugging Face Model IDs (`configs/config.py`)

| Role | Config Name | Hugging Face Model ID | Family | Gated | Module |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Primary Eval | `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | `causal` | No | `src/models/chat_model.py` |
| Primary Eval | `gemma-3-4b` | `google/gemma-3-4b-it` | `gemma3` | Yes | `src/models/chat_model.py` |
| Primary Eval | `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | `llama3` | Yes | `src/models/chat_model.py` |
| Optional Large | `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | `llama3` | Yes | `src/models/chat_model.py` |
| Optional Large | `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | `causal` | No | `src/models/chat_model.py` |
| Optional Large | `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | `causal` | Yes | `src/models/chat_model.py` |
| LLM Judge | `JUDGE_HF_ID` | `Qwen/Qwen2.5-7B-Instruct` | `causal` | No | `src/evaluation/judge.py` |
| Dense Retriever | `DENSE_MODEL_ID` | `BAAI/bge-small-en-v1.5` | — | No | `src/retrieval/dense.py` |
| Dense Fallback | `DENSE_FALLBACK_ID` | `sentence-transformers/all-MiniLM-L6-v2` | — | No | `src/retrieval/dense.py` |

---

## 3. Hugging Face Datasets Integrations

When rebuilding datasets from scratch (`--all_data`, `force_rebuild=True`, or missing bundled/workspace JSONL files), `src/data_loader.py` integrates with Hugging Face `datasets.load_dataset`:

| Benchmark Source | Primary Dataset ID | Split / Config | Fallback Dataset ID | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **FEVER** | `copenlu/fever_gold_evidence` | `split="validation"` | `fever/fever` (`"v1.0"`, `split="labelled_dev"`) | Extracts `REFUTES` claims (`N_FEVER = 200`, or all if `TAKE_ALL_DATA = True`) and `gold_titles` |
| **TruthfulQA** | `truthfulqa/truthful_qa` | `"generation"`, `split="validation"` | `truthful_qa` (`"generation"`, `split="validation"`) | Extracts misconceptions (`N_TRUTHFULQA = 100`, or all 817 if `TAKE_ALL_DATA = True`) |
| **Wikipedia** *(config only)* | `wikimedia/wikipedia` | `WIKI_CONFIG = "20231101.en"` | — | Defined in `configs/config.py` (`N_DISTRACTORS = 50000`); not currently invoked by `src/` (distractors come from `data/distractors_sample.jsonl`) |

---

## 4. Optional External Search Engine Integration

- **Pyserini / Apache Lucene (`src/retrieval/bm25.py`):**
  - `PyseriniLuceneBM25` wraps `pyserini.search.lucene.LuceneSearcher(index_dir)` with `set_bm25(0.9, 0.4)`.
  - Requires Java 21 and a prebuilt Lucene index in `cfg.INDEX_DIR`, enabled only when `build_bm25_retriever(..., prefer_lucene=True)`.
  - Default pipeline (`src/retrieval/manager.py`) calls `build_bm25_retriever()` with `prefer_lucene=False`, using pure-Python `PureBM25` instead.

---

## 5. Notebook & Publication Export Integrations

- **Interactive Notebooks:**
  - `Run_SycophancyRAGBench.ipynb`: Single-cell Colab/Kaggle launcher invoking `run_all.run_pipeline()` and rendering figures inline via `IPython.display`.
  - `scripts/notebooks/00_setup_and_data.ipynb` through `04_stats_and_publication.ipynb`: Modular stage-by-stage notebooks generated by `scripts/generate_notebooks.py`.
- **LaTeX & Manuscript Outputs (`src/analysis/plots.py`):**
  - Writes `table1_main_results.tex` and `table2_mcnemar_tests.tex` to `TABLE_DIR` (`stats/tables/`).
  - Writes 300 DPI `.png` and vector `.pdf` figures (`sycophancy_by_model_and_retriever.*`, `evidence_user_conflict_breakdown.*`) to `FIG_DIR` (`stats/figures/`).
