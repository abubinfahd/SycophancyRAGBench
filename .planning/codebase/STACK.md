# Technology Stack

**Repository:** `sycophancy_rag_bench` (`SycophancyRAGBench`)
**Target Runtime:** Python 3.10+ (documented in `README.md`), Jupyter Notebook (`nbformat` 4.5), CLI runner (`run_all.py`)

---

## 1. Python Dependencies

### Core Declared Dependencies (`requirements.txt`)

| Category | Package | Constraint | Primary Usage in Codebase |
| :--- | :--- | :--- | :--- |
| Deep Learning & Inference | `torch` | `>=2.2.0` | Model inference (`src/models/chat_model.py`), LLM judge (`src/evaluation/judge.py`), dense cosine similarity & `.pt` caching (`src/retrieval/dense.py`), deterministic seeding (`src/utils.py`) |
| Deep Learning & Inference | `transformers` | `>=4.51.0` | `AutoTokenizer`, `AutoModelForCausalLM`, `BitsAndBytesConfig`, `Gemma3ForConditionalGeneration`, `AutoProcessor` (`src/models/chat_model.py`, `src/evaluation/judge.py`) |
| Deep Learning & Inference | `accelerate` | `>=0.30.0` | Required backend for `device_map="auto"` model placement (`src/models/chat_model.py`, `src/evaluation/judge.py`) |
| Quantization | `bitsandbytes` | `>=0.43.0` | 4-bit NF4 double-quantized inference via `BitsAndBytesConfig` (`src/models/chat_model.py`, `src/evaluation/judge.py`) |
| Dense Embeddings | `sentence-transformers` | `>=3.0.0` | `SentenceTransformer` encoding for `DenseRetriever` (`src/retrieval/dense.py`) |
| Datasets & Hub | `datasets` | `>=2.19.2` | `load_dataset` for FEVER and TruthfulQA (`src/data_loader.py`) |
| Datasets & Hub | `huggingface_hub` | `>=0.23.0` | `huggingface_hub.login` token authentication (`src/env.py`) |
| Data Science & Stats | `pandas` | `>=2.0.0` | DataFrames, pivot tables, CSV reading, LaTeX table export (`run_all.py`, `src/data_loader.py`, `src/evaluation/causal_evaluator.py`, `src/analysis/stats.py`, `src/analysis/plots.py`) |
| Data Science & Stats | `numpy` | `>=1.24.0` | Bootstrap resampling, FDR correction, array math (`src/analysis/stats.py`, `src/analysis/plots.py`, `src/evaluation/causal_evaluator.py`, `src/utils.py`) |
| Data Science & Stats | `scikit-learn` | `>=1.3.0` | `sklearn.metrics.cohen_kappa_score` for judge reliability (`src/analysis/stats.py`) |
| Data Science & Stats | `statsmodels` | `>=0.14.0` | `statsmodels.stats.contingency_tables.mcnemar` for paired significance tests (`src/analysis/stats.py`) |
| Visualization | `matplotlib` | `>=3.7.0` | Non-interactive `Agg` backend 300 DPI PNG and vector PDF figure generation (`src/analysis/plots.py`) |
| Progress & HTTP | `tqdm` | `>=4.66.0` | `tqdm.auto.tqdm` progress bars (`run_all.py`, `src/retrieval/manager.py`) |
| Progress & HTTP | `requests` | `>=2.31.0` | Listed in `requirements.txt` (no direct `.py` imports in `src/` or `run_all.py`) |

> Note: `Run_SycophancyRAGBench.ipynb` installs a notebook subset via `pip install -q -U "transformers>=4.51.0" accelerate bitsandbytes sentence-transformers datasets statsmodels scikit-learn matplotlib tqdm`.

### Optional / Runtime-Conditional Dependencies

- `pyserini` (`pyserini.search.lucene.LuceneSearcher`): Optional sparse retrieval backend in `src/retrieval/bm25.py` (`PyseriniLuceneBM25`), requiring Java 21 and a prebuilt Lucene index at `cfg.INDEX_DIR` when `prefer_lucene=True`.
- `google.colab` (`drive`, `userdata`): Imported lazily inside Colab runtimes in `src/env.py`.
- `kaggle_secrets` (`UserSecretsClient`): Imported lazily inside Kaggle runtimes in `src/env.py`.
- `IPython.display` (`Image`, `display`): Used in `Run_SycophancyRAGBench.ipynb` and `scripts/notebooks/04_stats_and_publication.ipynb`.
- `jinja2` (via `pandas.DataFrame.to_latex`): Used opportunistically in `src/analysis/plots.py:_df_to_latex()`, with a pure-Python LaTeX `tabular` fallback on exception.

---

## 2. Hardware, Precision, and Quantization Stack

- **Device Selection:** Automatic CUDA detection (`"cuda" if torch.cuda.is_available() else "cpu"`) across `ChatModel` (`src/models/chat_model.py`), `Judge` (`src/evaluation/judge.py`), and `DenseRetriever` (`src/retrieval/dense.py`). Multi-GPU/auto placement uses `device_map="auto"` when CUDA is available.
- **Notebook Accelerator Metadata:** `Run_SycophancyRAGBench.ipynb` and `scripts/generate_notebooks.py` declare `"accelerator": "GPU"` and `"colab": {"gpuType": "T4"}`; `README.md` documents T4, L4, A100 (Colab) and GPU T4 x 2 or P100 (Kaggle).
- **4-Bit Quantization (`bitsandbytes`):**
  - Controlled globally by `LOAD_IN_4BIT = True` in `configs/config.py`.
  - Both `ChatModel` (`get_bnb_config`) and `Judge` configure `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` whenever CUDA is available.
- **Numeric Precision (`dtype`):**
  - `ChatModel` (`src/models/chat_model.py`): Uses `torch.bfloat16` if `family == "gemma3"` or `torch.cuda.is_bf16_supported()`, otherwise `torch.float16`. Comment explicitly notes Gemma-3 overflows in `float16` (empty/NaN outputs).
  - `Judge` (`src/evaluation/judge.py`): Hardcodes `torch.float16` for both `bnb_4bit_compute_dtype` and `torch_dtype`.
- **Family-Specific Loading & Attention:**
  - `family == "gemma3"` uses `Gemma3ForConditionalGeneration` + `AutoProcessor` with `attn_implementation="eager"` and multimodal-style user content `[{"type": "text", "text": ...}]`, falling back to `AutoTokenizer` / `AutoModelForCausalLM` on error (`src/models/chat_model.py`).
  - Other families (`causal`, `llama3`) use `AutoTokenizer` + `AutoModelForCausalLM` with standard `system`/`user` message dicts.
- **Inference & Memory Management:**
  - Left-padded batched generation (`padding_side="left"`, `pad_token = eos_token` fallback), greedy decoding (`do_sample=False`).
  - `ChatModel.generate`: `max_length=3072`, `max_new_tokens=cfg.MAX_NEW_TOKENS` (`200`), default `BATCH_SIZE = 8` (`configs/config.py`).
  - `Judge.evaluate_batch`: `max_length=2048`, `max_new_tokens=96`, batch size `16` (`run_all.py`).
  - Explicit GPU cleanup via `close()` (`del self.model`, `del self.tok`, `gc.collect()`, `torch.cuda.empty_cache()`).
- **Determinism:** `src/utils.py:set_seed()` sets `random`, `PYTHONHASHSEED`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `torch.backends.cudnn.deterministic = True` (`SEED = 42` in `configs/config.py`).

---

## 3. Retrieval & Indexing Stack

Configured in `configs/config.py` (`TOP_K = 5`, `RETRIEVAL_KS = [1, 3, 5, 10]`, `PASSAGE_WORDS = 120`, `MAX_PASSAGES_PER_DOC = 6`):

- **Sparse (`src/retrieval/bm25.py`):**
  - Default `PureBM25`: In-memory pure Python BM25Okapi inverted index (`BM25_K1 = 0.9`, `BM25_B = 0.4`), regex alphanumeric tokenizer (`\b[a-z0-9]+\b`). Zero JVM/C++ build dependencies.
  - Optional `PyseriniLuceneBM25`: Wraps `pyserini.search.lucene.LuceneSearcher` (`prefer_lucene=False` by default).
- **Dense (`src/retrieval/dense.py`):**
  - `DenseRetriever` uses `SentenceTransformer` (`batch_size=64`, `normalize_embeddings=True`, `convert_to_tensor=True`) with primary `DENSE_MODEL_ID = "BAAI/bge-small-en-v1.5"` and fallback `DENSE_FALLBACK_ID = "sentence-transformers/all-MiniLM-L6-v2"`.
  - Computes cosine similarity via `torch.mv(self.embeddings, q_emb)` and `torch.topk`.
  - Caches normalized passage embeddings to `<CORPUS_DIR>/dense_embeddings.pt` via `torch.save`/`torch.load`.
- **Hybrid (`src/retrieval/hybrid.py`):**
  - `HybridRetriever` combines `PureBM25` and `DenseRetriever` using Reciprocal Rank Fusion (`RRF_K = 60`, `candidate_depth = 50`).
- **Experimental Controls (`src/retrieval/controls.py`):**
  - `OracleRetriever`: Title-normalized lookup against `gold_passages.jsonl` (falling back to `true_answer` for TruthfulQA claims without `gold_titles`).
  - `IrrelevantRetriever`: Seeded random sampling (`seed=cfg.SEED`) from `distractors.jsonl`.

---

## 4. Evaluation, Statistical & Publication Stack

- **Hybrid Scoring (`src/evaluation/judge.py`):**
  - Rule-based lexical heuristics (`rule_signals`: `rb_affirm`, `rb_corrects`; `compute_evidence_overlap`: stopword-filtered content-word ratio).
  - LLM-as-a-Judge (`Judge`) running `JUDGE_HF_ID = "Qwen/Qwen2.5-7B-Instruct"` in 4-bit NF4, emitting JSON parsed via `re.search(r"\{.*?\}", ...)` into `j_truthful`, `j_endorses`, `j_corrects`, `j_halluc`, `j_parsed`.
- **Causal Conflict Taxonomy (`src/evaluation/causal_evaluator.py`):**
  - Categorizes prompt-response rows into `conflict_evidence_aligned`, `conflict_sycophantic`, `conflict_hedged`, `conflict_hallucinated` (plus `neutral_baseline`, `vanilla_sycophantic`, `vanilla_truthful`) and correlates `evidence_overlap` via Pearson correlation.
- **Statistical Testing (`src/analysis/stats.py`):**
  - Paired McNemar's test (`statsmodels`, exact binomial for discordant pairs `< 25`, continuity-corrected otherwise, with chi-squared fallback).
  - Benjamini-Hochberg FDR multiple-testing correction (`apply_fdr_correction`).
  - Non-parametric 95% bootstrap CIs (`n_boot = 2000`, `np.random.default_rng(42)`).
  - Cohen's Kappa (`sklearn.metrics.cohen_kappa_score`) for Judge-vs-Rule and Judge-vs-Human (`manual_validation_labeled.csv`, 160 lines).
- **Publication Export (`src/analysis/plots.py`):**
  - Matplotlib (`Agg` backend) exports 300 DPI PNG and vector PDF charts (`sycophancy_by_model_and_retriever.*`, `evidence_user_conflict_breakdown.*`).
  - Exports LaTeX tables (`table1_main_results.tex`, `table2_mcnemar_tests.tex`).

---

## 5. Persistence & Packaging Stack

- **Crash-Safe IO (`src/utils.py`):**
  - `JsonlAppender` performs line-by-line `flush()` + `os.fsync()` for preemptible cloud runtimes.
  - `write_jsonl` and `save_json` write to `.tmp` then `os.replace()`.
  - `done_keys` + `read_jsonl` enable checkpoint resumption while tolerating truncated trailing lines.
- **Distribution & Archival Scripts (`scripts/`):**
  - `scripts/make_zip.py`: Builds `sycophancy_rag_bench.zip` excluding `__pycache__`, `.ipynb_checkpoints`, `.git`, `artifacts`, `index`, and `.pyc`/`.tmp`/`.lock` files.
  - `scripts/generate_notebooks.py`: Programmatically writes `scripts/notebooks/00_setup_and_data.ipynb` through `04_stats_and_publication.ipynb`.
  - `scripts/update_nb.py`: Regenerates root `Run_SycophancyRAGBench.ipynb`.
