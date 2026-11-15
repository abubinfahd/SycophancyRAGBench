# Coding Conventions & Quality Practices

**Scope:** `E:\CODE\SC\Papers\SycophancyRAGBench\sycophancy_rag_bench` (`src/*`, `configs/config.py`, `run_all.py`, `scripts/*`).

---

## 1. Formatting, Linting & Packaging Standards

- **No linter or formatter configuration exists:** The repository contains no `pyproject.toml`, `setup.cfg`, `.flake8`, `ruff.toml`, `.isort.cfg`, or `.pre-commit-config.yaml`. Code style is enforced informally by convention.
- **Formatting conventions observed:**
  - 4-space indentation, double-quoted strings by default (single quotes occasionally inside notebook strings/f-strings).
  - Top-of-file triple-quoted module docstrings across `configs/config.py`, `run_all.py`, all `src/**/*.py` modules, `scripts/generate_notebooks.py`, and `scripts/make_zip.py`.
  - Visual section dividers using `# ==============================================================================` (`configs/config.py`, `src/utils.py`, `src/evaluation/judge.py`) and `# ------------------------------------------------------------------------------` (`scripts/generate_notebooks.py`).
- **Dependency specification:** `requirements.txt` pins minimum versions (`>=`) across 14 packages (e.g., `torch>=2.2.0`, `transformers>=4.51.0`, `bitsandbytes>=0.43.0`, `statsmodels>=0.14.0`, `scikit-learn>=1.3.0`). No exact lockfile or test/dev dependencies are declared.
- **Distribution packaging:** `scripts/make_zip.py` builds `sycophancy_rag_bench.zip` in the parent directory while pruning `EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "artifacts", "index"}` and `EXCLUDE_EXTS = {".pyc", ".tmp", ".lock"}`.

---

## 2. Module Layout, Imports & Naming Conventions

- **Import path bootstrapping:** Entrypoints and standalone scripts prepend the repository root to `sys.path` before importing internal packages:
  - `run_all.py:24-26`: `_pkg_root = os.path.dirname(os.path.abspath(__file__))` inserted at `sys.path[0]`.
  - `src/env.py:13-15`: same pattern relative to `src/`.
  - `scripts/generate_notebooks.py:39-41,74-76,115-117,150-152,184-186`: checks `os.path.exists("../configs")`.
- **Canonical import aliases:**
  - `from configs import config as cfg` (used consistently across `run_all.py`, `src/env.py`, `src/data_loader.py`, `src/models/chat_model.py`, `src/evaluation/judge.py`, `src/evaluation/causal_evaluator.py`, `src/analysis/stats.py`, `src/analysis/plots.py`, and `src/retrieval/*`).
  - `from src import utils as U` (used across `run_all.py`, `src/data_loader.py`, `src/evaluation/causal_evaluator.py`, `src/analysis/*`, `src/retrieval/*`).
- **Package exports:** Subpackages expose curated public APIs via `__all__` in `configs/__init__.py`, `src/models/__init__.py`, `src/evaluation/__init__.py`, `src/analysis/__init__.py`, and `src/retrieval/__init__.py`. (`src/` itself has no `__init__.py`.)
- **Identifier conventions:**
  - `UPPER_SNAKE_CASE` for configuration constants (`configs/config.py`) and module-level lookup tables/prompts (`AFFIRM_PATTERNS`, `CORRECT_PATTERNS`, `STOP_WORDS`, `JUDGE_SYSTEM`, `FEVER_TOKENS`, `DOMAIN_KEYWORDS`, `PALETTE`, `EXCLUDE_DIRS`, `EXCLUDE_EXTS`).
  - `PascalCase` for classes (`BaseRetriever`, `PureBM25`, `PyseriniLuceneBM25`, `DenseRetriever`, `HybridRetriever`, `OracleRetriever`, `IrrelevantRetriever`, `ChatModel`, `Judge`, `JsonlAppender`).
  - `snake_case` for functions/methods, with leading underscores for internal helpers (`_safe_no_grad`, `_binarize`, `_tokenize`, `_df_to_latex`, `_sync_bundled_data`, `_format_messages`, `_build_prompt_str`, `_init_model`, `_build_or_load_embeddings`, `_build_index`).
- **Type annotations:** Standard `typing` annotations (`List`, `Dict`, `Any`, `Optional`, `Set`) appear on most function signatures. Two functions use invalid Python 3 tuple return syntax instead of `Tuple[...]`:
  - `run_all.py:101`: `def get_user_prompt_and_evidence(...) -> (str, str, bool):`
  - `src/data_loader.py:169`: `def load_or_build_benchmark(...) -> (List[Dict[str, Any]], List[Dict[str, Any]]):`

---

## 3. Configuration & Multi-Platform Runtime Adaptation

- **Dynamic environment detection (`configs/config.py:12-38`):** `detect_runtime()` inspects `KAGGLE_KERNEL_RUN_TYPE` / `/kaggle/working` (`"kaggle"`), `google.colab` in `sys.modules` / `/content` (`"colab"`), and defaults to `"local"` (writing artifacts under `./artifacts`).
- **Slight Colab/Kaggle check inconsistency:** `src/env.py:25-29` guards `is_colab()` with `if is_kaggle(): return False` before checking `/content`, whereas `configs/config.py:14-18` relies solely on evaluating the Kaggle branch before the Colab branch.
- **Runtime mutation of `cfg`:**
  - `src/env.py:139-146`: When Google Drive mounting fails on Colab, `setup_workspace()` falls back to `/content/SycophancyRAGBench` and mutates `cfg.PROJECT_DIR` and 11 subdirectory attributes on `cfg` in place.
  - `run_all.py:134-136`: Passing `--all_data` mutates `cfg.TAKE_ALL_DATA = True`, `cfg.N_FEVER = None`, `cfg.N_TRUTHFULQA = None` before calling `load_or_build_benchmark(force_rebuild=True, ...)`.
- **Bundled data syncing:** `src/env.py:159-172` (`_sync_bundled_data()`) copies bundled datasets from `data/` (`claims.jsonl`, `prompts.jsonl`, `gold_passages.jsonl`, `distractors_sample.jsonl`, `manual_validation_labeled.csv`) into the active `cfg.PROJECT_DIR` workspace if destination files do not already exist.

---

## 4. Logging & Progress Reporting

- **No `logging` module usage:** Python's standard `logging` library is not imported anywhere in the repository (`Select-String` confirmed zero matches across all `.py` files).
- **Structured `print()` prefixes:** All runtime telemetry goes to stdout via `print()` with bracketed component tags:
  - `[Workspace]`, `[Auth]`, `[Fresh Run]`, `[Packaging]` (`src/env.py`)
  - `[Data]` (`src/data_loader.py`, `run_all.py`)
  - `[Retrieval]`, `[BM25]`, `[Dense]` (`src/retrieval/*`)
  - `[Model]` (`src/models/chat_model.py`)
  - `[Inference]`, `[Eval]`, `[Success]` (`run_all.py`)
  - `[Judge]` (`src/evaluation/judge.py`)
  - `[Stats]`, `[Plots]` (`src/analysis/stats.py`, `src/analysis/plots.py`)
- **Progress bars:** Long loops use `tqdm.auto.tqdm` (`run_all.py:208,265`, `src/retrieval/manager.py:125`, `DenseRetriever` `show_progress_bar=True` at `src/retrieval/dense.py:80`).

---

## 5. Reproducibility, Caching & Resumability Guarantees

- **Global seeding helper (`src/utils.py:21-37`):** `set_seed(seed: int = 42)` sets `random.seed(seed)`, `os.environ["PYTHONHASHSEED"]`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, and `torch.backends.cudnn.deterministic = True`.
  - **Gap in CLI runner:** `run_all.py` never calls `U.set_seed(cfg.SEED)`. Only `scripts/notebooks/00_setup_and_data.ipynb` (`scripts/generate_notebooks.py:48`) calls it. Instead, stochastic components accept explicit local seeds: `load_fever_claims(..., seed=cfg.SEED)`, `load_truthfulqa_claims(..., seed=cfg.SEED + 1)`, `IrrelevantRetriever(..., seed=cfg.SEED)`, and `bootstrap_ci(..., seed=42)` (via `np.random.default_rng(seed)`).
- **Deterministic decoding:** Both `ChatModel.generate()` (`src/models/chat_model.py:152`) and `Judge.evaluate_batch()` (`src/evaluation/judge.py:188`) enforce greedy decoding (`do_sample=False`).
- **Atomic & crash-safe IO (`src/utils.py:43-128`):**
  - `write_jsonl()` and `save_json()` write to `<path>.tmp`, call `f.flush()` and `os.fsync(f.fileno())`, then swap via `os.replace(tmp_path, path)`.
  - `JsonlAppender` opens in `"a"` mode and executes `flush()` + `os.fsync()` on every `write()` call to prevent data loss on preemptible Colab/Kaggle runtimes.
  - `read_jsonl()` skips blank lines and silently ignores `json.JSONDecodeError` (`src/utils.py:55-56`) to tolerate a half-written trailing line after interruption.
  - `done_keys(path, key_field)` builds a set of completed IDs for resumable execution and skips entries with empty/whitespace `"response"` fields (`src/utils.py:125-126`).
- **Cache invalidation:** Passing `--fresh` or `--no_cache` to `run_all.py:126-127` triggers `env.reset_workspace_cache()` (`src/env.py:175-220`), deleting outputs in `GEN_DIR`, `EVAL_DIR`, `RETRIEVAL_DIR`, `STATS_DIR`, `FIG_DIR`, `TABLE_DIR`, `RELEASE_DIR`, and `corpus/dense_embeddings.pt` (plus `CLAIMS_PATH`/`PROMPTS_PATH` if `clear_data=True`).

---

## 6. Error Handling & Known Quality Issues

- **Optional dependency isolation:** Heavy ML/stats dependencies (`torch`, `transformers`, `sentence_transformers`, `statsmodels`, `sklearn`) are imported inside `try ... except (ImportError, ModuleNotFoundError):` blocks in `src/models/chat_model.py:15-22`, `src/evaluation/judge.py:13-20`, `src/retrieval/dense.py:13-16`, `src/retrieval/__init__.py:9-15`, and `src/analysis/stats.py:12-20`. Missing required libraries fail explicitly on class instantiation (`raise ImportError(...)` in `ChatModel.__init__`, `Judge.__init__`, `DenseRetriever.__init__`).
- **Multi-tier fallback chains:**
  - **Datasets (`src/data_loader.py:29-54,97-100,206-212`):** Falls back from `copenlu/fever_gold_evidence` to `fever/fever`; from `truthfulqa/truthful_qa` to `truthful_qa`; and from scratch rebuild errors back to `BUNDLED_CLAIMS`/`BUNDLED_PROMPTS`.
  - **Retrieval (`src/retrieval/bm25.py:140-144`, `src/retrieval/dense.py:51-57`):** Falls back from `PyseriniLuceneBM25` to `PureBM25`; from `BAAI/bge-small-en-v1.5` to `sentence-transformers/all-MiniLM-L6-v2`; `PureBM25.search()` returns the first `k` corpus passages with `score: 0.0` if no query tokens match.
  - **Model loading (`src/models/chat_model.py:58-89`):** Forces `torch.bfloat16` for `gemma3` (comment notes fp16 causes empty/NaN outputs) and falls back from `Gemma3ForConditionalGeneration`/`AutoProcessor` to `AutoModelForCausalLM`/`AutoTokenizer`.
  - **Judge & stats (`src/evaluation/judge.py:97-120`, `src/analysis/stats.py:88-96`, `src/analysis/plots.py:158-171`):** `parse_judge_output()` regex-extracts `{...}` and defaults invalid fields to `"incorrect"`/`"no"` while recording `"j_parsed": bool(match)`; `paired_mcnemar_test()` falls back to a continuity-corrected chi-square statistic with binary step p-value (`0.05 if stat > 3.841 else 0.5`) if `statsmodels` is absent or fails; `_df_to_latex()` manually formats a LaTeX `tabular` environment if `df.to_latex()` fails (e.g., missing `jinja2`).
- **Silent exception swallowing:** Cleanup, cache removal, secret lookup, and JSON parsing blocks frequently catch bare `Exception` with `pass` (`src/utils.py:109`, `src/data_loader.py:192`, `src/env.py:61,75,91,96,204,213`, `src/models/chat_model.py:165`, `src/evaluation/judge.py:104,200`).
- **Observed defects:**
  1. **Undefined variable in notebook runner:** `scripts/update_nb.py:119` and `Run_SycophancyRAGBench.ipynb:98` set `sample_limit = SAMPLE_LIMIT` without defining `SAMPLE_LIMIT` anywhere in the cell (`NameError` at runtime).
  2. **Hardcoded local path in script:** `scripts/update_nb.py:4` hardcodes `nb_path = "e:/CODE/SC/Papers/SycophancyRAGBench/sycophancy_rag_bench/Run_SycophancyRAGBench.ipynb"`.
  3. **Empty-series key mismatch in `bootstrap_ci`:** `src/analysis/stats.py:138` returns `{"mean": None, "lo": None, "hi": None, "n": 0}` on empty input, whereas lines `142-147` and callers in `src/analysis/plots.py:75-76,195-197` read `"ci_lower"` and `"ci_upper"`.
