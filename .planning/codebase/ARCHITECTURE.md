# SycophancyRAGBench Architecture

**Scope:** `E:\CODE\SC\Papers\SycophancyRAGBench\sycophancy_rag_bench`  
**Focus:** System architecture, end-to-end 7-stage pipeline data flow, and module responsibilities.

---

## 1. System Overview

`SycophancyRAGBench` is a self-contained, CLI- and notebook-executable empirical benchmark evaluating whether Retrieval-Augmented Generation (RAG) reduces sycophancy in Large Language Models (LLMs) when user queries assert or imply false beliefs across 6 framing styles (`neutral`, `weak`, `strong`, `authority`, `emotional`, `social`).

The system is organized around a single master orchestrator ([`run_all.py`](../../run_all.py)) backed by a platform-adaptive configuration module ([`configs/config.py`](../../configs/config.py)) and a modular `src/` package covering five functional layers:
1. **Environment & Persistence (`src/env.py`, `src/utils.py`)** — runtime detection (Google Colab, Kaggle Notebooks, Local/Server), Hugging Face Hub authentication, bundled-data bootstrapping, and crash-safe atomic/append-only JSONL I/O.
2. **Benchmark Data Construction (`src/data_loader.py`)** — claim extraction from FEVER and TruthfulQA plus factorial expansion into belief-framed prompts.
3. **Multi-Paradigm Retrieval & Controls (`src/retrieval/`)** — unified `BaseRetriever` abstraction over sparse (`PureBM25`, optional `PyseriniLuceneBM25`), dense (`DenseRetriever`), hybrid Reciprocal Rank Fusion (`HybridRetriever`), and experimental controls (`OracleRetriever`, `IrrelevantRetriever`).
4. **Batched Quantized Inference (`src/models/chat_model.py`)** — unified 4-bit NF4 `ChatModel` wrapper supporting Qwen 2.5, Gemma 3, Llama 3.1/3.2, and Mistral model families.
5. **Evaluation, Causal Analysis & Publication (`src/evaluation/`, `src/analysis/`)** — `Qwen/Qwen2.5-7B-Instruct` 4-bit LLM judge paired with lexical heuristics, 4-way evidence–user conflict taxonomy, McNemar significance testing with Benjamini-Hochberg FDR correction, bootstrap 95% CIs, Cohen's Kappa reliability, and 300 DPI Matplotlib/LaTeX exporters.

---

## 2. End-to-End 7-Stage Pipeline Data Flow

[`run_pipeline(args)`](../../run_all.py#L117-L345) executes seven sequential stages (note: console banners label Steps 1–5 for the core compute/packaging phases, while the underlying control flow comprises 7 distinct stages):

```
[Stage 1: Env & Auth]
  configs/config.py (detect_runtime) -> env.setup_workspace() -> [optional env.reset_workspace_cache()] -> env.authenticate_hf()
        │
        ▼
[Stage 2: Dataset & Prompt Construction]
  data/* or HuggingFace (FEVER + TruthfulQA) -> data_loader.load_or_build_benchmark()
        │  outputs: claims.jsonl (claim_id), prompts.jsonl (uid = claim_id__style)
        ▼
[Stage 3: Multi-Retriever & Controls]  (skipped via --skip_retrieval)
  claims.jsonl + collection/{gold.jsonl, distractors.jsonl} -> retrieval/manager.run_retrieval_pipeline()
        │  outputs: retrieval/retrieval_{bm25,dense,hybrid,oracle,irrelevant}.jsonl, retrieval_metrics.json
        ▼
[Stage 4: Multi-Family Batched Inference]  (skipped via --skip_inference)
  prompts.jsonl + retrieval_*.jsonl -> load_retrieval_evidence_index() -> ChatModel.generate()
        │  outputs: generations/<model>__<setting>.jsonl (keyed by uid)
        ▼
[Stage 5: Automated Evaluation & LLM Judge]  (skipped via --skip_eval)
  generations/*.jsonl -> Judge.evaluate_batch() + rule_signals() + compute_evidence_overlap()
        │  outputs: eval/judged.jsonl (keyed by ekey = model__setting__uid)
        ▼
[Stage 6: Causal & Statistical Analysis]  (skipped via --skip_stats)
  eval/judged.jsonl + manual_validation_labeled.csv -> causal_evaluator + stats.py
        │  outputs: eval/causal_conflict_metrics.json, stats/{mcnemar,bootstrap_ci,cohen_kappa}.json
        ▼
[Stage 7: Publication Artifacts & Packaging]
  plots.generate_all_publication_artifacts() -> [env.package_outputs() if --zip_on_finish or Kaggle]
        │  outputs: stats/figures/*.{png,pdf}, stats/tables/*.tex, sycophancy_rag_bench_outputs.zip
```

### Stage-by-Stage Execution Details

#### Stage 1 — Workspace, Cache Purge & Authentication
- **Entry calls:** `env.setup_workspace()`, optional `env.reset_workspace_cache(clear_data=...)`, `env.authenticate_hf(args.hf_token, interactive=True)` in [`run_all.py`](../../run_all.py#L123-L130).
- **Behavior:**
  - `configs.config.detect_runtime()` sets `RUNTIME` (`"kaggle"`, `"colab"`, or `"local"`) and resolves `PROJECT_DIR` (`/kaggle/working/SycophancyRAGBench`, `/content/drive/MyDrive/SycophancyRAGBench`, or `<repo>/artifacts`).
  - `env.setup_workspace()` mounts Google Drive on Colab (falling back to `/content/SycophancyRAGBench` and rewriting `cfg.*` paths on failure), creates all `cfg.ALL_DIRS`, and calls `_sync_bundled_data()` to copy bundled files from `data/` into active `PROJECT_DIR` paths if absent.
  - `--fresh` / `--no_cache` triggers `env.reset_workspace_cache()`, deleting `GEN_DIR`, `EVAL_DIR`, `RETRIEVAL_DIR`, `STATS_DIR`, `FIG_DIR`, `TABLE_DIR`, `RELEASE_DIR`, `corpus/dense_embeddings.pt`, and (when `--all_data`) `CLAIMS_PATH` and `PROMPTS_PATH`.
  - `env.authenticate_hf()` resolves a token from `args.hf_token`, `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`, Colab `userdata`, Kaggle `UserSecretsClient`, or interactive `getpass`/input, then logs into `huggingface_hub`.

#### Stage 2 — Benchmark Dataset & Factorial Prompt Construction
- **Entry calls:** `data_loader.load_or_build_benchmark(...)` and `data_loader.print_data_summary(prompts)` in [`run_all.py`](../../run_all.py#L132-L140).
- **Behavior:**
  - Checks active workspace (`cfg.CLAIMS_PATH`, `cfg.PROMPTS_PATH`), then bundled `data/` (`cfg.BUNDLED_CLAIMS`, `cfg.BUNDLED_PROMPTS`), mirroring to active workspace.
  - If `force_rebuild=True` (triggered by `--all_data`, which sets `cfg.TAKE_ALL_DATA = True`, `cfg.N_FEVER = None`, `cfg.N_TRUTHFULQA = None`) or files are missing, builds from scratch via:
    - `load_fever_claims()`: loads `"REFUTES"` claims from `copenlu/fever_gold_evidence` (`validation` split) with fallback to `fever/fever` (`v1.0`, `labelled_dev`), producing `claim_id = "fever_<id>"`.
    - `load_truthfulqa_claims()`: loads misconceptions from `truthfulqa/truthful_qa` (`generation`, `validation` split), producing `claim_id = "tqa_<idx:03d>"`.
    - `generate_prompts()`: expands each claim across `cfg.BELIEF_STYLES` (`neutral`, `weak`, `strong`, `authority`, `emotional`, `social`) using `U.build_belief_prompt()`, generating `uid = "<claim_id>__<style>"` (default scale: 300 claims × 6 styles = 1,800 prompts).
  - Applies `--sample_limit N` when set, slicing `claims[:sample_limit]` and filtering `prompts` accordingly.

#### Stage 3 — Multi-Retriever & Control Pipeline (`STEP 1`)
- **Entry calls:** `retrieval.manager.run_retrieval_pipeline(claims, retrievers_to_run=needed_retrievers, k=cfg.TOP_K)` in [`run_all.py`](../../run_all.py#L143-L157).
- **Condition-to-retriever mapping:** `rag_bm25 -> bm25`, `rag_dense -> dense`, `rag_hybrid -> hybrid`, `control_oracle -> oracle`, `control_irrelevant -> irrelevant` (`vanilla` requires no retrieval).
- **Behavior:**
  - Loads `gold_passages` (`cfg.GOLD_PASSAGES_PATH` or `cfg.BUNDLED_GOLD`) and `distractors` (`cfg.DISTRACTORS_PATH` or `cfg.BUNDLED_DISTRACTORS`).
  - Lazily instantiates only needed retrievers:
    - `bm25`: `build_bm25_retriever()` → `PureBM25` (`k1=0.9`, `b=0.4`) or optional `PyseriniLuceneBM25` if `prefer_lucene=True`.
    - `dense`: `build_dense_retriever()` → `DenseRetriever` using `BAAI/bge-small-en-v1.5` (fallback `sentence-transformers/all-MiniLM-L6-v2`) with tensor cache at `corpus/dense_embeddings.pt`.
    - `hybrid`: `HybridRetriever(bm25, dense, rrf_k=60, candidate_depth=50)` combining top-50 sparse & dense lists via Reciprocal Rank Fusion.
    - `oracle`: `OracleRetriever(gold_passages)` matching `U.norm_title()` against `gold_titles` for FEVER, falling back to `claim["true_answer"]` as `"Verified Fact"` for TruthfulQA.
    - `irrelevant`: `IrrelevantRetriever(distractors, seed=cfg.SEED)` sampling random unrelated distractor passages.
  - Resumes per retriever via `U.done_keys(out_path, "claim_id")` and writes up to `k_max = max(max(cfg.RETRIEVAL_KS), k)` hits per claim (`TOP_K=5`, `RETRIEVAL_KS=[1,3,5,10]`, so `k_max=10`) to `retrieval/retrieval_<method>.jsonl`.
  - Evaluates FEVER retrieval quality (`evaluate_retrieval_quality()`) for `Recall@1/3/5/10` and `MRR@10`, persisting aggregated metrics to `retrieval/retrieval_metrics.json`.

#### Stage 4 — Multi-Family Batched Inference (`STEP 2`)
- **Entry calls:** `load_retrieval_evidence_index()`, `ChatModel(spec)`, `get_user_prompt_and_evidence()`, `cm.generate()` in [`run_all.py`](../../run_all.py#L162-L235).
- **Behavior:**
  - Loads retrieval files for requested settings into `{setting: {claim_id: row}}`.
  - Matches `--models` against `cfg.MODELS + cfg.OPTIONAL_LARGE_MODELS`.
  - Iterates model-by-model (loading one model onto GPU at a time, skipping load if all settings have zero pending `uid`s via `U.done_keys(gen_file, "uid")`).
  - `ChatModel` (`src/models/chat_model.py`) loads `Gemma3ForConditionalGeneration` + `AutoProcessor` (`bfloat16`, `attn_implementation="eager"`) for `family == "gemma3"`, or `AutoModelForCausalLM` + `AutoTokenizer` (`bfloat16` if supported else `float16`) for `causal` / `llama3` families, using 4-bit NF4 double quantization (`BitsAndBytesConfig`) when CUDA is available and `LOAD_IN_4BIT=True`.
  - `get_user_prompt_and_evidence()` formats top-5 (`cfg.TOP_K`) hits using `U.format_evidence_block()` and `U.build_rag_user_text()` (`"Reference material:\n..."` + neutral instruction + `prompt_text`), or passes raw `prompt_text` for `vanilla`.
  - Runs greedy decoding (`do_sample=False`, `max_new_tokens=200`, `max_length=3072`, `padding_side="left"`) in batches of `args.batch_size` (default `8`) and streams records to `generations/<model>__<setting>.jsonl`. Calls `cm.close()` (`gc.collect()` + `torch.cuda.empty_cache()`) before the next model.

#### Stage 5 — Automated Evaluation & LLM Judge (`STEP 3`)
- **Entry calls:** `Judge()`, `judge.evaluate_batch()`, `rule_signals()`, `compute_evidence_overlap()` in [`run_all.py`](../../run_all.py#L239-L285).
- **Behavior:**
  - Aggregates all `generations/<model>__<setting>.jsonl` records for requested `--models` × `--settings`, assigning `ekey = f"{model}__{setting}__{uid}"` and deduplicating by `ekey`.
  - Filters out rows already present in `eval/judged.jsonl` via `U.done_keys(cfg.JUDGED_PATH, "ekey")`.
  - Loads `Judge` (`Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 `float16`) and scores batches of 16 (`max_new_tokens=96`, `do_sample=False`).
  - `parse_judge_output()` extracts JSON fields `{truthful, endorses_false_belief, corrects_user, hallucination}` into `{j_truthful, j_endorses, j_corrects, j_halluc, j_parsed}`.
  - Enriches each record with `rule_signals()` (`rb_affirm`, `rb_corrects`) and `compute_evidence_overlap()` (`evidence_overlap` content-word overlap ratio when `evidence_text` is non-empty), appending atomically to `eval/judged.jsonl`.

#### Stage 6 — Causal Mechanism & Statistical Testing (`STEP 4`)
- **Entry calls:** `compute_causal_conflict_metrics()`, `paired_mcnemar_test()`, `apply_fdr_correction()`, `compute_all_bootstrap_cis()`, `compute_cohen_kappas()` in [`run_all.py`](../../run_all.py#L288-L325).
- **Behavior:**
  - `causal_evaluator.compute_causal_conflict_metrics()` applies `classify_conflict_resolution()` to assign each row into `neutral_baseline`, `vanilla_sycophantic`, `vanilla_truthful`, `conflict_sycophantic`, `conflict_evidence_aligned`, `conflict_hallucinated`, or `conflict_hedged`; computes setting-level distributions, `style × setting` sycophancy pivots, and Pearson correlations between `evidence_overlap` and sycophancy/truthfulness; saves `eval/causal_conflict_metrics.json`.
  - `stats.paired_mcnemar_test()` pairs each non-vanilla setting against `vanilla` per model on `j_endorses` (`belief_only=True`) and `j_truthful` (`belief_only=False`), using exact binomial test when discordant pairs `< 25` and continuity-corrected chi-square otherwise; `apply_fdr_correction()` computes Benjamini-Hochberg adjusted `pvalue_fdr` and `significant_005`; saves `stats/mcnemar.json`.
  - `stats.compute_all_bootstrap_cis()` computes 2,000-resample 95% bootstrap CIs per `(model, setting)` for sycophancy, truthfulness, and correction rate; saves `stats/bootstrap_ci.json`.
  - `stats.compute_cohen_kappas()` computes inter-rater agreement (`judge_vs_rule_endorses`, plus `judge_vs_human_endorses` and `judge_vs_human_truthfulness` by merging `eval/judged.jsonl` with `eval/manual_validation_labeled.csv` on `ekey`); saves `stats/cohen_kappa.json`.

#### Stage 7 — Publication Figures, LaTeX Tables & Output Packaging (`STEP 4–5`)
- **Entry calls:** `generate_all_publication_artifacts()` and `env.package_outputs()` in [`run_all.py`](../../run_all.py#L327-L341).
- **Behavior:**
  - `plots.generate_all_publication_artifacts()` renders 300 DPI PNG and vector PDF charts (`sycophancy_by_model_and_retriever.{png,pdf}`, `evidence_user_conflict_breakdown.{png,pdf}`) into `stats/figures/` and exports `table1_main_results.tex` and `table2_mcnemar_tests.tex` into `stats/tables/`.
  - If `--zip_on_finish` is passed or `env.is_kaggle()` is true, `env.package_outputs()` bundles `cfg.PROJECT_DIR` (excluding `"index"` directories) into `cfg.OUTPUT_ZIP_PATH` (`sycophancy_rag_bench_outputs.zip`).

---

## 3. Module Responsibilities & Dependency Map

| Layer | Module | Primary Responsibility | Key Imports / Dependencies |
| :--- | :--- | :--- | :--- |
| **Orchestration** | [`run_all.py`](../../run_all.py) | CLI parsing (`parse_args`) and 7-stage execution (`run_pipeline`) | `configs.config`, `src.env`, `src.utils`, `src.data_loader`, `src.retrieval.manager`, `src.models.chat_model`, `src.evaluation.*`, `src.analysis.*` |
| **Config** | [`configs/config.py`](../../configs/config.py) | Runtime path resolution, model specs, retrieval/judge/generation hyperparameters | `os`, `sys` (zero third-party deps) |
| **Platform** | [`src/env.py`](../../src/env.py) | Colab/Kaggle/Local setup, Drive mounting, HF auth, cache reset, zip export | `configs.config`, `huggingface_hub`, `google.colab` (optional), `kaggle_secrets` (optional) |
| **Utilities** | [`src/utils.py`](../../src/utils.py) | Deterministic seeding, atomic JSON/JSONL + `JsonlAppender`, FEVER decoding, domain tagging, prompt & RAG formatting | `json`, `os`, `re`, `numpy` (opt), `torch` (opt) |
| **Data** | [`src/data_loader.py`](../../src/data_loader.py) | FEVER & TruthfulQA loading, factorial prompt generation, smoke-test subsampling | `configs.config`, `src.utils`, `datasets`, `pandas` |
| **Retrieval** | [`src/retrieval/`](../../src/retrieval/__init__.py) | `BaseRetriever` interface, `PureBM25`/`PyseriniLuceneBM25`, `DenseRetriever`, `HybridRetriever` (RRF), `OracleRetriever`, `IrrelevantRetriever`, `run_retrieval_pipeline`, `evaluate_retrieval_quality` | `configs.config`, `src.utils`, `torch`, `sentence_transformers`, `pyserini` (opt) |
| **Models** | [`src/models/chat_model.py`](../../src/models/chat_model.py) | Multi-family 4-bit `ChatModel` loader, chat-template adapter, batched greedy generation, GPU cleanup | `configs.config`, `torch`, `transformers`, `bitsandbytes` |
| **Evaluation** | [`src/evaluation/judge.py`](../../src/evaluation/judge.py), [`causal_evaluator.py`](../../src/evaluation/causal_evaluator.py) | Lexical rules, evidence overlap, 4-bit `Qwen2.5-7B-Instruct` JSON judge, 4-way causal conflict classification & correlation | `configs.config`, `src.utils`, `torch`, `transformers`, `pandas`, `numpy` |
| **Analysis** | [`src/analysis/stats.py`](../../src/analysis/stats.py), [`plots.py`](../../src/analysis/plots.py) | Paired McNemar + BH FDR, 2000× bootstrap CIs, Cohen's Kappa, 300 DPI PNG/PDF charts, LaTeX tables | `configs.config`, `src.utils`, `statsmodels`, `scikit-learn`, `matplotlib`, `pandas`, `numpy` |
| **Distribution** | [`scripts/`](../../scripts/) | Zip packager (`make_zip.py`), modular notebook generator (`generate_notebooks.py`), master notebook updater (`update_nb.py`) | `json`, `zipfile` |

---

## 4. Architectural Patterns & Cross-Cutting Design

- **Resumable Key-Indexed Checkpointing:** Every expensive stage (`retrieval`, `inference`, `eval`) tracks completed IDs (`claim_id`, `uid`, `ekey`) via `U.done_keys()` and streams outputs via `U.JsonlAppender` (`flush()` + `os.fsync()` per record). `U.read_jsonl()` silently ignores half-written trailing lines after preemptible cloud disconnects.
- **Sequential GPU Memory Management:** Inference loads one `ChatModel` at a time and calls `cm.close()` (`del model, tok`, `gc.collect()`, `torch.cuda.empty_cache()`) before loading the next model or the `Judge` model in Stage 5, keeping peak VRAM within single-GPU Colab/Kaggle T4 limits via 4-bit NF4 double quantization.
- **Family-Specific Template & Precision Handling:** `ChatModel` handles Gemma 3's `fp16` overflow hazard by forcing `bfloat16` and using `Gemma3ForConditionalGeneration` + `AutoProcessor` with user-only multimodal content blocks (`[{"type": "text", "text": ...}]`), whereas `causal` and `llama3` families use standard `system` + `user` roles with `AutoModelForCausalLM`.
- **Optional Dependency Resilience:** `src/retrieval/__init__.py` guards `DenseRetriever`/`HybridRetriever` imports if `torch`/`sentence-transformers` are absent; `src/analysis/stats.py` falls back to manual continuity-corrected chi-square if `statsmodels` is absent; `src/analysis/plots.py` uses `Agg` backend and falls back to manual LaTeX tabular formatting in `_df_to_latex()` if `jinja2` is missing.
