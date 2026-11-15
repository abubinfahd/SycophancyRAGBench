# Codebase Concerns: `sycophancy_rag_bench`

Audit of technical debt, hardcoded values, environment fragility, edge cases, statistical/methodological issues, missing tests, and security/credential risks across `E:\CODE\SC\Papers\SycophancyRAGBench\sycophancy_rag_bench`.

---

## 1. Critical Bugs & Broken Entry Points

- **Undefined `SAMPLE_LIMIT` (`NameError`) in universal notebook**:
  - `Run_SycophancyRAGBench.ipynb` (`source` lines 98, 113) and its generator `scripts/update_nb.py` (lines 119, 134) assign `sample_limit = SAMPLE_LIMIT` in `ConfigArgs` and interpolate `SAMPLE_LIMIT` in a status print, but `SAMPLE_LIMIT` is never defined in the cell. Running the notebook raises `NameError: name 'SAMPLE_LIMIT' is not defined`.
- **Fallback McNemar p-value hardcodes `0.05` / `0.5` and appears in shipped artifacts**:
  - `src/analysis/stats.py` (lines 88–95) catches any exception from `statsmodels.stats.contingency_tables.mcnemar` (or missing `statsmodels`) and assigns `pval = 0.05 if stat > 3.841 else 0.5`.
  - This fallback produced the shipped `artifacts/stats/tables/table2_mcnemar_tests.tex` (lines 7–8), where both rows report exact `0.05` for `pvalue` and `pvalue_fdr` despite differing statistics (`n10_fixed=17, n01_worsened=33` and `n10_fixed=0, n01_worsened=34`).
- **Inverted McNemar `n10_fixed` / `n01_worsened` semantics for positive outcomes (`j_truthful`)**:
  - `paired_mcnemar_test` (`src/analysis/stats.py`, lines 49–50, 74–75, 105–106) hardcodes `n10` as `"n10_fixed"` (1 in baseline, 0 in compare) and `n01` as `"n01_worsened"` (0 in baseline, 1 in compare). That holds for `j_endorses` (1 = sycophantic), but is inverted for `j_truthful` (1 = correct). In `artifacts/stats/tables/table2_mcnemar_tests.tex` (line 8), `j_truthful` improving from `0.83` to `1.0` is labeled `n10_fixed = 0, n01_worsened = 34`.
- **Key mismatch in `bootstrap_ci` empty-series return (`"lo"`/`"hi"` vs `"ci_lower"`/`"ci_upper"`)**:
  - `bootstrap_ci` (`src/analysis/stats.py`, lines 137–147) returns `{"mean": None, "lo": None, "hi": None, "n": 0}` on empty input, vs `{"mean": ..., "ci_lower": ..., "ci_upper": ..., "n": ...}` on non-empty input.
  - Callers in `src/analysis/plots.py` (lines 75–76, 195–197) unconditionally index `ci["ci_lower"]` and `ci["ci_upper"]` (and multiply `ci['mean']*100` in `export_latex_tables`), raising `KeyError` / `TypeError` if any `(model, setting)` subgroup has zero belief prompts.
- **Cohen's Kappa human-validation merge key mismatch (`ekey`)**:
  - `run_all.py` (line 252) builds `ekey = f"{r['model']}__{r['setting']}__{r['uid']}"` using settings such as `rag_bm25`, `rag_dense`, and `rag_hybrid`.
  - Bundled `data/manual_validation_labeled.csv` (lines 2–10; unique settings `rag`, `vanilla`) uses `rag` (e.g., `gemma-3-4b__rag__tqa_3__authority`). `pd.merge(df, h_df, on="ekey", ...)` in `compute_cohen_kappas` (`src/analysis/stats.py`, line 187) therefore fails to match any RAG rows from current pipeline runs.

---

## 2. Technical Debt & Repository Drift

- **Notebook directory mismatch & stale notebook stubs**:
  - `README.md` (lines 179–184) references `notebooks/00_setup_and_data.ipynb`–`04_stats_and_publication.ipynb`, whereas `scripts/generate_notebooks.py` (line 8) writes them to `scripts/notebooks/`. No top-level `notebooks/` directory exists.
  - `scripts/notebooks/02_inference.ipynb` (`InferenceArgs`, lines 30–40) and `03_evaluation_and_causal.ipynb` (`EvalArgs`, lines 29–39) omit `fresh`, `no_cache`, and `all_data` attributes defined in `run_all.py` (`parse_args`, lines 55–64).
  - `scripts/notebooks/04_stats_and_publication.ipynb` (lines 27, 38–45) imports `compute_all_bootstrap_cis` and `compute_cohen_kappas` without calling them, skips `compute_causal_conflict_metrics`, and runs `paired_mcnemar_test` only on `"j_endorses"` (omitting `"j_truthful"` from `run_all.py`, lines 311–312).
- **Orphaned / unused config definitions (`configs/config.py`)**:
  - `RELEASE_DIR` (line 50), `CORPUS_STATE` (line 71), `METRICS_PATH` (line 83), `MANUAL_SAMPLE` (line 85), `DOMAINS` (line 103), `CORE_SETTINGS` (line 152), `WIKI_DATASET` / `WIKI_CONFIG` / `N_DISTRACTORS` (lines 174–176), and `CAUSAL_CONFLICT_CATEGORIES` (lines 188–193) are never referenced by `run_all.py` or `src/`.
  - `WIKI_DATASET` / `N_DISTRACTORS = 50000` has no corpus-streaming implementation; retrieval only reads `GOLD_PASSAGES_PATH` / `DISTRACTORS_PATH` (or bundled fallbacks).
- **Tracked build artifacts & partial results in workspace**:
  - `artifacts/data/{claims,prompts}.jsonl`, `artifacts/stats/figures/*`, `artifacts/stats/tables/*.tex`, and `**/__pycache__/*.pyc` are present in the working tree.
  - `artifacts/stats/tables/table1_main_results.tex` (lines 7–8) and `table2_mcnemar_tests.tex` (lines 7–8) contain partial/anomalous metrics for `llama-3.2-3b` only (`vanilla` vs `rag_hybrid`, with `Truthfulness = 100.0`, `Pushback Rate = 100.0`, `Sycophancy Rate = 50.0` for `rag_hybrid`).
- **Unescaped LaTeX fallback in `_df_to_latex`**:
  - `src/analysis/plots.py` (lines 158–171) falls back to manual string formatting when `df.to_latex()` fails (e.g., `jinja2` absent, as `jinja2` is omitted from `requirements.txt`), ignoring `escape=escape`. In `artifacts/stats/tables/table2_mcnemar_tests.tex` (lines 6–8), column names and values retain raw underscores (`baseline_setting`, `rag_hybrid`, `j_endorses`), causing LaTeX compilation errors.

---

## 3. Hardcoded Paths, Values & Environment Fragility

- **Hardcoded Windows path in notebook script**:
  - `scripts/update_nb.py` (line 4) hardcodes `nb_path = "e:/CODE/SC/Papers/SycophancyRAGBench/sycophancy_rag_bench/Run_SycophancyRAGBench.ipynb"`, failing on any other path or OS.
- **Fragile Colab/Kaggle detection via filesystem existence**:
  - `detect_runtime()` (`configs/config.py`, lines 12–18) and `is_kaggle()` / `is_colab()` (`src/env.py`, lines 20–29) classify any machine with `/kaggle/working` as Kaggle and any machine with `/content` as Colab, overriding local `PROJECT_DIR` to `/kaggle/working/SycophancyRAGBench` or `/content/drive/MyDrive/SycophancyRAGBench`.
- **Incomplete `cfg` fallback on Colab Drive mount failure**:
  - `setup_workspace()` (`src/env.py`, lines 138–146) updates `cfg.PROJECT_DIR` and the 11 directory variables (`DATA_DIR`..`RELEASE_DIR`) when Google Drive fails to mount, but does **not** update `cfg.ALL_DIRS` or the file path constants (`CLAIMS_PATH`, `PROMPTS_PATH`, `GOLD_PASSAGES_PATH`, `DISTRACTORS_PATH`, `RETRIEVAL_RESULTS`, `RETRIEVAL_METRICS`, `JUDGED_PATH`, `CAUSAL_METRICS_PATH`, `MANUAL_LABELS`) bound at import time in `configs/config.py` (lines 52–86). Downstream reads/writes still point to `/content/drive/MyDrive/SycophancyRAGBench/...`.
- **`reset_workspace_cache()` breaks `--fresh` + human kappa & skips `COLLECTION_DIR`/`INDEX_DIR`**:
  - `run_all.py` calls `env.setup_workspace()` (line 123, which copies `BUNDLED_HUMAN_LABELS` to `cfg.MANUAL_LABELS` inside `EVAL_DIR`) *before* `env.reset_workspace_cache()` (line 127), which wipes `cfg.EVAL_DIR` (lines 181–196). A `--fresh` run deletes `cfg.MANUAL_LABELS`, silently disabling human-validation Kappa in `compute_cohen_kappas`.
  - Conversely, `reset_workspace_cache()` does not clear `cfg.COLLECTION_DIR` or `cfg.INDEX_DIR` despite its docstring claiming to purge retrieval indexes (lines 177–184).
- **`--all_data` incompatible with bundled gold passages**:
  - `run_all.py` (lines 132–137) allows `--all_data` (`N_FEVER = None`, `N_TRUTHFULQA = None`), rebuilding `claims.jsonl` from HuggingFace (`src/data_loader.py`, lines 17–140). However, `gold_passages.jsonl` only stores 596 passages (234 unique titles) for the bundled 200 FEVER claims. Extra FEVER claims lack gold passages in `COLLECTION_DIR`, degrading `BM25`/`Dense`/`Hybrid` and forcing `OracleRetriever` into its TruthfulQA fallback (`src/retrieval/controls.py`, lines 51–59).
- **Dependency gaps (`requirements.txt` & `Run_SycophancyRAGBench.ipynb`)**:
  - `requirements.txt` (lines 1–23) specifies only `>=` lower bounds and omits `jinja2` (needed for `pandas.DataFrame.to_latex`) and `pyserini` (optional in `src/retrieval/bm25.py`, line 108).
  - `Run_SycophancyRAGBench.ipynb` (`source` line 70) runs `!pip install -q -U "transformers>=4.51.0" accelerate bitsandbytes sentence-transformers datasets statsmodels scikit-learn matplotlib tqdm`, omitting `torch`, `pandas`, `numpy`, `huggingface_hub`, and `jinja2`.
- **Hardcoded model/judge limits & batching**:
  - `ChatModel.generate` (`src/models/chat_model.py`, line 145) hardcodes `max_length=3072` with `truncation=True`; because `build_rag_user_text` (`src/utils.py`, lines 273–278) puts retrieved passages *before* the user prompt, right-truncation can clip the user's belief claim and question.
  - `Judge` (`src/evaluation/judge.py`) hardcodes `torch.float16` (lines 148, 155; unlike `ChatModel` which checks `is_bf16_supported`), truncates responses at `[:1200]` chars (line 81), sets `max_length=2048` (line 181) and `max_new_tokens=96` (line 187), and `run_all.py` (line 265) hardcodes judge batch size `16` instead of `args.batch_size`.
  - `build_bm25_retriever` (`src/retrieval/bm25.py`, line 133) defaults `prefer_lucene=False`, and `src/retrieval/manager.py` (line 94) never enables it.

---

## 4. Runtime & Data Edge Cases

- **Dense embedding cache collision (`src/retrieval/dense.py`)**:
  - `_build_or_load_embeddings` (lines 60–68) validates `dense_embeddings.pt` solely by `cached.shape[0] == len(self.corpus)`. Changing `DENSE_MODEL_ID` or passage contents/order while keeping passage count equal silently loads incompatible embeddings.
- **Duplicate `ekey` / empty-response resumption risks (`run_all.py`, `src/utils.py`, `src/analysis/stats.py`)**:
  - `done_keys` (`src/utils.py`, lines 124–126) ignores records with empty `"response"`, so failed empty generations remain in `<model>__<setting>.jsonl` and are re-appended on resume; `run_all.py` deduplicates `all_gen_rows` by `ekey` (line 255), but `JUDGED_PATH` is read directly via `U.read_jsonl(cfg.JUDGED_PATH)` (line 290) without `ekey` deduplication.
  - If `df` has duplicate `(model, setting, uid)` rows, `paired_mcnemar_test` (`src/analysis/stats.py`, lines 56–69) indexing `v.loc[idx]` / `r.loc[idx]` on non-unique `uid`s will raise `ValueError` or miscount pairs.
- **Empty `JUDGED_PATH` crash**:
  - `run_all.py` (lines 288–301) checks `os.path.exists(cfg.JUDGED_PATH)` before reading `judged_data`, but if the file exists and is empty, `df_judged["model"]` raises `KeyError`.
- **Silent parse failures in `parse_judge_output` (`src/evaluation/judge.py`)**:
  - Lines 99–120 match the first `{.*?}` (fails on nested braces) and default unparsed `"truthful"` to `"incorrect"` and unparsed boolean flags to `"no"`. Downstream stats (`stats.py`, `causal_evaluator.py`, `plots.py`) never filter or report `j_parsed == False`.
- **BM25 zero-overlap fallback returns fixed corpus prefix (`src/retrieval/bm25.py`)**:
  - When no query terms match (`if not scores:`, lines 81–89), `PureBM25.search` returns the first `k` corpus documents (`self.corpus[0:k]`) with `score: 0.0`, injecting arbitrary leading gold passages (`Arthur Schopenhauer`, `Alexandria`) instead of empty hits.
- **Unicode normalization gap in FEVER title matching (`src/utils.py`, `data/claims.jsonl`)**:
  - `norm_title` (`src/utils.py`, lines 149–151) does not apply `unicodedata.normalize("NFC", ...)`. In `data/claims.jsonl`, 199/200 FEVER claims match `data/gold_passages.jsonl`; the remaining claim (`fever_78426`, line 8) uses decomposed `José_Ferrer` (`e\u0301`).
- **Concurrent JSONL append hazard (`src/utils.py`)**:
  - `JsonlAppender` (lines 91–116) opens in `"a"` mode without file locking; parallel runs sharing `PROJECT_DIR` can interleave writes. Also `read_jsonl` (lines 48–57) silently drops *any* malformed JSON line anywhere in the file, not only a truncated trailing line.
- **`export_latex_tables` return list vs written files (`src/analysis/plots.py`)**:
  - When `mcnemar_rows` is empty (lines 211–224), `table2_mcnemar_tests.tex` is not created, yet `[t1_tex_path, t2_tex_path]` is still returned.

---

## 5. Statistical & Methodological Edge Cases

- **Clustered observations treated as independent in Bootstrap CIs & McNemar tests (`src/analysis/stats.py`)**:
  - Each of the 300 claims generates 5 correlated belief prompts (`weak`, `strong`, `authority`, `emotional`, `social`). `bootstrap_ci` (lines 134–147) resamples individual prompt rows rather than clustering by `claim_id`, and `paired_mcnemar_test` (lines 39–109) runs on all `uid`s rather than claim-level aggregates, underestimating variance and inflating effective sample size ($5\times$).
- **Inconsistent `Pushback Rate` / `correction_rate` denominator**:
  - `compute_all_bootstrap_cis` (`src/analysis/stats.py`, lines 160–164) computes `correction_rate` on `belief_grp` (`style != "neutral"`), whereas `export_latex_tables` (`src/analysis/plots.py`, lines 187–197) computes `ci_corr` on `grp` (including `neutral` prompts).
- **Intra-family LLM judge bias (`configs/config.py`)**:
  - `JUDGE_HF_ID = "Qwen/Qwen2.5-7B-Instruct"` (line 181) evaluates `Qwen/Qwen2.5-3B-Instruct` alongside `gemma-3-4b` and `llama-3.2-3b`, risking same-family bias without cross-judge validation.
- **`OracleRetriever` asymmetry for TruthfulQA (`src/retrieval/controls.py`)**:
  - FEVER claims receive up to `k=5` multi-sentence Wikipedia passages (~120 words each), whereas TruthfulQA claims (`gold_titles == []`) receive a single synthetic `"Verified Fact"` passage containing `claim["true_answer"]` (lines 50–59), conflating retrieval upper-bound behavior with direct answer injection.
- **Single-reference `true_answer` & generic FEVER ground truth (`src/data_loader.py`)**:
  - `load_truthfulqa_claims` (line 122) keeps only `trues[0]` rather than all `correct_answers`; `load_fever_claims` (line 46) sets `true_answer` to `"The statement is FALSE; it is refuted by the reference evidence."` without the correcting fact, limiting judge granularity on `"correct"` vs `"partial"`.
- **Conflict taxonomy mislabels `control_irrelevant` and ignores `j_truthful` / evidence grounding (`src/evaluation/causal_evaluator.py`)**:
  - `classify_conflict_resolution` (lines 38–48) enters conflict categories whenever `had_evidence` is `True`, which includes `control_irrelevant` (off-topic distractors)—labeling responses as `"conflict_evidence_aligned"` even when no relevant evidence was provided.
  - Line 43 (`elif truthful or corrects:`) classifies a factually `"incorrect"` response as `"conflict_evidence_aligned"` if `j_corrects == "yes"`, and never checks `evidence_overlap`.
- **Substring domain tagging bug (`src/utils.py`)**:
  - `tag_domain` (lines 210–218) uses substring containment (`kw in t`) without word boundaries. Because `"app"` is in `DOMAIN_KEYWORDS["technology"]` (line 200), any TruthfulQA question containing `"happens"` (e.g., `tqa_66` and `tqa_26` in `data/manual_validation_labeled.csv`, lines 3, 7) is misclassified as `"technology"`.

---

## 6. Missing Tests & Validation Gaps

- **No automated test suite or CI**:
  - No `tests/` directory, `pytest`/`unittest` files, or CI workflow exist anywhere in `sycophancy_rag_bench/`. Core parsing (`parse_judge_output`), retrieval (`PureBM25`, `HybridRetriever`), statistics (`paired_mcnemar_test`, `apply_fdr_correction`, `bootstrap_ci`), and end-to-end `--sample_limit` execution are untested.
- **No schema validation on JSONL/CSV inputs**:
  - `src/data_loader.py` and `run_all.py` perform no schema validation on bundled or generated `claims.jsonl`, `prompts.jsonl`, or `judged.jsonl`, and `README.md` (line 177) states 160 human validation labels while `data/manual_validation_labeled.csv` has 159 data rows (160 lines including header).

---

## 7. Security & Credential Risks

- **Notebook encourages hardcoding `HF_TOKEN` (`Run_SycophancyRAGBench.ipynb`, `scripts/update_nb.py`)**:
  - `Run_SycophancyRAGBench.ipynb` (`source` lines 76–78) instructs users to paste `"hf_..."` directly into `HF_TOKEN = None`, risking accidental token commits or sharing in exported `.ipynb` files.
- **Plaintext HF token persistence (`src/env.py`)**:
  - `authenticate_hf` (line 112) calls `huggingface_hub.login(token=tok, add_to_git_credential=False)`, persisting the token on disk in shared cloud/container environments.
- **Unsafe `torch.load` deserialization (`src/retrieval/dense.py`)**:
  - Line 62 calls `torch.load(self.cache_path, map_location=self.device)` without `weights_only=True`, allowing arbitrary code execution via pickle if a cached `.pt` file is tampered with or shared.
- **Unpinned remote model/dataset revisions (`src/models/chat_model.py`, `src/evaluation/judge.py`, `src/retrieval/dense.py`, `src/data_loader.py`)**:
  - `from_pretrained(...)`, `SentenceTransformer(...)`, and `load_dataset(...)` do not pin HuggingFace `revision` hashes.
- **Over-broad `"index"` substring exclusion in `package_outputs` (`src/env.py`)**:
  - Line 238 (`if exclude_large_index and "index" in root:`) matches any directory path containing `"index"` anywhere in `root` rather than `cfg.INDEX_DIR` specifically, and zips all remaining files under `cfg.PROJECT_DIR`.
