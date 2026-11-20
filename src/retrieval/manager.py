"""
Retrieval Manager for SycophancyRAGBench.
Orchestrates multi-retriever execution (BM25, Dense, Hybrid, Oracle, Irrelevant)
and evaluates retrieval quality (Recall@k and MRR@10).
"""

import os
import json
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

from configs import config as cfg
from src import utils as U
from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import build_bm25_retriever
from src.retrieval.dense import build_dense_retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.controls import OracleRetriever, IrrelevantRetriever


def evaluate_retrieval_quality(
    retrieval_results: List[Dict[str, Any]],
    ks: List[int] = None
) -> Dict[str, Any]:
    """Compute Recall@k and MRR@10 against FEVER claims with gold evidence titles."""
    ks = ks or cfg.RETRIEVAL_KS
    fever_claims = [r for r in retrieval_results if r.get("source") == "fever" and r.get("gold_titles")]
    n = len(fever_claims)
    if n == 0:
        return {"n_eval": 0}

    recall = {k: 0 for k in ks}
    reciprocal_ranks = 0.0

    for r in fever_claims:
        gold_norm = {U.norm_title(t) for t in r.get("gold_titles", [])}
        hits = r.get("hits", [])

        # Find first relevant hit rank (1-indexed)
        first_rank = None
        for i, h in enumerate(hits):
            if U.norm_title(h.get("title", "")) in gold_norm:
                first_rank = i + 1
                break

        for k in ks:
            if first_rank and first_rank <= k:
                recall[k] += 1
        if first_rank and first_rank <= 10:
            reciprocal_ranks += (1.0 / first_rank)

    return {
        "n_fever_claims": n,
        "recall_at_k": {f"R@{k}": round(recall[k] / n, 4) for k in ks},
        "mrr_at_10": round(reciprocal_ranks / n, 4)
    }


def run_retrieval_pipeline(
    claims: List[Dict[str, Any]],
    retrievers_to_run: Optional[List[str]] = None,
    k: int = cfg.TOP_K
) -> Dict[str, str]:
    """
    Run retrieval for all specified retrievers and controls.
    Returns dictionary mapping retriever name -> results filepath.
    """
    to_run = retrievers_to_run or ["bm25", "dense", "hybrid", "oracle", "irrelevant"]
    results_paths = {}
    all_metrics = {}

    print(f"\n[Retrieval] Initiating retrieval pipeline for {len(claims)} claims across: {to_run}")

    # 1. Load gold and distractor passages
    gold_passages = []
    for gp_path in [cfg.GOLD_PASSAGES_PATH, cfg.BUNDLED_GOLD]:
        if os.path.exists(gp_path):
            gold_passages = U.read_jsonl(gp_path)
            if gold_passages:
                break

    distractors = []
    for dp_path in [cfg.DISTRACTORS_PATH, cfg.BUNDLED_DISTRACTORS]:
        if os.path.exists(dp_path):
            distractors = U.read_jsonl(dp_path)
            if distractors:
                break

    # Check which retrievers actually have unretrieved claims before loading heavy models/indexes
    methods_needing_compute = []
    for method in to_run:
        out_path = cfg.RETRIEVAL_RESULTS.get(method, os.path.join(cfg.RETRIEVAL_DIR, f"retrieval_{method}.jsonl"))
        done = U.done_keys(out_path, "claim_id")
        if any(c["claim_id"] not in done for c in claims):
            methods_needing_compute.append(method)

    # Lazy-load retrievers only if needed
    instantiated = {}

    if any(m in methods_needing_compute for m in ["bm25", "hybrid"]):
        print("[Retrieval] Initializing BM25 retriever...")
        instantiated["bm25"] = build_bm25_retriever()

    if any(m in methods_needing_compute for m in ["dense", "hybrid"]):
        print("[Retrieval] Initializing Dense retriever...")
        instantiated["dense"] = build_dense_retriever()

    if "hybrid" in methods_needing_compute:
        print("[Retrieval] Initializing Hybrid RRF retriever...")
        instantiated["hybrid"] = HybridRetriever(instantiated["bm25"], instantiated["dense"])

    if "oracle" in methods_needing_compute:
        instantiated["oracle"] = OracleRetriever(gold_passages)

    if "irrelevant" in methods_needing_compute:
        instantiated["irrelevant"] = IrrelevantRetriever(distractors, seed=cfg.SEED)

    k_max = max(max(cfg.RETRIEVAL_KS), k)

    for method in to_run:
        out_path = cfg.RETRIEVAL_RESULTS.get(method, os.path.join(cfg.RETRIEVAL_DIR, f"retrieval_{method}.jsonl"))
        results_paths[method] = out_path

        done = U.done_keys(out_path, "claim_id")
        todo = [c for c in claims if c["claim_id"] not in done]

        if not todo:
            print(f"  [{method}] All {len(claims)} claims already retrieved.")
        else:
            print(f"  [{method}] Retrieving for {len(todo)} claims ({len(done)} already done)...")
            retriever = instantiated[method]
            with U.JsonlAppender(out_path) as app:
                for c in tqdm(todo, desc=f"Retriever: {method}"):
                    query = c.get("retrieval_query", c.get("false_claim", ""))
                    if hasattr(retriever, "search_for_claim"):
                        hits = retriever.search_for_claim(c, k=k_max)
                    else:
                        hits = retriever.search(query, k=k_max)

                    app.write({
                        "claim_id": c["claim_id"],
                        "retriever": method,
                        "source": c["source"],
                        "query": query,
                        "gold_titles": c.get("gold_titles", []),
                        "hits": hits
                    })

        # Evaluate quality metrics for this retriever
        res = U.read_jsonl(out_path)
        metrics = evaluate_retrieval_quality(res)
        all_metrics[method] = metrics
        print(f"  [{method}] Metrics: {json.dumps(metrics.get('recall_at_k', {}))} | MRR@10: {metrics.get('mrr_at_10')}")

    # Save aggregated retrieval metrics
    U.save_json(cfg.RETRIEVAL_METRICS, all_metrics)
    print(f"[Retrieval] All metrics saved to {cfg.RETRIEVAL_METRICS}")
    return results_paths
