"""
Dataset construction & loading for SycophancyRAGBench:
- FEVER refuted claims (with gold Wikipedia titles)
- TruthfulQA human misconceptions (with verified true refutations)
- 1,800 factorial prompts across 6 belief styles
"""

import os
import random
from typing import List, Dict, Any, Optional
import pandas as pd

from configs import config as cfg
from src import utils as U


def load_fever_claims(n: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load refuted claims from FEVER.
    If n is None, loads ALL available refuted claims.
    Uses copenlu/fever_gold_evidence with fallback to fever/fever.
    """
    from datasets import load_dataset
    target_str = f"{n} refuted claims" if n else "ALL available refuted claims"
    print(f"[Data] Loading FEVER dataset ({target_str})...")

    claims = []
    # Try modern parquet-based copenlu/fever_gold_evidence first
    try:
        ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
        for r in ds:
            if r.get("label") != "REFUTES":
                continue
            cid = str(r.get("original_id") or r.get("id"))
            fc = r.get("claim", "").strip()
            ev_list = r.get("evidence", [])
            gold_titles = []
            for ev_item in ev_list:
                if isinstance(ev_item, list) and len(ev_item) > 0:
                    gold_titles.append(ev_item[0])

            claims.append({
                "claim_id": f"fever_{cid}",
                "source": "fever",
                "false_claim": fc,
                "true_answer": "The statement is FALSE; it is refuted by the reference evidence.",
                "gold_titles": sorted(list(set(gold_titles))),
                "domain": U.tag_domain(fc),
                "question": f'Is the following statement true? "{fc}"',
                "retrieval_query": fc
            })
    except Exception as e:
        print(f"[Data] copenlu/fever_gold_evidence failed ({e}), trying fever/fever...")
        ds = load_dataset("fever/fever", "v1.0", split="labelled_dev")
        by_id = {}
        for r in ds:
            if r["label"] != "REFUTES":
                continue
            cid = r["id"]
            rec = by_id.setdefault(cid, {
                "claim_id": f"fever_{cid}",
                "source": "fever",
                "false_claim": r["claim"].strip(),
                "true_answer": "The statement is FALSE; it is refuted by the reference evidence.",
                "gold_titles": set()
            })
            for ev_set in r.get("evidence", []):
                for ev_item in ev_set:
                    title = ev_item[2] if len(ev_item) > 2 else None
                    if title:
                        rec["gold_titles"].add(title)
        for v in by_id.values():
            if v["gold_titles"]:
                v["gold_titles"] = sorted(list(v["gold_titles"]))
                v["domain"] = U.tag_domain(v["false_claim"])
                v["question"] = f'Is the following statement true? "{v["false_claim"]}"'
                v["retrieval_query"] = v["false_claim"]
                claims.append(v)

    rng = random.Random(seed)
    rng.shuffle(claims)
    selected = claims[:n] if n else claims

    print(f"[Data] Selected {len(selected)} FEVER refuted claims.")
    return selected


def load_truthfulqa_claims(n: Optional[int] = None, seed: int = 43) -> List[Dict[str, Any]]:
    """
    Load common human misconceptions from TruthfulQA generation validation split.
    If n is None, loads ALL (817) questions.
    """
    from datasets import load_dataset
    target_str = f"{n} misconceptions" if n else "ALL (817) misconceptions"
    print(f"[Data] Loading TruthfulQA dataset ({target_str})...")

    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except Exception:
        ds = load_dataset("truthful_qa", "generation", split="validation")

    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)

    cat_map = {
        "Health": "medical", "Nutrition": "medical", "Psychology": "medical",
        "Science": "scientific", "Physics": "scientific", "Biology": "scientific",
        "History": "historical", "Law": "political", "Politics": "political",
        "Economics": "political", "Technology": "technology", "Weather": "climate"
    }

    out = []
    for r in rows:
        q = r["question"].strip()
        falses = [a.strip() for a in r.get("incorrect_answers", []) if a.strip()]
        trues = [a.strip() for a in r.get("correct_answers", []) if a.strip()]
        if not falses or not trues:
            continue

        fc = rng.choice(falses)
        tc = trues[0]
        cat = r.get("category", "")
        domain = cat_map.get(cat, U.tag_domain(q + " " + fc))

        out.append({
            "claim_id": f"tqa_{len(out):03d}",
            "source": "truthfulqa",
            "domain": domain,
            "false_claim": fc if fc.endswith(".") else (fc + "."),
            "true_answer": tc,
            "question": q,
            "retrieval_query": q,
            "gold_titles": []
        })
        if n and len(out) >= n:
            break

    print(f"[Data] Selected {len(out)} TruthfulQA misconception claims.")
    return out


def generate_prompts(claims: List[Dict[str, Any]], styles: List[str] = None) -> List[Dict[str, Any]]:
    """Expand claims into factorial prompts across all belief styles."""
    styles = styles or cfg.BELIEF_STYLES
    prompts = []
    for c in claims:
        for s in styles:
            p_text = U.build_belief_prompt(s, c["false_claim"], c["question"], c["source"])
            prompts.append({
                "uid": f"{c['claim_id']}__{s}",
                "claim_id": c["claim_id"],
                "source": c["source"],
                "domain": c["domain"],
                "style": s,
                "false_claim": c["false_claim"],
                "true_answer": c["true_answer"],
                "question": c["question"],
                "retrieval_query": c["retrieval_query"],
                "gold_titles": c.get("gold_titles", []),
                "prompt_text": p_text
            })
    return prompts


def load_or_build_benchmark(
    force_rebuild: bool = False,
    sample_limit: Optional[int] = None
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    Load existing benchmark claims and prompts from disk,
    or build them if not found (or if force_rebuild=True).
    Returns (claims, prompts).
    """
    claims = []
    prompts = []

    # 1. Check active workspace files
    if os.path.exists(cfg.CLAIMS_PATH) and os.path.exists(cfg.PROMPTS_PATH) and not force_rebuild:
        claims = U.read_jsonl(cfg.CLAIMS_PATH)
        prompts = U.read_jsonl(cfg.PROMPTS_PATH)
        print(f"[Data] Loaded {len(claims)} claims and {len(prompts)} prompts from active workspace.")
    # 2. Check bundled package files
    elif os.path.exists(cfg.BUNDLED_CLAIMS) and os.path.exists(cfg.BUNDLED_PROMPTS) and not force_rebuild:
        claims = U.read_jsonl(cfg.BUNDLED_CLAIMS)
        prompts = U.read_jsonl(cfg.BUNDLED_PROMPTS)
        print(f"[Data] Loaded {len(claims)} claims and {len(prompts)} prompts from pre-bundled package data.")
        # Mirror to active workspace
        try:
            U.write_jsonl(cfg.CLAIMS_PATH, claims)
            U.write_jsonl(cfg.PROMPTS_PATH, prompts)
        except Exception:
            pass
    else:
        print("[Data] Generating benchmark dataset from scratch...")
        try:
            fever = load_fever_claims(cfg.N_FEVER, cfg.SEED)
            tqa = load_truthfulqa_claims(cfg.N_TRUTHFULQA, cfg.SEED + 1)
            claims = fever + tqa
            prompts = generate_prompts(claims, cfg.BELIEF_STYLES)

            U.write_jsonl(cfg.CLAIMS_PATH, claims)
            U.write_jsonl(cfg.PROMPTS_PATH, prompts)
            print(f"[Data] Saved {len(claims)} claims -> {cfg.CLAIMS_PATH}")
            print(f"[Data] Saved {len(prompts)} prompts -> {cfg.PROMPTS_PATH}")
        except Exception as e:
            if os.path.exists(cfg.BUNDLED_CLAIMS) and os.path.exists(cfg.BUNDLED_PROMPTS):
                print(f"[Data] Rebuild encountered ({e}). Falling back to bundled claims.")
                claims = U.read_jsonl(cfg.BUNDLED_CLAIMS)
                prompts = U.read_jsonl(cfg.BUNDLED_PROMPTS)
            else:
                raise e

    # Handle quick smoke-test sample limit if requested
    if sample_limit and sample_limit < len(claims):
        selected_claims = claims[:sample_limit]
        claim_ids = {c["claim_id"] for c in selected_claims}
        selected_prompts = [p for p in prompts if p["claim_id"] in claim_ids]
        print(f"[Data] (Smoke-test mode) Limiting run to {len(selected_claims)} claims ({len(selected_prompts)} prompts).")
        return selected_claims, selected_prompts

    return claims, prompts


def print_data_summary(prompts: List[Dict[str, Any]]):
    """Print tabular distribution summary."""
    df = pd.DataFrame(prompts)
    print("\n--- Benchmark Data Summary ---")
    print(f"Total Prompts: {len(df)}")
    print(f"Claims:        {df['claim_id'].nunique()}")
    print("\nPrompts per Style:")
    print(df["style"].value_counts().to_string())
    print("\nPrompts per Domain:")
    print(df["domain"].value_counts().to_string())
    print("\nPrompts per Source:")
    print(df["source"].value_counts().to_string())
    print("------------------------------\n")
