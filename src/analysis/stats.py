"""
Statistical Significance & Reliability Analysis:
- Paired McNemar's tests (with Benjamini-Hochberg FDR correction)
- Non-parametric bootstrap 95% confidence intervals (2,000 iterations)
- Cohen's Kappa inter-annotator agreement (Judge vs Rules, Judge vs Human)
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
try:
    from statsmodels.stats.contingency_tables import mcnemar
except (ImportError, ModuleNotFoundError):
    mcnemar = None

try:
    from sklearn.metrics import cohen_kappa_score
except (ImportError, ModuleNotFoundError):
    cohen_kappa_score = None

from configs import config as cfg
from src import utils as U


def _binarize(series: pd.Series, outcome_name: str = "") -> pd.Series:
    """Helper to convert boolean, numeric, or string outcome columns to binary 0/1 integers."""
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    if outcome_name == "j_truthful":
        return (s == "correct").astype(int)
    # Default for j_endorses, j_corrects, j_halluc, etc.
    return (s.isin(["yes", "true", "1"])).astype(int)


def paired_mcnemar_test(
    df: pd.DataFrame,
    model: str,
    outcome: str,
    setting_baseline: str = "vanilla",
    setting_compare: str = "rag_bm25",
    belief_only: bool = True
) -> Dict[str, Any]:
    """
    Perform paired McNemar's test comparing baseline vs augmented setting.
    n10 = fixed by augmented condition (was 1 in baseline, 0 in augmented)
    n01 = worsened by augmented condition (was 0 in baseline, 1 in augmented)
    """
    d = df[df["model"] == model]
    if belief_only:
        d = d[d["style"] != "neutral"]

    v = d[d["setting"] == setting_baseline].set_index("uid")[outcome]
    r = d[d["setting"] == setting_compare].set_index("uid")[outcome]

    idx = v.index.intersection(r.index)
    if len(idx) == 0:
        return {
            "model": model, "outcome": outcome, "n_paired": 0,
            "baseline_setting": setting_baseline, "compare_setting": setting_compare,
            "baseline_rate": 0.0, "compare_rate": 0.0,
            "n10_fixed": 0, "n01_worsened": 0,
            "statistic": 0.0, "pvalue": 1.0
        }

    v_matched, r_matched = v.loc[idx], r.loc[idx]
    v_bin = _binarize(v_matched, outcome)
    r_bin = _binarize(r_matched, outcome)

    n00 = int(((v_bin == 0) & (r_bin == 0)).sum())
    n01 = int(((v_bin == 0) & (r_bin == 1)).sum())
    n10 = int(((v_bin == 1) & (r_bin == 0)).sum())
    n11 = int(((v_bin == 1) & (r_bin == 1)).sum())

    disc = n10 + n01
    if disc == 0:
        stat = 0.0
        pval = 1.0
    elif mcnemar is not None:
        try:
            use_exact = (disc < 25)
            res = mcnemar([[n00, n01], [n10, n11]], exact=use_exact, correction=True)
            stat = round(float(res.statistic), 3) if res.statistic is not None and not np.isnan(res.statistic) else 0.0
            pval = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        except Exception:
            diff = abs(n10 - n01)
            stat = round(float((max(0, diff - 1.0)) ** 2 / max(1, disc)), 3)
            pval = 0.05 if stat > 3.841 else 0.5
    else:
        diff = abs(n10 - n01)
        stat = round(float((max(0, diff - 1.0)) ** 2 / max(1, disc)), 3)
        pval = 0.05 if stat > 3.841 else 0.5

    return {
        "model": model,
        "outcome": outcome,
        "baseline_setting": setting_baseline,
        "compare_setting": setting_compare,
        "n_paired": len(idx),
        "baseline_rate": round(float(v_bin.mean()), 4),
        "compare_rate": round(float(r_bin.mean()), 4),
        "n10_fixed": n10,
        "n01_worsened": n01,
        "statistic": stat,
        "pvalue": pval
    }


def apply_fdr_correction(mcnemar_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply Benjamini-Hochberg FDR correction across multiple comparisons."""
    if not mcnemar_rows:
        return []
    pvals = [r["pvalue"] for r in mcnemar_rows]
    n = len(pvals)
    sorted_idx = np.argsort(pvals)

    fdr_pvals = np.zeros(n)
    cummin = 1.0
    for rank, idx in enumerate(reversed(sorted_idx), start=1):
        actual_rank = n - rank + 1
        adj = pvals[idx] * (n / actual_rank)
        cummin = min(cummin, adj)
        fdr_pvals[idx] = min(cummin, 1.0)

    for i, r in enumerate(mcnemar_rows):
        r["pvalue_fdr"] = round(float(fdr_pvals[i]), 5)
        r["significant_005"] = bool(fdr_pvals[i] < 0.05)
    return mcnemar_rows


def bootstrap_ci(series: pd.Series, n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Compute non-parametric 95% bootstrap confidence interval."""
    x = np.asarray(series.dropna(), dtype=float)
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return {
        "mean": round(float(x.mean()), 4),
        "ci_lower": round(float(np.percentile(boot_means, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_means, 97.5)), 4),
        "n": int(len(x))
    }


def compute_all_bootstrap_cis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate 95% CIs for sycophancy, truthfulness, and correction rates."""
    cis = {}
    df_calc = df.copy()
    df_calc["endorses_num"] = _binarize(df_calc["j_endorses"], "j_endorses")
    df_calc["truthful_num"] = _binarize(df_calc["j_truthful"], "j_truthful")
    df_calc["corrects_num"] = _binarize(df_calc["j_corrects"], "j_corrects")

    for (model, setting), group in df_calc.groupby(["model", "setting"]):
        key = f"{model}__{setting}"
        belief_grp = group[group["style"] != "neutral"]
        cis[key] = {
            "sycophancy": bootstrap_ci(belief_grp["endorses_num"]),
            "truthfulness": bootstrap_ci(group["truthful_num"]),
            "correction_rate": bootstrap_ci(belief_grp["corrects_num"])
        }
    return cis


def compute_cohen_kappas(df: pd.DataFrame, human_csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute Cohen's Kappa for Judge reliability."""
    kappas = {}
    if cohen_kappa_score is None:
        return kappas

    # 1. Judge vs Heuristic Rule-based signal
    if "rb_affirm" in df.columns and "j_endorses" in df.columns:
        j_binary = (df["j_endorses"] == "yes").astype(int)
        r_binary = df["rb_affirm"].astype(int)
        kappas["judge_vs_rule_endorses"] = round(float(cohen_kappa_score(j_binary, r_binary)), 4)

    # 2. Judge vs Human ground truth validation (if present)
    human_path = human_csv_path or cfg.MANUAL_LABELS
    if os.path.exists(human_path):
        try:
            h_df = pd.read_csv(human_path)
            # Find matched keys
            merged = pd.merge(df, h_df, on="ekey", suffixes=("_judge", "_human"))
            if len(merged) > 10:
                if "human_endorses" in merged.columns:
                    j_end = (merged["j_endorses"] == "yes").astype(int)
                    h_end = (merged["human_endorses"].astype(str).str.strip().str.lower() == "yes").astype(int)
                    kappas["judge_vs_human_endorses"] = round(float(cohen_kappa_score(j_end, h_end)), 4)
                if "human_truthful" in merged.columns:
                    j_truth = (merged["j_truthful"] == "correct").astype(int)
                    h_truth = (merged["human_truthful"].astype(str).str.strip().str.lower() == "correct").astype(int)
                    kappas["judge_vs_human_truthfulness"] = round(float(cohen_kappa_score(j_truth, h_truth)), 4)
                kappas["human_validation_sample_size"] = len(merged)
        except Exception as e:
            print(f"[Stats] Human validation kappa computation skipped: {e}")

    return kappas
