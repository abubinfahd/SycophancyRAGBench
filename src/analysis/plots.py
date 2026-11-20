"""
Publication-Quality Figures & LaTeX Table Exporters:
Produces 300 DPI PNGs, vector PDFs, and formatted LaTeX tables for academic publication.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Colab/Kaggle
import matplotlib.pyplot as plt

from configs import config as cfg
from src import utils as U
from src.analysis.stats import bootstrap_ci, _binarize


# Academic color palette (Colorblind-friendly / Nature-style)
PALETTE = {
    "vanilla": "#D95F02",            # Red-Orange (high baseline sycophancy)
    "rag_bm25": "#7570B3",           # Purple (sparse)
    "rag_dense": "#1B9E77",          # Green (dense)
    "rag_hybrid": "#386CB0",         # Blue (hybrid)
    "control_oracle": "#66A61E",     # Lime (oracle ceiling)
    "control_irrelevant": "#E6AB02", # Ochre (noise floor)
    "rag_hybrid_caev": "#08306B",    # Deep Navy (Hybrid + CAEV mitigation)
    "oracle_caev": "#00441B",        # Deep Forest Green (Oracle + CAEV mitigation)
}

ORDERED_SETTINGS = [
    "vanilla", "rag_bm25", "rag_dense", "rag_hybrid",
    "rag_hybrid_caev", "control_oracle", "oracle_caev", "control_irrelevant"
]


def setup_plot_style():
    """Configure matplotlib aesthetics for publication."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


def plot_sycophancy_by_model_and_condition(df: pd.DataFrame, output_dir: str = None) -> List[str]:
    """Figure 1: Sycophancy Rate across Models & Retrieval Conditions."""
    setup_plot_style()
    out_dir = output_dir or cfg.FIG_DIR
    os.makedirs(out_dir, exist_ok=True)

    df_belief = df[df["style"] != "neutral"].copy()
    df_belief["endorses_num"] = _binarize(df_belief["j_endorses"], "j_endorses")

    models = sorted(df_belief["model"].unique())
    settings = [s for s in ORDERED_SETTINGS if s in df_belief["setting"].unique()]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.8), 5.5))
    width = 0.8 / len(settings)
    x = np.arange(len(models))

    for i, s in enumerate(settings):
        means = []
        err_low = []
        err_high = []
        for m in models:
            sub = df_belief[(df_belief["model"] == m) & (df_belief["setting"] == s)]["endorses_num"]
            ci = bootstrap_ci(sub)
            mean_val = ci["mean"] if ci["mean"] is not None else 0.0
            means.append(mean_val)
            err_low.append(mean_val - ci["ci_lower"] if ci["ci_lower"] is not None else 0.0)
            err_high.append(ci["ci_upper"] - mean_val if ci["ci_upper"] is not None else 0.0)

        offset = (i - len(settings) / 2 + 0.5) * width
        color = PALETTE.get(s, "#888888")
        label = s.replace("_", " ").title()

        ax.bar(
            x + offset, means, width,
            yerr=[err_low, err_high], capsize=3,
            label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.6
        )

    ax.set_ylabel("Sycophancy Rate (False Belief Endorsement)")
    ax.set_title("Sycophancy Rate across Model Families & Retrieval Paradigms")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    png_path = os.path.join(out_dir, "sycophancy_by_model_and_retriever.png")
    pdf_path = os.path.join(out_dir, "sycophancy_by_model_and_retriever.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    return [png_path, pdf_path]


def plot_evidence_user_conflict_breakdown(df: pd.DataFrame, output_dir: str = None) -> List[str]:
    """Figure 2: Causal Evidence-User Conflict Resolution Breakdown."""
    setup_plot_style()
    out_dir = output_dir or cfg.FIG_DIR
    os.makedirs(out_dir, exist_ok=True)

    from src.evaluation.causal_evaluator import classify_conflict_resolution
    df_belief = df[df["style"] != "neutral"].copy()
    df_belief["conflict_class"] = df_belief.apply(classify_conflict_resolution, axis=1)

    settings = [s for s in ORDERED_SETTINGS if s in df_belief["setting"].unique()]

    conflict_cats = [
        ("conflict_evidence_aligned", "Evidence-Aligned (Epistemic Victory)", "#2CA02C"),
        ("conflict_sycophantic", "Sycophantic Capitulation", "#D62728"),
        ("conflict_hedged", "Compromised / Hedged", "#1F77B4"),
        ("conflict_hallucinated", "Hallucinated Divergence", "#9467BD"),
        ("vanilla_truthful", "Vanilla Truthful", "#2CA02C"),
        ("vanilla_sycophantic", "Vanilla Sycophantic", "#D62728")
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(settings))

    for cat_key, cat_label, cat_color in conflict_cats:
        shares = []
        for s in settings:
            sub = df_belief[df_belief["setting"] == s]
            share = (sub["conflict_class"] == cat_key).mean() if len(sub) > 0 else 0.0
            shares.append(share)

        if sum(shares) > 0.001:
            ax.bar(
                [s.replace("_", "\n") for s in settings],
                shares, bottom=bottom,
                label=cat_label, color=cat_color, alpha=0.85, edgecolor="white", width=0.55
            )
            bottom += np.array(shares)

    ax.set_ylabel("Proportion of Belief Prompts")
    ax.set_title("Causal Resolution of Evidence-User Conflict Across Conditions")
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "evidence_user_conflict_breakdown.png")
    pdf_path = os.path.join(out_dir, "evidence_user_conflict_breakdown.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return [png_path, pdf_path]


def _df_to_latex(df: pd.DataFrame, escape: bool = True) -> str:
    """Export DataFrame to LaTeX with graceful fallback if jinja2 is not installed."""
    try:
        return df.to_latex(index=False, escape=escape)
    except Exception:
        cols = list(df.columns)
        col_spec = "l" * len(cols)
        header = " & ".join([str(c) for c in cols]) + " \\\\ \\hline\n"
        rows = []
        for _, r in df.iterrows():
            row_str = " & ".join([str(v) for v in r.values]) + " \\\\"
            rows.append(row_str)
        body = "\n".join(rows)
        return f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n{header}{body}\n\\hline\n\\end{{tabular}}\n"


def export_latex_tables(df: pd.DataFrame, mcnemar_rows: List[Dict[str, Any]], output_dir: str = None) -> List[str]:
    """Generate conference-ready LaTeX tables for insertion into paper manuscript."""
    out_dir = output_dir or cfg.TABLE_DIR
    os.makedirs(out_dir, exist_ok=True)

    df_calc = df.copy()
    df_calc["endorses_num"] = _binarize(df_calc["j_endorses"], "j_endorses")
    df_calc["truthful_num"] = _binarize(df_calc["j_truthful"], "j_truthful")
    df_calc["corrects_num"] = _binarize(df_calc["j_corrects"], "j_corrects")

    # Table 1: Main benchmark results (Model x Setting)
    t1_rows = []
    for (model, setting), grp in df_calc.groupby(["model", "setting"]):
        b_grp = grp[grp["style"] != "neutral"]
        ci_syco = bootstrap_ci(b_grp["endorses_num"])
        ci_truth = bootstrap_ci(grp["truthful_num"])
        ci_corr = bootstrap_ci(b_grp["corrects_num"])

        t1_rows.append({
            "Model": model.upper(),
            "Condition": setting.replace("_", " ").title(),
            "Sycophancy Rate": f"{ci_syco['mean']*100:.1f} \\small{{[{ci_syco['ci_lower']*100:.1f}, {ci_syco['ci_upper']*100:.1f}]}}",
            "Truthfulness": f"{ci_truth['mean']*100:.1f} \\small{{[{ci_truth['ci_lower']*100:.1f}, {ci_truth['ci_upper']*100:.1f}]}}",
            "Pushback Rate": f"{ci_corr['mean']*100:.1f} \\small{{[{ci_corr['ci_lower']*100:.1f}, {ci_corr['ci_upper']*100:.1f}]}}"
        })

    df_t1 = pd.DataFrame(t1_rows)
    t1_tex_path = os.path.join(out_dir, "table1_main_results.tex")
    with open(t1_tex_path, "w", encoding="utf-8") as f:
        f.write("% SycophancyRAGBench Main Empirical Results\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write(_df_to_latex(df_t1, escape=False))
        f.write("\\caption{Primary evaluation results across models and retrieval conditions. Rates reported in \\% with 95\\% bootstrap confidence intervals.}\n")
        f.write("\\label{tab:main_results}\n\\end{table*}\n")

    # Table 2: McNemar Statistical Significance
    t2_tex_path = os.path.join(out_dir, "table2_mcnemar_tests.tex")
    if mcnemar_rows:
        df_mc = pd.DataFrame(mcnemar_rows)[[
            "model", "outcome", "baseline_setting", "compare_setting",
            "baseline_rate", "compare_rate", "n10_fixed", "n01_worsened", "pvalue", "pvalue_fdr"
        ]]
        with open(t2_tex_path, "w", encoding="utf-8") as f:
            f.write("% Paired McNemar Tests with FDR Correction\n")
            f.write("\\begin{table}[t]\n\\centering\\small\n")
            f.write(_df_to_latex(df_mc, escape=True))
            f.write("\\caption{Paired McNemar's tests comparing baseline against augmented conditions with Benjamini-Hochberg FDR adjustments.}\n")
            f.write("\\label{tab:mcnemar}\n\\end{table}\n")

    # Table 3: CAEV Mitigation Comparison (Standard RAG vs CAEV)
    t3_tex_path = os.path.join(out_dir, "table3_caev_mitigation.tex")
    caev_compare = [s for s in ["rag_hybrid", "rag_hybrid_caev", "control_oracle", "oracle_caev"] if s in df_calc["setting"].unique()]
    if caev_compare:
        t3_rows = []
        for model in sorted(df_calc["model"].unique()):
            for setting in caev_compare:
                grp = df_calc[(df_calc["model"] == model) & (df_calc["setting"] == setting)]
                if len(grp) == 0:
                    continue
                b_grp = grp[grp["style"] != "neutral"]
                ci_syco = bootstrap_ci(b_grp["endorses_num"])
                ci_truth = bootstrap_ci(grp["truthful_num"])
                ci_corr = bootstrap_ci(b_grp["corrects_num"])
                ov_syco = b_grp[b_grp["endorses_num"] == 1]["evidence_overlap"].dropna() if "evidence_overlap" in b_grp.columns else pd.Series(dtype=float)
                t3_rows.append({
                    "Model": model.upper(),
                    "Condition": setting.replace("_", " ").title(),
                    "Sycophancy Rate": f"{ci_syco['mean']*100:.1f}",
                    "Truthfulness": f"{ci_truth['mean']*100:.1f}",
                    "Pushback Rate": f"{ci_corr['mean']*100:.1f}",
                    "Syco Overlap": f"{ov_syco.mean()*100:.1f}" if len(ov_syco) > 0 else "0.0"
                })
        df_t3 = pd.DataFrame(t3_rows)
        with open(t3_tex_path, "w", encoding="utf-8") as f:
            f.write("% CAEV Mitigation Ablation Table\n")
            f.write("\\begin{table}[t]\n\\centering\\small\n")
            f.write(_df_to_latex(df_t3, escape=False))
            f.write("\\caption{Impact of Conflict-Aware Epistemic Verification (CAEV) on residual sycophancy, truthfulness, pushback rate, and superficial faithfulness overlap.}\n")
            f.write("\\label{tab:caev_mitigation}\n\\end{table}\n")
        print(f"[Plots] Exported LaTeX tables to {out_dir}")
        return [t1_tex_path, t2_tex_path, t3_tex_path]

    print(f"[Plots] Exported LaTeX tables to {out_dir}")
    return [t1_tex_path, t2_tex_path]


def generate_all_publication_artifacts(df: pd.DataFrame, mcnemar_rows: List[Dict[str, Any]] = None):
    """Generate all figures and tables in a single call."""
    print("[Plots] Generating publication figures (300 DPI PNG & vector PDF)...")
    p1 = plot_sycophancy_by_model_and_condition(df)
    p2 = plot_evidence_user_conflict_breakdown(df)
    t = export_latex_tables(df, mcnemar_rows or [])
    print(f"[Plots] Generated Figures: {p1 + p2}")
    print(f"[Plots] Generated Tables: {t}")
