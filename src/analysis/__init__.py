"""
Analysis subpackage for SycophancyRAGBench.
"""

from .stats import (
    paired_mcnemar_test,
    apply_fdr_correction,
    bootstrap_ci,
    compute_all_bootstrap_cis,
    compute_cohen_kappas
)
from .plots import (
    plot_sycophancy_by_model_and_condition,
    plot_evidence_user_conflict_breakdown,
    export_latex_tables,
    generate_all_publication_artifacts
)

__all__ = [
    "paired_mcnemar_test",
    "apply_fdr_correction",
    "bootstrap_ci",
    "compute_all_bootstrap_cis",
    "compute_cohen_kappas",
    "plot_sycophancy_by_model_and_condition",
    "plot_evidence_user_conflict_breakdown",
    "export_latex_tables",
    "generate_all_publication_artifacts",
]
