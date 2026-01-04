"""
Representational Similarity Analysis (RSA) for DRIADA.

This module provides tools for computing and comparing representational
dissimilarity matrices (RDMs) from neural population data.
"""

from .core import (
    compute_rdm,
    compute_rdm_from_timeseries_labels,
    compute_rdm_from_trials,
    compare_rdms,
    bootstrap_rdm_comparison,
    compute_rdm_unified,
    rsa_compare,
)

from .integration import (
    compute_experiment_rdm,
    compute_mvdata_rdm,
    rsa_between_experiments,
)

from .visual import (
    plot_rdm,
    plot_rdm_comparison,
)

__all__ = [
    # Core functions
    "compute_rdm",
    "compute_rdm_from_timeseries_labels",
    "compute_rdm_from_trials",
    "compare_rdms",
    "bootstrap_rdm_comparison",
    "compute_rdm_unified",
    "rsa_compare",  # New simplified API
    # Integration
    "compute_experiment_rdm",
    "compute_mvdata_rdm",
    "rsa_between_experiments",
    # Visualization
    "plot_rdm",
    "plot_rdm_comparison",
]
