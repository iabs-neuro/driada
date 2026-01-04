"""
INTENSE: Information-Theoretic Evaluation of Neuronal Selectivity

A framework for analyzing neuronal selectivity to behavioral and environmental
variables using information theory, particularly mutual information analysis
with rigorous statistical testing.
"""

# Disentanglement analysis functions
from .disentanglement import (
    DEFAULT_MULTIFEATURE_MAP,
    create_multifeature_map,
    disentangle_all_selectivities,
    disentangle_pair,
    get_disentanglement_summary,
)

# Core computation functions
from .intense_base import (
    IntenseResults,
    calculate_optimal_delays,
    calculate_optimal_delays_parallel,
    compute_me_stats,
    get_calcium_feature_me_profile,
    get_multicomp_correction_thr,
    scan_pairs,
    scan_pairs_parallel,
    scan_pairs_router,
)

# Pipeline functions
from .pipelines import (
    compute_cell_cell_significance,
    compute_cell_feat_significance,
    compute_embedding_selectivity,
    compute_feat_feat_significance,
)

# Statistical functions
from .stats import (
    chebyshev_ineq,
    criterion1,
    criterion2,
    get_all_nonempty_pvals,
    get_distribution_function,
    get_gamma_p,
    get_lognormal_p,
    get_mask,
    get_mi_distr_pvalue,
    get_table_of_stats,
    merge_stage_significance,
    merge_stage_stats,
    stats_not_empty,
)

# Visualization functions
from .visual import (
    plot_disentanglement_heatmap,
    plot_disentanglement_summary,
    plot_neuron_feature_density,
    plot_neuron_feature_pair,
    plot_pc_activity,
    plot_selectivity_heatmap,
    plot_shadowed_groups,
)

__all__ = [
    # Core computation
    "calculate_optimal_delays",
    "calculate_optimal_delays_parallel",
    "get_calcium_feature_me_profile",
    "scan_pairs",
    "scan_pairs_parallel",
    "scan_pairs_router",
    "compute_me_stats",
    "get_multicomp_correction_thr",
    "IntenseResults",
    # Pipeline
    "compute_cell_feat_significance",
    "compute_feat_feat_significance",
    "compute_cell_cell_significance",
    "compute_embedding_selectivity",
    # Statistics
    "chebyshev_ineq",
    "get_lognormal_p",
    "get_gamma_p",
    "get_distribution_function",
    "get_mi_distr_pvalue",
    "get_mask",
    "stats_not_empty",
    "criterion1",
    "criterion2",
    "get_all_nonempty_pvals",
    "get_table_of_stats",
    "merge_stage_stats",
    "merge_stage_significance",
    # Visualization
    "plot_pc_activity",
    "plot_neuron_feature_density",
    "plot_neuron_feature_pair",
    "plot_shadowed_groups",
    "plot_disentanglement_heatmap",
    "plot_disentanglement_summary",
    "plot_selectivity_heatmap",
    # Disentanglement
    "disentangle_pair",
    "disentangle_all_selectivities",
    "create_multifeature_map",
    "get_disentanglement_summary",
    "DEFAULT_MULTIFEATURE_MAP",
]
