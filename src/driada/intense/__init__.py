"""
INTENSE: Information-Theoretic Evaluation of Neuronal Selectivity

A framework for analyzing neuronal selectivity to behavioral and environmental
variables using information theory, particularly mutual information analysis
with rigorous statistical testing.
"""

# Core computation functions
from .intense_base import (
    calculate_optimal_delays,
    calculate_optimal_delays_parallel,
    get_calcium_feature_me_profile,
    scan_pairs,
    scan_pairs_parallel,
    scan_pairs_router,
    compute_me_stats,
    get_multicomp_correction_thr,
    IntenseResults,
)

# Pipeline functions
from .pipelines import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance,
    compute_embedding_selectivity,
)

# Statistical functions
from .stats import (
    chebyshev_ineq,
    get_lognormal_p,
    get_gamma_p,
    get_distribution_function,
    get_mi_distr_pvalue,
    get_mask,
    stats_not_empty,
    criterion1,
    criterion2,
    get_all_nonempty_pvals,
    get_table_of_stats,
    merge_stage_stats,
    merge_stage_significance,
)

# Visualization functions
from .visual import (
    plot_pc_activity,
    plot_neuron_feature_density,
    plot_neuron_feature_pair,
    plot_shadowed_groups,
    plot_disentanglement_heatmap,
    plot_disentanglement_summary,
    plot_selectivity_heatmap,
)

# Disentanglement analysis functions
from .disentanglement import (
    disentangle_pair,
    disentangle_all_selectivities,
    create_multifeature_map,
    get_disentanglement_summary,
    DEFAULT_MULTIFEATURE_MAP,
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
