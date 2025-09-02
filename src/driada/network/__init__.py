"""
Network Analysis Module for DRIADA

This module provides tools for analyzing functional networks in neural data.
"""

# Core class
from .net_base import Network

# Graph utilities
from .graph_utils import (
    get_giant_cc_from_graph,
    get_giant_scc_from_graph,
    remove_selfloops_from_graph,
    remove_isolates_from_graph,
    remove_isolates_and_selfloops_from_graph,
    small_world_index,
)

# Matrix utilities
from .matrix_utils import (
    get_ccs_from_adj,
    get_sccs_from_adj,
    get_giant_cc_from_adj,
    get_giant_scc_from_adj,
    remove_selfloops_from_adj,
    remove_isolates_from_adj,
    get_symmetry_index,
    get_laplacian,
    get_norm_laplacian,
    get_rw_laplacian,
    get_trans_matrix,
)

# Spectral analysis
from .spectral import (
    free_entropy,
    q_entropy,
    spectral_entropy,
)

# Quantum-inspired methods
from .quantum import (
    get_density_matrix,
    renyi_divergence,
    js_divergence,
)

# Randomization
from .randomization import (
    adj_random_rewiring_iom_preserving,
    random_rewiring_IOM_preserving,
    randomize_graph,
)

# Drawing utilities
from .drawing import (
    draw_net,
    draw_degree_distr,
    draw_spectrum,
    draw_eigenvectors,
    show_mat,
    plot_lem_embedding,
)

__all__ = [
    # Core class
    "Network",
    # Graph utilities
    "get_giant_cc_from_graph",
    "get_giant_scc_from_graph",
    "remove_selfloops_from_graph",
    "remove_isolates_from_graph",
    "remove_isolates_and_selfloops_from_graph",
    "small_world_index",
    # Matrix utilities
    "get_ccs_from_adj",
    "get_sccs_from_adj",
    "get_giant_cc_from_adj",
    "get_giant_scc_from_adj",
    "remove_selfloops_from_adj",
    "remove_isolates_from_adj",
    "get_symmetry_index",
    "get_laplacian",
    "get_norm_laplacian",
    "get_rw_laplacian",
    "get_trans_matrix",
    # Spectral analysis
    "free_entropy",
    "q_entropy",
    "spectral_entropy",
    # Quantum methods
    "get_density_matrix",
    "renyi_divergence",
    "js_divergence",
    # Randomization
    "adj_random_rewiring_iom_preserving",
    "random_rewiring_IOM_preserving",
    "randomize_graph",
    # Drawing
    "draw_net",
    "draw_degree_distr",
    "draw_spectrum",
    "draw_eigenvectors",
    "show_mat",
    "plot_lem_embedding",
]