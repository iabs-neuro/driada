"""
Synthetic data generation for neural manifold analysis.

This module provides functions to generate synthetic neural data with realistic
properties for testing and demonstration purposes. It supports various manifold
types (circular, 2D spatial) and mixed populations with both manifold and
feature-selective cells.

Module Structure (hierarchical):
    core.py         -> Calcium dynamics foundation
    utils.py        -> Utility functions
    time_series.py  -> Time series + random walks
    tuning.py       -> Tuning curves (von Mises, Gaussian, sigmoid, threshold)
    generators.py   -> All experiment generators

The canonical generator is `generate_tuned_selectivity_exp()`. Other generators
are maintained for backward compatibility but delegate to the canonical
generator internally.

All functions remain available through this __init__.py for backward compatibility.
"""

# =============================================================================
# Core utilities and calcium dynamics
# =============================================================================
from .core import (
    DEFAULT_T_RISE,
    validate_peak_rate,
    generate_pseudo_calcium_signal,
    generate_pseudo_calcium_multisignal,
)

# =============================================================================
# Time series generation utilities (including random walks)
# =============================================================================
from .time_series import (
    generate_binary_time_series,
    generate_fbm_time_series,
    apply_poisson_to_binary_series,
    delete_one_islands,
    select_signal_roi,
    discretize_via_roi,
    # Random walks (consolidated from manifold_*.py)
    generate_circular_random_walk,
    generate_2d_random_walk,
)

# =============================================================================
# Tuning curves (consolidated into tuning.py)
# =============================================================================
from .tuning import (
    von_mises_tuning_curve,
    gaussian_place_field,
    sigmoid_tuning_curve,
    threshold_response,
    compute_speed_from_positions,
    compute_head_direction_from_positions,
    combine_responses,
)

# =============================================================================
# Generators (consolidated into generators.py)
# =============================================================================
from .generators import (
    # Canonical generator
    generate_tuned_selectivity_exp,
    ground_truth_to_selectivity_matrix,
    # Mixed selectivity
    generate_multiselectivity_patterns,
    generate_synthetic_exp_with_mixed_selectivity,
    # Circular manifold (head direction cells)
    generate_circular_manifold_neurons,
    generate_circular_manifold_data,
    generate_circular_manifold_exp,
    # 2D spatial manifold (place cells)
    generate_2d_manifold_neurons,
    generate_2d_manifold_data,
    generate_2d_manifold_exp,
    # Legacy/convenience wrappers
    generate_synthetic_data,
    generate_synthetic_exp,
    generate_mixed_population_exp,
)

# =============================================================================
# Export all functions for backward compatibility
# =============================================================================
__all__ = [
    # Core utilities
    "DEFAULT_T_RISE",
    "validate_peak_rate",
    "generate_pseudo_calcium_signal",
    "generate_pseudo_calcium_multisignal",
    # Time series utilities
    "generate_binary_time_series",
    "generate_fbm_time_series",
    "apply_poisson_to_binary_series",
    "delete_one_islands",
    "select_signal_roi",
    "discretize_via_roi",
    # Random walks
    "generate_circular_random_walk",
    "generate_2d_random_walk",
    # Tuning curves
    "von_mises_tuning_curve",
    "gaussian_place_field",
    "sigmoid_tuning_curve",
    "threshold_response",
    "compute_speed_from_positions",
    "compute_head_direction_from_positions",
    "combine_responses",
    # Canonical generator
    "generate_tuned_selectivity_exp",
    "ground_truth_to_selectivity_matrix",
    # Mixed selectivity
    "generate_multiselectivity_patterns",
    "generate_synthetic_exp_with_mixed_selectivity",
    # Circular manifold
    "generate_circular_manifold_neurons",
    "generate_circular_manifold_data",
    "generate_circular_manifold_exp",
    # 2D spatial manifold
    "generate_2d_manifold_neurons",
    "generate_2d_manifold_data",
    "generate_2d_manifold_exp",
    # Legacy/convenience wrappers
    "generate_synthetic_data",
    "generate_synthetic_exp",
    "generate_mixed_population_exp",
]
