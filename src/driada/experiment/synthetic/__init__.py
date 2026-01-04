"""
Synthetic data generation for neural manifold analysis.

This module provides functions to generate synthetic neural data with realistic
properties for testing and demonstration purposes. It supports various manifold
types (circular, 2D spatial, 3D spatial) and mixed populations with both
manifold and feature-selective cells.

The original monolithic synthetic.py file has been split into focused modules
for better organization and maintainability. All functions remain available
through this __init__.py for backward compatibility.
"""

# Core utilities and calcium dynamics
from .core import (
    validate_peak_rate,
    generate_pseudo_calcium_signal,
    generate_pseudo_calcium_multisignal,
)

# Time series generation utilities
from .time_series import (
    generate_binary_time_series,
    generate_fbm_time_series,
    apply_poisson_to_binary_series,
    delete_one_islands,
    select_signal_roi,
    discretize_via_roi,
)

# Circular manifold (head direction cells)
from .manifold_circular import (
    generate_circular_random_walk,
    von_mises_tuning_curve,
    generate_circular_manifold_neurons,
    generate_circular_manifold_data,
    generate_circular_manifold_exp,
)

# 2D spatial manifold (place cells)
from .manifold_spatial_2d import (
    generate_2d_random_walk,
    gaussian_place_field,
    generate_2d_manifold_neurons,
    generate_2d_manifold_data,
    generate_2d_manifold_exp,
)

# 3D spatial manifold (3D place cells)
from .manifold_spatial_3d import (
    generate_3d_random_walk,
    gaussian_place_field_3d,
    generate_3d_manifold_neurons,
    generate_3d_manifold_data,
    generate_3d_manifold_exp,
)

# Mixed selectivity
from .mixed_selectivity import (
    generate_multiselectivity_patterns,
    generate_mixed_selective_signal,
    generate_synthetic_data_mixed_selectivity,
    generate_synthetic_exp_with_mixed_selectivity,
)

# High-level experiment generators
from .experiment_generators import (
    generate_synthetic_data,
    generate_synthetic_exp,
    generate_mixed_population_exp,
)

# Export all functions for backward compatibility
__all__ = [
    # Core utilities
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
    # Circular manifold
    "generate_circular_random_walk",
    "von_mises_tuning_curve",
    "generate_circular_manifold_neurons",
    "generate_circular_manifold_data",
    "generate_circular_manifold_exp",
    # 2D spatial manifold
    "generate_2d_random_walk",
    "gaussian_place_field",
    "generate_2d_manifold_neurons",
    "generate_2d_manifold_data",
    "generate_2d_manifold_exp",
    # 3D spatial manifold
    "generate_3d_random_walk",
    "gaussian_place_field_3d",
    "generate_3d_manifold_neurons",
    "generate_3d_manifold_data",
    "generate_3d_manifold_exp",
    # Mixed selectivity
    "generate_multiselectivity_patterns",
    "generate_mixed_selective_signal",
    "generate_synthetic_data_mixed_selectivity",
    "generate_synthetic_exp_with_mixed_selectivity",
    # High-level generators
    "generate_synthetic_data",
    "generate_synthetic_exp",
    "generate_mixed_population_exp",
]
