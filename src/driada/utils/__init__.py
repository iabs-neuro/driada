"""
Utility functions for DRIADA.

This module provides various utility functions for data manipulation,
plotting, matrix operations, and other common operations.
"""

# Data manipulation utilities
from .data import (
    populate_nested_dict,
    nested_dict_to_seq_of_tables,
    add_names_to_nested_dict,
    retrieve_relevant_from_nested_dict,
    rescale,
    get_hash,
    phase_synchrony,
    correlation_matrix,
    cross_correlation_matrix,
    norm_cross_corr,
    to_numpy_array,
)

# Matrix operations
from .matrix import (
    nearestPD,
    is_positive_definite,
)

# Plotting utilities
from .plot import (
    make_beautiful,
    create_default_figure,
    plot_mat,
)

# Advanced visualization utilities
from .visual import (
    plot_embedding_comparison,
    plot_trajectories,
    plot_component_interpretation,
    plot_embeddings_grid,
    plot_neuron_selectivity_summary,
    plot_component_selectivity_heatmap,
    DEFAULT_DPI,
)

# Output utilities
from .output import show_output

# Naming utilities
from .naming import construct_session_name

# GIF utilities
from .gif import (
    erase_all,
    save_image_series,
    create_gif_from_image_series,
)

# Spatial analysis utilities
from .spatial import (
    compute_occupancy_map,
    compute_rate_map,
    extract_place_fields,
    compute_spatial_information_rate,
    compute_spatial_decoding_accuracy,
    compute_spatial_information,
    compute_spatial_metrics,
)

# Signal generation and analysis utilities
from .signals import (
    brownian,
    approximate_entropy,
    filter_1d_timeseries,
    filter_signals,
    adaptive_filter_signals,
)

__all__ = [
    # Data manipulation
    "populate_nested_dict",
    "nested_dict_to_seq_of_tables",
    "add_names_to_nested_dict",
    "retrieve_relevant_from_nested_dict",
    "rescale",
    "get_hash",
    "phase_synchrony",
    "correlation_matrix",
    "cross_correlation_matrix",
    "norm_cross_corr",
    "to_numpy_array",
    # Matrix operations
    "nearestPD",
    "is_positive_definite",
    # Plotting
    "make_beautiful",
    "create_default_figure",
    "plot_mat",
    # Advanced visualization
    "plot_embedding_comparison",
    "plot_trajectories",
    "plot_component_interpretation",
    "plot_embeddings_grid",
    "plot_neuron_selectivity_summary",
    "plot_component_selectivity_heatmap",
    "DEFAULT_DPI",
    # Output
    "show_output",
    # Naming
    "construct_session_name",
    # GIF
    "erase_all",
    "save_image_series",
    "create_gif_from_image_series",
    # Spatial analysis
    "compute_occupancy_map",
    "compute_rate_map",
    "extract_place_fields",
    "compute_spatial_information_rate",
    "compute_spatial_decoding_accuracy",
    "compute_spatial_information",
    "compute_spatial_metrics",
    # Signal generation and analysis
    "brownian",
    "approximate_entropy",
    "filter_1d_timeseries",
    "filter_signals",
    "adaptive_filter_signals",
]
