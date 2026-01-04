"""
Utility functions for DRIADA.

This module provides various utility functions for data manipulation,
plotting, matrix operations, and other common operations.
"""

# Data manipulation utilities
from .data import (
    add_names_to_nested_dict,
    correlation_matrix,
    cross_correlation_matrix,
    get_hash,
    nested_dict_to_seq_of_tables,
    norm_cross_corr,
    phase_synchrony,
    populate_nested_dict,
    read_hdf5_to_dict,
    rescale,
    retrieve_relevant_from_nested_dict,
    to_numpy_array,
    write_dict_to_hdf5,
)

# GIF utilities
from .gif import (
    create_gif_from_image_series,
    erase_all,
    save_image_series,
)

# Matrix operations
from .matrix import (
    is_positive_definite,
    nearestPD,
)

# Naming utilities
from .naming import construct_session_name

# Neural data utilities
from .neural import (
    generate_pseudo_calcium_multisignal,
    generate_pseudo_calcium_signal,
)

# Output utilities
from .output import show_output

# Plotting utilities
from .plot import (
    create_default_figure,
    make_beautiful,
    plot_mat,
)

# Signal generation and analysis utilities
from .signals import (
    adaptive_filter_signals,
    approximate_entropy,
    brownian,
    filter_1d_timeseries,
    filter_signals,
)

# Spatial analysis utilities
from .spatial import (
    analyze_spatial_coding,
    compute_occupancy_map,
    compute_rate_map,
    compute_spatial_decoding_accuracy,
    compute_spatial_information,
    compute_spatial_information_rate,
    compute_spatial_metrics,
    extract_place_fields,
    filter_by_speed,
)

# Advanced visualization utilities
from .visual import (
    DEFAULT_DPI,
    plot_component_interpretation,
    plot_component_selectivity_heatmap,
    plot_embedding_comparison,
    plot_embeddings_grid,
    plot_neuron_selectivity_summary,
    plot_trajectories,
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
    "write_dict_to_hdf5",
    "read_hdf5_to_dict",
    # Matrix operations
    "nearestPD",
    "is_positive_definite",
    # Neural data
    "generate_pseudo_calcium_signal",
    "generate_pseudo_calcium_multisignal",
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
    "filter_by_speed",
    "analyze_spatial_coding",
    "compute_spatial_metrics",
    # Signal generation and analysis
    "brownian",
    "approximate_entropy",
    "filter_1d_timeseries",
    "filter_signals",
    "adaptive_filter_signals",
]
