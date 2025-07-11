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
    correlation_matrix_old,
    cross_correlation_matrix,
    norm_cross_corr,
    to_numpy_array,
    write_dict_to_hdf5,
    read_hdf5_to_dict,
)

# Matrix operations
from .matrix import (
    nearestPD,
    isPD,
)

# Neural data utilities
from .neural import (
    generate_pseudo_calcium_signal,
    generate_pseudo_calcium_multisignal,
)

# Plotting utilities
from .plot import (
    make_beautiful,
    create_default_figure,
    plot_mat,
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

# Repository utilities
from .repo import (
    clone_org_repo,
    reload_module,
    import_external_repositories,
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
    "correlation_matrix_old",
    "cross_correlation_matrix",
    "norm_cross_corr",
    "to_numpy_array",
    "write_dict_to_hdf5",
    "read_hdf5_to_dict",
    # Matrix operations
    "nearestPD",
    "isPD",
    # Neural data
    "generate_pseudo_calcium_signal",
    "generate_pseudo_calcium_multisignal",
    # Plotting
    "make_beautiful",
    "create_default_figure",
    "plot_mat",
    # Output
    "show_output",
    # Naming
    "construct_session_name",
    # GIF
    "erase_all",
    "save_image_series",
    "create_gif_from_image_series",
    # Repository
    "clone_org_repo",
    "reload_module",
    "import_external_repositories",
]