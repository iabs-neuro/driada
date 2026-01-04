"""
DRIADA - Dimensionality Reduction for Integrated Activity Data

A library for single neuron and population-level analysis of inner workings
of intelligent systems, from brain neural recordings in vivo to RNNs.
"""

__version__ = "0.6.6"

# Core modules
from . import (
    dim_reduction,
    dimensionality,
    experiment,
    gdrive,
    information,
    integration,
    intense,
    network,
    rsa,
    utils,
)

# Dimensionality reduction classes
from .dim_reduction import Embedding, MVData

# Dimensionality estimation
from .dimensionality import eff_dim, nn_dimension, pca_dimension

# Experiment utilities
# Key classes
from .experiment import (
    Experiment,
    generate_2d_manifold_exp,
    generate_3d_manifold_exp,
    generate_circular_manifold_exp,
    generate_mixed_population_exp,
    generate_synthetic_exp,
    load_exp_from_aligned_data,
    load_exp_from_pickle,
    load_experiment,
    save_exp_to_pickle,
)

# Common information theory functions
from .information import (
    MultiTimeSeries,
    TimeSeries,
    conditional_mi,
    get_mi,
    interaction_information,
)

# Main INTENSE pipeline functions
from .intense import (
    compute_cell_cell_significance,
    compute_cell_feat_significance,
    compute_feat_feat_significance,
)
from .network import Network

__all__ = [
    # Version
    "__version__",
    # Modules
    "intense",
    "information",
    "experiment",
    "utils",
    "rsa",
    "network",
    "dim_reduction",
    "dimensionality",
    "integration",
    "gdrive",
    # Core classes
    "Experiment",
    "TimeSeries",
    "MultiTimeSeries",
    "Network",
    # INTENSE pipelines
    "compute_cell_feat_significance",
    "compute_feat_feat_significance",
    "compute_cell_cell_significance",
    # Information theory
    "get_mi",
    "conditional_mi",
    "interaction_information",
    # Experiment utilities
    "load_experiment",
    "load_exp_from_aligned_data",
    "save_exp_to_pickle",
    "load_exp_from_pickle",
    "generate_synthetic_exp",
    "generate_circular_manifold_exp",
    "generate_2d_manifold_exp",
    "generate_3d_manifold_exp",
    "generate_mixed_population_exp",
    # Dimensionality reduction
    "MVData",
    "Embedding",
    # Dimensionality estimation
    "eff_dim",
    "nn_dimension",
    "pca_dimension",
]
