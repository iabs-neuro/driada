"""
DRIADA - Dimensionality Reduction for Integrated Activity Data

A library for single neuron and population-level analysis of inner workings
of intelligent systems, from brain neural recordings in vivo to RNNs.
"""

__version__ = "0.6.4"

# Core modules
from . import intense
from . import information
from . import experiment
from . import utils
from . import rsa
from . import network
from . import dim_reduction
from . import dimensionality
from . import integration
from . import gdrive

# Key classes
from .experiment import Experiment
from .information import TimeSeries, MultiTimeSeries
from .network import Network

# Main INTENSE pipeline functions
from .intense import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance,
)

# Common information theory functions
from .information import (
    get_mi,
    conditional_mi,
    interaction_information,
)

# Experiment utilities
from .experiment import (
    load_experiment,
    load_exp_from_aligned_data,
    save_exp_to_pickle,
    load_exp_from_pickle,
    generate_synthetic_exp,
    generate_circular_manifold_exp,
    generate_2d_manifold_exp,
    generate_3d_manifold_exp,
    generate_mixed_population_exp,
)

# Dimensionality reduction classes
from .dim_reduction import MVData, Embedding

# Dimensionality estimation
from .dimensionality import eff_dim, nn_dimension, pca_dimension

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
