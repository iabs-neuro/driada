"""
DRIADA - Dimensionality Reduction for Integrated Activity Data

A library for single neuron and population-level analysis of inner workings
of intelligent systems, from brain neural recordings in vivo to RNNs.
"""

import os as _os
import sys as _sys

# Suppress Google API Python version warning (Python 3.10 EOL is Oct 2026)
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*Python version.*google.api_core.*")
del _warnings

__version__ = "1.0.0"

# =============================================================================
# Global configuration
# =============================================================================

# Joblib parallel backend for all driada parallel computations
# Can be set via DRIADA_PARALLEL_BACKEND environment variable
# Options: "loky" (true parallelism), "threading" (stable), "multiprocessing"
# Default: "threading" on Windows (loky has ~100x overhead), "loky" elsewhere
_default_backend = "threading" if _sys.platform == "win32" else "loky"
PARALLEL_BACKEND = _os.environ.get("DRIADA_PARALLEL_BACKEND", _default_backend)
del _os, _sys, _default_backend


def set_parallel_backend(backend: str):
    """Set the joblib parallel backend for all driada computations.

    Parameters
    ----------
    backend : str
        One of "loky", "threading", or "multiprocessing".
        - "loky": Default, true parallelism via separate processes
        - "threading": Stable, good for NumPy-heavy code (releases GIL)
        - "multiprocessing": Legacy process-based parallelism

    Notes
    -----
    If experiencing hangs on Windows or remote machines, try "threading".
    """
    global PARALLEL_BACKEND
    if backend not in ("loky", "threading", "multiprocessing"):
        raise ValueError(f"Invalid backend: {backend}. Choose from: loky, threading, multiprocessing")
    PARALLEL_BACKEND = backend

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
    generate_mixed_population_exp,
)

# Principled selectivity generator
from .experiment.synthetic import generate_tuned_selectivity_exp

# Dimensionality reduction classes
from .dim_reduction import MVData, Embedding

# Dimensionality estimation
from .dimensionality import eff_dim, nn_dimension, pca_dimension

__all__ = [
    # Version
    "__version__",
    # Configuration
    "PARALLEL_BACKEND",
    "set_parallel_backend",
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
    "generate_mixed_population_exp",
    "generate_tuned_selectivity_exp",
    # Dimensionality reduction
    "MVData",
    "Embedding",
    # Dimensionality estimation
    "eff_dim",
    "nn_dimension",
    "pca_dimension",
]
