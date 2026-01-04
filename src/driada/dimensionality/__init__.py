"""
Dimensionality estimation module for DRIADA.

This module provides various methods for estimating the intrinsic dimensionality
of datasets, including both linear and nonlinear approaches.
"""

from .effective import eff_dim
from .intrinsic import correlation_dimension, geodesic_dimension, nn_dimension
from .linear import effective_rank, pca_dimension, pca_dimension_profile

__all__ = [
    # Effective dimensionality
    "eff_dim",
    # Intrinsic dimensionality
    "nn_dimension",
    "correlation_dimension",
    "geodesic_dimension",
    # Linear dimensionality
    "pca_dimension",
    "pca_dimension_profile",
    "effective_rank",
]
