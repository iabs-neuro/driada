"""
Dimensionality Reduction Module for DRIADA

This module provides various dimensionality reduction algorithms and
utilities for analyzing high-dimensional neural data.
"""

from .data import MVData
from .dr_base import METHODS_DICT, DRMethod
from .embedding import Embedding
from .graph import ProximityGraph

# Import manifold metrics
from .manifold_metrics import (
    compute_distance_matrix,
    knn_preservation_rate,
    trustworthiness,
    continuity,
    geodesic_distance_correlation,
    stress,
    circular_structure_preservation,
    procrustes_analysis,
    manifold_preservation_score,
)

__all__ = [
    # Core classes
    "MVData",
    "METHODS_DICT",
    "DRMethod",
    "Embedding",
    "ProximityGraph",
    # Manifold metrics
    "compute_distance_matrix",
    "knn_preservation_rate",
    "trustworthiness",
    "continuity",
    "geodesic_distance_correlation",
    "stress",
    "circular_structure_preservation",
    "procrustes_analysis",
    "manifold_preservation_score",
]