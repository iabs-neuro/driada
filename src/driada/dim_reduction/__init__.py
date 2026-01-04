"""
Dimensionality Reduction Module for DRIADA

This module provides various dimensionality reduction algorithms and
utilities for analyzing high-dimensional neural data.
"""

from .data import MVData
from .dr_base import (
    METHODS_DICT,
    DRMethod,
    e_param_filter,
    g_param_filter,
    m_param_filter,
    merge_params_with_defaults,
)
from .embedding import Embedding
from .graph import ProximityGraph

# Import manifold metrics
from .manifold_metrics import (  # Reconstruction validation functions
    circular_distance,
    circular_structure_preservation,
    compute_decoding_accuracy,
    compute_distance_matrix,
    compute_embedding_alignment_metrics,
    compute_reconstruction_error,
    continuity,
    extract_angles_from_embedding,
    geodesic_distance_correlation,
    knn_preservation_rate,
    manifold_preservation_score,
    manifold_reconstruction_score,
    procrustes_analysis,
    stress,
    trustworthiness,
)
from .sequences import dr_sequence

__all__ = [
    # Core classes
    "MVData",
    "METHODS_DICT",
    "DRMethod",
    "Embedding",
    "ProximityGraph",
    # Functions
    "dr_sequence",
    "merge_params_with_defaults",
    "e_param_filter",
    "g_param_filter",
    "m_param_filter",
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
    # Reconstruction validation functions
    "circular_distance",
    "extract_angles_from_embedding",
    "compute_reconstruction_error",
    "compute_embedding_alignment_metrics",
    "compute_decoding_accuracy",
    "manifold_reconstruction_score",
]
