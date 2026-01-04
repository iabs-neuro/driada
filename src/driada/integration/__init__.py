"""
Integration Module for DRIADA

This module provides components that integrate different parts of DRIADA,
bridging single-neuron analysis (INTENSE) with population-level analysis.
"""

from .manifold_analysis import compare_embeddings, get_functional_organization

__all__ = ["get_functional_organization", "compare_embeddings"]
