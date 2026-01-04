"""
Integration Module for DRIADA

This module provides components that integrate different parts of DRIADA,
bridging single-neuron analysis (INTENSE) with population-level analysis.
"""

from .manifold_analysis import get_functional_organization, compare_embeddings

__all__ = ["get_functional_organization", "compare_embeddings"]