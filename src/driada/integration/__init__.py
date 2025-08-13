"""
Integration Module for DRIADA

This module provides components that integrate different parts of DRIADA,
bridging single-neuron analysis (INTENSE) with population-level analysis.
"""

from .selectivity_mapper import SelectivityManifoldMapper

__all__ = ["SelectivityManifoldMapper"]
