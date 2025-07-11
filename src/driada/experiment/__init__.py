"""
Experiment module for DRIADA.

This module provides the core Experiment class and utilities for loading,
saving, and generating synthetic experimental data.
"""

# Core experiment class
from .exp_base import Experiment

# Loading and saving functions
from .exp_build import (
    load_experiment,
    load_exp_from_aligned_data,
    save_exp_to_pickle,
    load_exp_from_pickle,
)

# Synthetic data generation
from .synthetic import (
    generate_synthetic_exp,
    generate_synthetic_exp_with_mixed_selectivity,
    generate_synthetic_data,
    generate_synthetic_data_mixed_selectivity,
    generate_pseudo_calcium_signal,
    generate_pseudo_calcium_multisignal,
    generate_mixed_selective_signal,
    generate_multiselectivity_patterns,
    generate_binary_time_series,
    generate_fbm_time_series,
)

# Neuron analysis
from .neuron import Neuron

# Wavelet analysis
from .wavelet_event_detection import (
    extract_wvt_events,
    events_from_trace,
    get_cwt_ridges,
)
from .wavelet_ridge import ridges_to_containers

__all__ = [
    # Core class
    "Experiment",
    # Loading/saving
    "load_experiment",
    "load_exp_from_aligned_data", 
    "save_exp_to_pickle",
    "load_exp_from_pickle",
    # Synthetic data
    "generate_synthetic_exp",
    "generate_synthetic_exp_with_mixed_selectivity",
    "generate_synthetic_data",
    "generate_synthetic_data_mixed_selectivity",
    "generate_pseudo_calcium_signal",
    "generate_pseudo_calcium_multisignal",
    "generate_mixed_selective_signal",
    "generate_multiselectivity_patterns",
    "generate_binary_time_series",
    "generate_fbm_time_series",
    # Neuron analysis
    "Neuron",
    # Wavelet analysis
    "extract_wvt_events",
    "events_from_trace",
    "get_cwt_ridges",
    "ridges_to_containers",
]
