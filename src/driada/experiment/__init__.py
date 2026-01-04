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
    load_demo_experiment,
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
    # Circular manifold functions
    generate_circular_random_walk,
    von_mises_tuning_curve,
    generate_circular_manifold_neurons,
    generate_circular_manifold_data,
    generate_circular_manifold_exp,
    # 2D spatial manifold functions
    generate_2d_random_walk,
    gaussian_place_field,
    generate_2d_manifold_neurons,
    generate_2d_manifold_data,
    generate_2d_manifold_exp,
    # 3D spatial manifold functions
    generate_3d_random_walk,
    gaussian_place_field_3d,
    generate_3d_manifold_neurons,
    generate_3d_manifold_data,
    generate_3d_manifold_exp,
    # Mixed population generator
    generate_mixed_population_exp,
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

# Spike reconstruction
from .spike_reconstruction import (
    reconstruct_spikes,
    wavelet_reconstruction,
    threshold_reconstruction,
)

__all__ = [
    # Core class
    "Experiment",
    # Loading/saving
    "load_experiment",
    "load_exp_from_aligned_data",
    "save_exp_to_pickle",
    "load_exp_from_pickle",
    "load_demo_experiment",
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
    # Circular manifold functions
    "generate_circular_random_walk",
    "von_mises_tuning_curve",
    "generate_circular_manifold_neurons",
    "generate_circular_manifold_data",
    "generate_circular_manifold_exp",
    # 2D spatial manifold functions
    "generate_2d_random_walk",
    "gaussian_place_field",
    "generate_2d_manifold_neurons",
    "generate_2d_manifold_data",
    "generate_2d_manifold_exp",
    # 3D spatial manifold functions
    "generate_3d_random_walk",
    "gaussian_place_field_3d",
    "generate_3d_manifold_neurons",
    "generate_3d_manifold_data",
    "generate_3d_manifold_exp",
    # Mixed population generator
    "generate_mixed_population_exp",
    # Neuron analysis
    "Neuron",
    # Wavelet analysis
    "extract_wvt_events",
    "events_from_trace",
    "get_cwt_ridges",
    "ridges_to_containers",
    # Spike reconstruction
    "reconstruct_spikes",
    "wavelet_reconstruction",
    "threshold_reconstruction",
]
