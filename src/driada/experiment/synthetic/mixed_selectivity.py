"""
Mixed selectivity generation for synthetic neural data.

This module contains functions for generating synthetic neural data with mixed
selectivity, where neurons can respond to multiple features simultaneously.
"""

import numpy as np
import tqdm
from .core import generate_pseudo_calcium_signal
from .time_series import (
    generate_binary_time_series,
    generate_fbm_time_series,
    discretize_via_roi,
    delete_one_islands,
)
from ..exp_base import Experiment
from ...information.info_base import TimeSeries, aggregate_multiple_ts


def generate_multiselectivity_patterns(
    n_neurons,
    n_features,
    mode="random",
    selectivity_prob=0.3,
    multi_select_prob=0.4,
    weights_mode="random",
    seed=None,
):
    """
    Generate selectivity patterns for neurons with mixed selectivity support.

    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    n_features : int
        Number of features.
    mode : str, optional
        Pattern generation mode: 'random', 'structured'. Default: 'random'.
    selectivity_prob : float, optional
        Probability of a neuron being selective to any feature. Default: 0.3.
    multi_select_prob : float, optional
        Probability of selective neuron having mixed selectivity. Default: 0.4.
    weights_mode : str, optional
        Weight generation mode: 'random', 'dominant', 'equal'. Default: 'random'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    selectivity_matrix : ndarray
        Matrix of shape (n_features, n_neurons) with selectivity weights.
        Non-zero values indicate selectivity strength.
    """
    if seed is not None:
        np.random.seed(seed)

    selectivity_matrix = np.zeros((n_features, n_neurons))

    for j in range(n_neurons):
        # Decide if neuron is selective
        if np.random.rand() > selectivity_prob:
            continue

        # Decide if neuron has mixed selectivity
        if np.random.rand() < multi_select_prob:
            # Mixed selectivity: 2-3 features
            n_select = np.random.choice([2, 3], p=[0.7, 0.3])
        else:
            # Single selectivity
            n_select = 1

        # Choose features (ensure we don't try to select more than available)
        n_select = min(n_select, n_features)
        if n_select == 0:
            continue
        selected_features = np.random.choice(n_features, n_select, replace=False)

        # Assign weights
        if weights_mode == "equal":
            weights = np.ones(n_select) / n_select
        elif weights_mode == "dominant":
            # One feature dominates
            weights = np.random.dirichlet([5] + [1] * (n_select - 1))
        else:  # random
            weights = np.random.dirichlet(np.ones(n_select))

        # Set weights in matrix
        selectivity_matrix[selected_features, j] = weights

    return selectivity_matrix


def generate_mixed_selective_signal(
    features,
    weights,
    duration,
    sampling_rate,
    rate_0=0.1,
    rate_1=1.0,
    skip_prob=0.1,
    ampl_range=(0.5, 2),
    decay_time=2,
    noise_std=0.1,
    seed=None,
):
    """
    Generate neural signal selective to multiple features.

    Parameters
    ----------
    features : list of arrays
        List of feature time series.
    weights : array-like
        Weights for each feature contribution.
    duration : float
        Signal duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    Other parameters same as generate_pseudo_calcium_signal.

    Returns
    -------
    signal : array
        Generated calcium signal.
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(duration * sampling_rate)

    # Create stronger mixed selectivity signals using OR logic
    # Key insight: For detectability, we need clear differences between
    # active and inactive states for each feature

    # First, determine when each feature drives the neuron
    feature_activations = []
    for feat_idx, (feat, weight) in enumerate(zip(features, weights)):
        if weight == 0:
            continue

        # Check if already binary
        unique_vals = np.unique(feat)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            # Already binary
            binary_activation = feat.astype(int)
        else:
            # Use ROI-based discretization for continuous
            binary_activation = discretize_via_roi(
                feat, seed=seed + feat_idx if seed else None
            )
            binary_activation = binary_activation.astype(int)

        # Store weighted activation
        feature_activations.append((binary_activation, weight))

    # Generate events using OR logic but with feature-specific contributions
    # This ensures neurons respond differently to different features
    all_events = np.zeros(length)

    # Pre-generate random numbers for efficiency
    rand_vals = np.random.rand(length)

    for t in range(length):
        # Calculate contribution from each active feature
        # Use OR logic: if any feature is active, neuron can fire
        # But the firing probability depends on WHICH features are active
        total_contribution = 0
        active_features = []

        for activation, weight in feature_activations:
            if activation[t] > 0:
                active_features.append(weight)

        if active_features:
            # Combine contributions - use sum but cap at 1.0
            # This makes different feature combinations produce different rates
            total_contribution = min(sum(active_features), 1.0)
            firing_rate = rate_0 + total_contribution * (rate_1 - rate_0)
        else:
            # Baseline state
            firing_rate = rate_0

        # Generate spike event
        if rand_vals[t] < firing_rate / sampling_rate:
            all_events[t] = 1

    # Apply skip probability if needed
    if skip_prob > 0:
        all_events = delete_one_islands(all_events.astype(int), skip_prob).astype(float)

    # Generate calcium signal with stronger response
    # Use larger amplitude range for better detectability
    enhanced_ampl_range = (ampl_range[0] * 1.5, ampl_range[1] * 1.5)
    calcium_signal = generate_pseudo_calcium_signal(
        duration=duration,
        events=all_events,
        sampling_rate=sampling_rate,
        amplitude_range=enhanced_ampl_range,
        decay_time=decay_time,
        noise_std=noise_std,
    )

    return calcium_signal


def generate_synthetic_data_mixed_selectivity(
    features_dict,
    n_neurons,
    selectivity_matrix,
    duration=600,
    seed=42,
    sampling_rate=20.0,
    rate_0=0.1,
    rate_1=1.0,
    skip_prob=0.0,
    ampl_range=(0.5, 2),
    decay_time=2,
    noise_std=0.1,
    verbose=True,
):
    """
    Generate synthetic data with mixed selectivity support.

    Parameters
    ----------
    features_dict : dict
        Dictionary of feature_name: feature_array pairs.
    n_neurons : int
        Number of neurons to generate.
    selectivity_matrix : ndarray
        Matrix of shape (n_features, n_neurons) with selectivity weights.
    Other parameters same as generate_synthetic_data.

    Returns
    -------
    all_signals : ndarray
        Neural signals of shape (n_neurons, n_timepoints).
    ground_truth : ndarray
        Ground truth selectivity matrix (same as input selectivity_matrix).
    """
    feature_names = list(features_dict.keys())
    feature_arrays = [features_dict[name] for name in feature_names]

    if verbose:
        print("Generating mixed-selective neural signals...")

    all_signals = []

    for j in tqdm.tqdm(range(n_neurons)):
        # Get selectivity pattern for this neuron
        weights = selectivity_matrix[:, j]
        selective_features = np.where(weights > 0)[0]

        if len(selective_features) == 0:
            # Non-selective neuron - just noise
            signal = np.random.normal(0, noise_std, int(duration * sampling_rate))
        else:
            # Get features and weights
            selected_feat_arrays = [feature_arrays[i] for i in selective_features]
            selected_weights = weights[selective_features]

            # Generate mixed selective signal
            signal = generate_mixed_selective_signal(
                selected_feat_arrays,
                selected_weights,
                duration,
                sampling_rate,
                rate_0,
                rate_1,
                skip_prob,
                ampl_range,
                decay_time,
                noise_std,
                seed=seed + j if seed is not None else None,
            )

        all_signals.append(signal)

    return np.vstack(all_signals), selectivity_matrix


def generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=4,
    n_continuous_feats=4,
    n_neurons=50,
    n_multifeatures=2,
    create_discrete_pairs=True,
    selectivity_prob=0.8,
    multi_select_prob=0.5,
    weights_mode="random",
    duration=1200,
    seed=42,
    fps=20,
    verbose=True,
    name_convention="str",
    rate_0=0.1,
    rate_1=1.0,
    skip_prob=0.1,
    ampl_range=(0.5, 2),
    decay_time=2,
    noise_std=0.1,
):
    """
    Generate synthetic experiment with mixed selectivity and multifeatures.

    Parameters
    ----------
    n_discrete_feats : int
        Number of discrete features to generate.
    n_continuous_feats : int
        Number of continuous features to generate.
    n_neurons : int
        Number of neurons to generate.
    n_multifeatures : int
        Number of multifeature combinations to create.
    create_discrete_pairs : bool
        If True, create discretized versions of continuous features.
    selectivity_prob : float
        Probability of a neuron being selective.
    multi_select_prob : float
        Probability of mixed selectivity for selective neurons.
    weights_mode : str
        Weight generation mode: 'random', 'dominant', 'equal'.
    duration : float
        Experiment duration in seconds.
    seed : int
        Random seed.
    fps : float
        Sampling rate.
    verbose : bool
        Print progress messages.
    name_convention : str, optional
        Naming convention for multifeatures. Options:
        - 'str' (default): Use string keys like 'xy', 'speed_direction'
        - 'tuple': Use tuple keys like ('x', 'y'), ('speed', 'head_direction') [DEPRECATED]
    rate_0 : float, optional
        Baseline spike rate in Hz. Default: 0.1.
    rate_1 : float, optional
        Active spike rate in Hz. Default: 1.0.
    skip_prob : float, optional
        Probability of skipping spikes. Default: 0.1.
    ampl_range : tuple, optional
        Range of spike amplitudes. Default: (0.5, 2).
    decay_time : float, optional
        Calcium decay time constant in seconds. Default: 2.
    noise_std : float, optional
        Standard deviation of additive noise. Default: 0.1.

    Returns
    -------
    exp : Experiment
        Synthetic experiment with mixed selectivity.
    selectivity_info : dict
        Dictionary containing:
        - 'matrix': selectivity matrix
        - 'feature_names': ordered list of feature names
        - 'multifeature_map': multifeature definitions
    """
    if seed is not None:
        np.random.seed(seed)

    length = int(duration * fps)
    features_dict = {}

    # Generate discrete features
    if verbose:
        print(f"Generating {n_discrete_feats} discrete features...")
    for i in range(n_discrete_feats):
        # Calculate avg_islands to achieve ~5% active time
        target_active_fraction = 0.05  # 5% active time
        avg_duration_frames = int(0.5 * fps)  # 0.5 seconds per island
        total_active_frames = int(length * target_active_fraction)
        avg_islands = max(1, int(total_active_frames / avg_duration_frames))

        binary_series = generate_binary_time_series(
            length, avg_islands=avg_islands, avg_duration=avg_duration_frames
        )
        features_dict[f"d_feat_{i}"] = binary_series

    # Generate continuous features
    if verbose:
        print(f"Generating {n_continuous_feats} continuous features...")
    for i in range(n_continuous_feats):
        fbm_series = generate_fbm_time_series(length, hurst=0.3, seed=seed + i + 100)
        features_dict[f"c_feat_{i}"] = fbm_series

        # Create discretized pairs if requested
        if create_discrete_pairs:
            disc_series = discretize_via_roi(fbm_series, seed=seed + i + 200)
            features_dict[f"d_feat_from_c{i}"] = disc_series

    # Create multifeatures from existing continuous features
    multifeatures_to_create = []
    if n_multifeatures > 0 and n_continuous_feats >= 2:
        if verbose:
            print(f"Creating {n_multifeatures} multifeatures...")

        # Get all continuous features
        continuous_feats = [f for f in features_dict.keys() if "c_feat" in f]

        # Create multifeatures by pairing continuous features
        multi_idx = 0
        for i in range(0, min(n_multifeatures * 2, len(continuous_feats)), 2):
            if multi_idx >= n_multifeatures:
                break
            if i + 1 < len(continuous_feats):
                feat1 = continuous_feats[i]
                feat2 = continuous_feats[i + 1]

                if name_convention == "str":
                    # String key for the multifeature
                    mf_name = f"multi{multi_idx}"
                    multifeatures_to_create.append((mf_name, (feat1, feat2)))
                else:  # 'tuple' convention (deprecated)
                    # DEPRECATED: Tuple convention will be removed in v2.0
                    # Use name_convention='str' instead
                    # The tuple key is duplicated here for backward compatibility
                    import warnings
                    warnings.warn(
                        "Tuple convention for multifeatures is deprecated and will be removed in v2.0. "
                        "Use name_convention='str' instead.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    multifeatures_to_create.append(((feat1, feat2), (feat1, feat2)))

                multi_idx += 1

    # Generate selectivity patterns
    all_feature_names = list(features_dict.keys())
    n_total_features = len(all_feature_names)

    if verbose:
        print(f"Generating selectivity patterns for {n_neurons} neurons...")
    selectivity_matrix = generate_multiselectivity_patterns(
        n_neurons,
        n_total_features,
        selectivity_prob=selectivity_prob,
        multi_select_prob=multi_select_prob,
        weights_mode=weights_mode,
        seed=seed + 300,
    )

    # Generate neural signals
    calcium_signals, _ = generate_synthetic_data_mixed_selectivity(
        features_dict,
        n_neurons,
        selectivity_matrix,
        duration=duration,
        seed=seed + 400,
        sampling_rate=fps,
        rate_0=rate_0,
        rate_1=rate_1,
        skip_prob=skip_prob,
        ampl_range=ampl_range,
        decay_time=decay_time,
        noise_std=noise_std,
        verbose=verbose,
    )

    # Create TimeSeries objects
    dynamic_features = {}
    for feat_name, feat_data in features_dict.items():
        # Determine if discrete
        unique_vals = np.unique(feat_data)
        is_discrete = len(unique_vals) <= 10 or (
            len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})
        )
        dynamic_features[feat_name] = TimeSeries(feat_data, discrete=is_discrete)

    # Add multifeatures using aggregate_multiple_ts
    for mf_key, mf_components in multifeatures_to_create:
        # Get component TimeSeries
        component_ts = []
        for component_name in mf_components:
            if (
                component_name in dynamic_features
                and not dynamic_features[component_name].discrete
            ):
                component_ts.append(dynamic_features[component_name])

        # Create MultiTimeSeries if all components are continuous
        if len(component_ts) == len(mf_components):
            dynamic_features[mf_key] = aggregate_multiple_ts(*component_ts)

    # Create experiment
    exp = Experiment(
        "SyntheticMixedSelectivity",
        calcium_signals,
        None,
        {},
        {"fps": fps},
        dynamic_features,
        reconstruct_spikes=None,
    )

    # Prepare selectivity info
    # Create multifeature map for return value
    multifeature_map = {}
    for i, (mf_key, mf_components) in enumerate(multifeatures_to_create):
        if isinstance(mf_key, str):
            # For string convention: components tuple -> multifeature name
            multifeature_map[mf_components] = mf_key
        else:
            # For tuple convention: components tuple -> generated name
            multifeature_map[mf_key] = f"multifeature_{i}"

    selectivity_info = {
        "matrix": selectivity_matrix,
        "feature_names": all_feature_names,
        "multifeature_map": multifeature_map,
    }

    return exp, selectivity_info
