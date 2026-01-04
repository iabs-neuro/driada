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
from ...utils.data import check_positive, check_nonnegative, check_unit


def generate_multiselectivity_patterns(
    n_neurons,
    n_features,
    selectivity_prob=0.3,
    multi_select_prob=0.4,
    weights_mode="random",
    seed=None,
):
    """Generate selectivity patterns for neurons with mixed selectivity support.

    Parameters
    ----------
    n_neurons : int
        Number of neurons. Must be positive.
    n_features : int
        Number of features. Must be positive.
    selectivity_prob : float, optional
        Probability of a neuron being selective to any feature. Must be in [0, 1]. Default: 0.3.
    multi_select_prob : float, optional
        Probability of selective neuron having mixed selectivity. Must be in [0, 1]. Default: 0.4.
    weights_mode : str, optional
        Weight generation mode: 'random', 'dominant', 'equal'. Default: 'random'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_features, n_neurons)
        Matrix with selectivity weights. Non-zero values indicate selectivity strength.
        Each column represents a neuron, each row represents a feature.
        
    Raises
    ------
    ValueError
        If n_neurons or n_features are not positive.
        If selectivity_prob or multi_select_prob are not in [0, 1].
        If weights_mode is not one of 'random', 'dominant', 'equal'.
        
    Notes
    -----
    - Each neuron has selectivity_prob chance of being selective to any features.
    - Selective neurons have multi_select_prob chance of mixed selectivity (2-3 features)
      vs single selectivity (1 feature).
    - Mixed selectivity neurons select 2 or 3 features with probability [0.7, 0.3].
    - Weights are assigned using Dirichlet distribution for natural weight distributions.
    - Setting numpy random state for reproducibility when seed is provided.    """
    # Input validation
    check_positive(n_neurons=n_neurons, n_features=n_features)
    
    check_unit(selectivity_prob=selectivity_prob, multi_select_prob=multi_select_prob)
    
    valid_weights_modes = ['random', 'dominant', 'equal']
    if weights_mode not in valid_weights_modes:
        raise ValueError(f"weights_mode must be one of {valid_weights_modes}, got {weights_mode}")
    
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
    """Generate neural signal selective to multiple features.

    Parameters
    ----------
    features : list of array-like
        List of feature time series. Each must be array-like of same length.
    weights : array-like
        Weights for each feature contribution. Must have same length as features.
        Values should be non-negative.
    duration : float
        Signal duration in seconds. Must be positive.
    sampling_rate : float
        Sampling rate in Hz. Must be positive.
    rate_0 : float, optional
        Baseline spike rate in Hz. Must be non-negative. Default: 0.1.
    rate_1 : float, optional
        Maximum spike rate in Hz. Must be non-negative. Default: 1.0.
    skip_prob : float, optional
        Probability of skipping spike islands. Must be in [0, 1]. Default: 0.1.
    ampl_range : tuple, optional
        Range of spike amplitudes (min, max). Default: (0.5, 2).
    decay_time : float, optional
        Calcium decay time constant in seconds. Must be positive. Default: 2.
    noise_std : float, optional
        Standard deviation of additive noise. Must be non-negative. Default: 0.1.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (int(duration * sampling_rate),)
        Generated calcium signal.
        
    Raises
    ------
    ValueError
        If features and weights have different lengths.
        If duration or sampling_rate are not positive.
        If rate_0, rate_1, or noise_std are negative.
        If skip_prob is not in [0, 1].
        If decay_time is not positive.
        
    Notes
    -----
    - Uses OR logic for feature activation: neuron fires if ANY feature is active.
    - Firing rate is modulated by sum of active feature weights (capped at 1.0).
    - Continuous features are discretized using ROI-based method.
    - Setting numpy random state for reproducibility when seed is provided.    """
    # Input validation
    check_positive(duration=duration, sampling_rate=sampling_rate, decay_time=decay_time)
    check_nonnegative(rate_0=rate_0, rate_1=rate_1, noise_std=noise_std)
    
    # Check array lengths match
    features = list(features)  # Ensure it's a list
    weights = np.asarray(weights)
    if len(features) != len(weights):
        raise ValueError(f"features and weights must have same length: {len(features)} vs {len(weights)}")
    
    check_unit(skip_prob=skip_prob)
    
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

    # Generate calcium signal
    calcium_signal = generate_pseudo_calcium_signal(
        duration=duration,
        events=all_events,
        sampling_rate=sampling_rate,
        amplitude_range=ampl_range,
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
    """Generate synthetic data with mixed selectivity support.

    Parameters
    ----------
    features_dict : dict
        Dictionary of feature_name: feature_array pairs. All arrays must have
        length int(duration * sampling_rate).
    n_neurons : int
        Number of neurons to generate. Must be positive.
    selectivity_matrix : ndarray
        Matrix of shape (n_features, n_neurons) with selectivity weights.
        n_features must match len(features_dict).
    duration : float, optional
        Signal duration in seconds. Must be positive. Default: 600.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    sampling_rate : float, optional
        Sampling rate in Hz. Must be positive. Default: 20.0.
    rate_0 : float, optional
        Baseline spike rate in Hz. Must be non-negative. Default: 0.1.
    rate_1 : float, optional
        Maximum spike rate in Hz. Must be non-negative. Default: 1.0.
    skip_prob : float, optional
        Probability of skipping spike islands. Must be in [0, 1]. Default: 0.0.
    ampl_range : tuple, optional
        Range of spike amplitudes (min, max). Default: (0.5, 2).
    decay_time : float, optional
        Calcium decay time constant in seconds. Must be positive. Default: 2.
    noise_std : float, optional
        Standard deviation of additive noise. Must be non-negative. Default: 0.1.
    verbose : bool, optional
        Print progress messages. Default: True.

    Returns
    -------
    all_signals : ndarray of shape (n_neurons, int(duration * sampling_rate))
        Neural calcium signals.
    ground_truth : ndarray
        The input selectivity_matrix (returned for convenience).
        
    Raises
    ------
    ValueError
        If n_neurons is not positive.
        If selectivity_matrix shape doesn't match (len(features_dict), n_neurons).
        If duration or sampling_rate are not positive.
        If rate_0, rate_1, or noise_std are negative.
        If skip_prob is not in [0, 1].
        If decay_time is not positive.
        
    Notes
    -----
    - Non-selective neurons (all zero weights) generate pure noise.
    - Each neuron gets a unique seed: base_seed + neuron_index.
    - Progress is displayed using tqdm if verbose=True.
    - Setting numpy random state for reproducibility when seed is provided.    """
    # Input validation
    check_positive(n_neurons=n_neurons, duration=duration, sampling_rate=sampling_rate, decay_time=decay_time)
    check_nonnegative(rate_0=rate_0, rate_1=rate_1, noise_std=noise_std)
    
    check_unit(skip_prob=skip_prob)
    
    # Check features_dict
    if not isinstance(features_dict, dict):
        raise ValueError("features_dict must be a dictionary")
    
    feature_names = list(features_dict.keys())
    feature_arrays = [features_dict[name] for name in feature_names]
    
    # Check selectivity matrix shape
    selectivity_matrix = np.asarray(selectivity_matrix)
    expected_shape = (len(feature_names), n_neurons)
    if selectivity_matrix.shape != expected_shape:
        raise ValueError(
            f"selectivity_matrix shape {selectivity_matrix.shape} doesn't match "
            f"expected shape {expected_shape}"
        )

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
    hurst=0.3,
    target_active_fraction=0.05,
    avg_active_duration=0.5,
):
    """Generate synthetic experiment with mixed selectivity and multifeatures.

    Parameters
    ----------
    n_discrete_feats : int, optional
        Number of discrete features to generate. Must be non-negative. Default: 4.
    n_continuous_feats : int, optional
        Number of continuous features to generate. Must be non-negative. Default: 4.
    n_neurons : int, optional
        Number of neurons to generate. Must be positive. Default: 50.
    n_multifeatures : int, optional
        Number of multifeature combinations to create. Must be non-negative. Default: 2.
    create_discrete_pairs : bool, optional
        If True, create discretized versions of continuous features. Default: True.
    selectivity_prob : float, optional
        Probability of a neuron being selective. Must be in [0, 1]. Default: 0.8.
    multi_select_prob : float, optional
        Probability of mixed selectivity for selective neurons. Must be in [0, 1]. Default: 0.5.
    weights_mode : str, optional
        Weight generation mode: 'random', 'dominant', 'equal'. Default: 'random'.
    duration : float, optional
        Experiment duration in seconds. Must be positive. Default: 1200.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    fps : float, optional
        Sampling rate in Hz. Must be positive. Default: 20.
    verbose : bool, optional
        Print progress messages. Default: True.
    name_convention : str, optional
        Naming convention for multifeatures. Options:
        - 'str' (default): Use string keys like 'multi0', 'multi1'
        - 'tuple': Use tuple keys like ('c_feat_0', 'c_feat_1') [DEPRECATED]
    rate_0 : float, optional
        Baseline spike rate in Hz. Must be non-negative. Default: 0.1.
    rate_1 : float, optional
        Active spike rate in Hz. Must be non-negative. Default: 1.0.
    skip_prob : float, optional
        Probability of skipping spike islands. Must be in [0, 1]. Default: 0.1.
    ampl_range : tuple, optional
        Range of spike amplitudes (min, max). Default: (0.5, 2).
    decay_time : float, optional
        Calcium decay time constant in seconds. Must be positive. Default: 2.
    noise_std : float, optional
        Standard deviation of additive noise. Must be non-negative. Default: 0.1.
    hurst : float, optional
        Hurst parameter for fractional Brownian motion. Must be in (0, 1). Default: 0.3.
    target_active_fraction : float, optional
        Target fraction of time discrete features are active. Must be in (0, 1). Default: 0.05.
    avg_active_duration : float, optional
        Average duration of active periods in seconds. Must be positive. Default: 0.5.

    Returns
    -------
    exp : Experiment
        Synthetic experiment with mixed selectivity.
    selectivity_info : dict
        Dictionary containing:
        - 'matrix': ndarray of shape (n_features, n_neurons) - selectivity weights
        - 'feature_names': list - ordered feature names matching matrix rows
        - 'multifeature_map': dict - maps component tuples to multifeature names
        
    Raises
    ------
    ValueError
        If n_neurons is not positive.
        If n_discrete_feats, n_continuous_feats, or n_multifeatures are negative.
        If duration or fps are not positive.
        If probabilities are not in [0, 1].
        If name_convention is not 'str' or 'tuple'.
        If hurst is not in (0, 1).
        If target_active_fraction is not in (0, 1).
        If avg_active_duration is not positive.
        
    Notes
    -----
    - Discrete features have configurable active time and duration.
    - Continuous features use fractional Brownian motion with configurable Hurst parameter.
    - Multifeatures are created by pairing consecutive continuous features.
    - Each generation stage uses a different seed offset (+100, +200, etc.).
    - Setting numpy random state for reproducibility when seed is provided.    """
    # Input validation
    check_positive(n_neurons=n_neurons, duration=duration, fps=fps, 
                   decay_time=decay_time, avg_active_duration=avg_active_duration)
    check_nonnegative(n_discrete_feats=n_discrete_feats, n_continuous_feats=n_continuous_feats,
                      n_multifeatures=n_multifeatures, rate_0=rate_0, rate_1=rate_1, 
                      noise_std=noise_std)
    
    # Check probabilities and fractions
    check_unit(selectivity_prob=selectivity_prob, multi_select_prob=multi_select_prob, 
               skip_prob=skip_prob)
    check_unit(left_open=True, right_open=True, target_active_fraction=target_active_fraction,
               hurst=hurst)
    
    # Check other parameters
    valid_weights_modes = ['random', 'dominant', 'equal']
    if weights_mode not in valid_weights_modes:
        raise ValueError(f"weights_mode must be one of {valid_weights_modes}, got {weights_mode}")
    
    valid_name_conventions = ['str', 'tuple']
    if name_convention not in valid_name_conventions:
        raise ValueError(f"name_convention must be one of {valid_name_conventions}, got {name_convention}")
    
    if seed is not None:
        np.random.seed(seed)

    length = int(duration * fps)
    features_dict = {}

    # Generate discrete features
    if verbose:
        print(f"Generating {n_discrete_feats} discrete features...")
    for i in range(n_discrete_feats):
        # Calculate avg_islands to achieve target active time
        avg_duration_frames = int(avg_active_duration * fps)  # Convert to frames
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
        fbm_series = generate_fbm_time_series(length, hurst=hurst, seed=seed + i + 100)
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
