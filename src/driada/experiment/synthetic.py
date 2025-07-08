import numpy as np
from fbm import FBM
import itertools
import tqdm
from .exp_base import *
from ..information.info_base import TimeSeries, MultiTimeSeries, aggregate_multiple_ts


def generate_pseudo_calcium_multisignal(n,
                                        events=None,
                                        duration=600,
                                        sampling_rate=20,
                                        event_rate=0.2,
                                        amplitude_range=(0.5,2),
                                        decay_time=2,
                                        noise_std=0.1):
    sigs = []
    for i in range(n):
        local_events = None
        if events is not None:
            local_events = events[i, :]

        sig = generate_pseudo_calcium_signal(events=local_events,
                                             duration=duration,
                                             sampling_rate=sampling_rate,
                                             event_rate=event_rate,
                                             amplitude_range=amplitude_range,
                                             decay_time=decay_time,
                                             noise_std=noise_std)
        sigs.append(sig)

    return np.vstack(sigs)


def generate_pseudo_calcium_signal(events=None,
                                   duration=600,
                                   sampling_rate=20.0,
                                   event_rate=0.2,
                                   amplitude_range=(0.5,2),
                                   decay_time=2,
                                   noise_std=0.1):

    """
    Generate a pseudo-calcium imaging signal with noise.

    Parameters:
    - duration: Total duration of the signal in seconds.
    - sampling_rate: Sampling rate in Hz.
    - event_rate: Average rate of calcium events per second.
    - amplitude_range: Tuple of (min, max) for the amplitude of calcium events.
    - decay_time: Time constant for the decay of calcium events in seconds.
    - noise_std: Standard deviation of the Gaussian noise to be added.

    Returns:
    - signal: Numpy array representing the pseudo-calcium signal.
    """

    if events is None:
        # Calculate number of samples
        num_samples = int(duration * sampling_rate)

        # Generate calcium events
        num_events = np.random.poisson(event_rate * duration)
        event_times = np.random.uniform(0, duration, num_events)
        event_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], num_events)

    else:
        num_samples = len(events)
        event_times = np.where(events>0)[0]
        event_amplitudes = events[event_times]

    # Initialize the signal with zeros
    signal = np.zeros(num_samples)

    # Add calcium events to the signal
    for t, a in zip(event_times, event_amplitudes):
        if events is None:
            event_index = int(t * sampling_rate)
        else:
            event_index = int(t)

        decay = np.exp(-np.arange(num_samples - event_index) / (decay_time * sampling_rate))
        signal[event_index:] += a * decay

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, num_samples)
    signal += noise

    return signal


def generate_binary_time_series(length, avg_islands, avg_duration):
    series = np.zeros(length, dtype=int)
    islands_count = 0
    current_state = 0  # 0 for off, 1 for on
    position = 0

    while position < length:
        if current_state == 0:
            # When off, decide how long to stay off based on desired number of islands
            # Lower avg_islands means longer off periods to ensure fewer islands
            off_duration = max(1, int(np.random.exponential(length / (avg_islands * 2))))
            duration = min(off_duration, length - position)
        else:
            # When on, stay on for the average duration +/- some randomness
            duration = max(1, int(np.random.normal(avg_duration, avg_duration / 2)))
            islands_count += 1

        # Ensure we don't go past the series length
        duration = min(duration, length - position)

        # Fill the series with the current state
        series[position:position + duration] = current_state

        # Switch state
        current_state = 1 - current_state
        position += duration

    # Adjust series to match the desired number of islands
    actual_islands = sum(1 for value, group in itertools.groupby(series) if value == 1)
    while actual_islands != avg_islands:
        if actual_islands < avg_islands:
            # If we have too few islands, turn on a random '0' to create a new island
            zero_positions = np.where(series == 0)[0]
            if len(zero_positions) > 0:
                turn_on = np.random.choice(zero_positions)
                series[turn_on] = 1
                actual_islands += 1
        else:
            # If we have too many islands, turn off a random '1' to merge islands
            one_positions = np.where(series == 1)[0]
            if len(one_positions) > 1:
                turn_off = np.random.choice(one_positions)
                series[turn_off] = 0
                actual_islands -= 1

    return series


def apply_poisson_to_binary_series(binary_series, rate_0, rate_1):
    length = len(binary_series)
    poisson_series = np.zeros(length, dtype=int)

    current_pos = 0
    for value, group in itertools.groupby(binary_series):
        run_length = len(list(group))
        if value == 0:
            poisson_series[current_pos:current_pos + run_length] = np.random.poisson(rate_0, run_length)
        else:
            poisson_series[current_pos:current_pos + run_length] = np.random.poisson(rate_1, run_length)
        current_pos += run_length

    return poisson_series


def generate_binary_time_series(length, avg_islands, avg_duration):
    series = np.zeros(length, dtype=int)
    current_state = 0
    position = 0

    while position < length:
        if current_state == 0:
            off_duration = max(1, int(np.random.exponential(length / (avg_islands * 2))))
            duration = min(off_duration, length - position)
        else:
            duration = max(1, int(np.random.normal(avg_duration, avg_duration / 2)))

        duration = min(duration, length - position)
        series[position:position + duration] = current_state
        current_state = 1 - current_state
        position += duration

    return series


from itertools import groupby
import numpy as np


def delete_one_islands(binary_ts, probability):
    # Ensure binary_ts is binary
    if not np.all(np.isin(binary_ts, [0, 1])):
        raise ValueError("binary_ts must be binary (0s and 1s)")

    # Create a copy of the input array
    result = binary_ts.copy()

    # Identify islands of 1s using groupby
    start = 0
    for key, group in groupby(binary_ts):
        length = sum(1 for _ in group)  # Count elements in the group
        if key == 1 and np.random.random() < probability:
            result[start:start + length] = 0
        start += length

    return result


def generate_fbm_time_series(length, hurst, seed=None):
    if seed is not None:
        np.random.seed(seed)

    f = FBM(n=length-1, hurst=hurst, length=1.0, method='daviesharte')
    fbm_series = f.fbm()

    return fbm_series


def select_signal_roi(values, seed=42):
    mean = np.mean(values)
    std = np.std(values)

    np.random.seed(seed)
    # Select random location within mean ± 2*std
    loc = np.random.uniform(mean - 1.5 * std, mean + 1.5 * std)

    # Define borders
    lower_border = loc - 0.5 * std
    upper_border = loc + 0.5 * std

    return loc, lower_border, upper_border


def generate_synthetic_data(nfeats, nneurons, ftype='c', duration=600, seed=42, sampling_rate=20.0,
                            rate_0=0.1, rate_1=1.0, skip_prob=0.0, hurst=0.5, ampl_range=(0.5, 2), decay_time=2,
                            avg_islands=10, avg_duration=5, noise_std=0.1, verbose=True):
    gt = np.zeros((nfeats, nneurons))
    length = int(duration * sampling_rate)

    print('Generating features...')
    all_feats = []
    for i in tqdm.tqdm(np.arange(nfeats)):
        if ftype == 'c':
            # Generate the series
            fbm_series = generate_fbm_time_series(length, hurst, seed=seed)
            all_feats.append(fbm_series)

        elif ftype == 'd':
            # Generate binary series
            binary_series = generate_binary_time_series(length, avg_islands, avg_duration * sampling_rate)
            all_feats.append(binary_series)

        else:
            raise ValueError('unknown feature flag')

        seed += 1  # save reproducibility, but break degeneracy

    print('Generating signals...')
    fois = np.random.choice(np.arange(nfeats), size=nneurons)
    gt[fois, np.arange(nneurons)] = 1  # add info about ground truth feature-signal connections
    all_signals = []

    for j in tqdm.tqdm(np.arange(nneurons)):
        foi = fois[j]
        if ftype == 'c':
            csignal = all_feats[foi]
            loc, lower_border, upper_border = select_signal_roi(csignal, seed=seed)
            # Generate binary series from a continuous one
            binary_series = np.zeros(length)
            binary_series[np.where((csignal >= lower_border) & (csignal <= upper_border))] = 1

        elif ftype == 'd':
            binary_series = all_feats[foi]

        else:
            raise ValueError('unknown feature flag')

        # randomly skip some on periods
        mod_binary_series = delete_one_islands(binary_series, skip_prob)

        # Apply Poisson process
        poisson_series = apply_poisson_to_binary_series(mod_binary_series,
                                                        rate_0 / sampling_rate,
                                                        rate_1 / sampling_rate)

        # Generate pseudo-calcium
        pseudo_calcium_signal = generate_pseudo_calcium_signal(duration=duration,
                                                               events=poisson_series,
                                                               sampling_rate=sampling_rate,
                                                               amplitude_range=ampl_range,
                                                               decay_time=decay_time,
                                                               noise_std=noise_std)

        all_signals.append(pseudo_calcium_signal)
        seed += 1  # save reproducibility, but break degeneracy

    return np.vstack(all_feats), np.vstack(all_signals), gt


def discretize_via_roi(continuous_signal, seed=None):
    """
    Discretize continuous signal using ROI (Region of Interest) selection method.
    This matches the discretization used in generate_synthetic_data.
    
    Parameters
    ----------
    continuous_signal : array-like
        Continuous signal to discretize.
    seed : int, optional
        Random seed for ROI selection reproducibility.
        
    Returns
    -------
    binary_signal : array
        Binary discretized signal (0s and 1s).
    roi_params : tuple
        (loc, lower_border, upper_border) - ROI parameters used.
    """
    loc, lower_border, upper_border = select_signal_roi(continuous_signal, seed=seed)
    binary_signal = np.zeros(len(continuous_signal))
    binary_signal[(continuous_signal >= lower_border) & (continuous_signal <= upper_border)] = 1
    return binary_signal.astype(int), (loc, lower_border, upper_border)


def generate_multiselectivity_patterns(n_neurons, n_features, mode='random', 
                                      selectivity_prob=0.3, multi_select_prob=0.4,
                                      weights_mode='random', seed=None):
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
        if weights_mode == 'equal':
            weights = np.ones(n_select) / n_select
        elif weights_mode == 'dominant':
            # One feature dominates
            weights = np.random.dirichlet([5] + [1] * (n_select - 1))
        else:  # random
            weights = np.random.dirichlet(np.ones(n_select))
            
        # Set weights in matrix
        selectivity_matrix[selected_features, j] = weights
    
    return selectivity_matrix


def generate_mixed_selective_signal(features, weights, duration, sampling_rate, 
                                   rate_0=0.1, rate_1=1.0, skip_prob=0.1,
                                   ampl_range=(0.5, 2), decay_time=2, noise_std=0.1,
                                   seed=None):
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
    combined_activation = np.zeros(length)
    
    # Combine feature activations
    for feat, weight in zip(features, weights):
        if weight == 0:
            continue
            
        # Check if already binary
        unique_vals = np.unique(feat)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            # Already binary
            binary_activation = feat.astype(float)
        else:
            # Use ROI-based discretization for continuous
            binary_activation, _ = discretize_via_roi(feat, seed=seed)
            binary_activation = binary_activation.astype(float)
            
        # Weight the activation
        combined_activation += weight * binary_activation
        if seed is not None:
            seed += 1
    
    # Threshold to get final binary activation
    threshold = np.random.uniform(0.3, 0.7)  # Flexible threshold
    final_activation = (combined_activation >= threshold).astype(int)
    
    # Add stochasticity
    mod_activation = delete_one_islands(final_activation, skip_prob)
    
    # Generate Poisson events
    poisson_series = apply_poisson_to_binary_series(mod_activation,
                                                    rate_0 / sampling_rate,
                                                    rate_1 / sampling_rate)
    
    # Generate calcium signal
    calcium_signal = generate_pseudo_calcium_signal(duration=duration,
                                                    events=poisson_series,
                                                    sampling_rate=sampling_rate,
                                                    amplitude_range=ampl_range,
                                                    decay_time=decay_time,
                                                    noise_std=noise_std)
    
    return calcium_signal


def generate_synthetic_data_mixed_selectivity(features_dict, n_neurons, selectivity_matrix,
                                             duration=600, seed=42, sampling_rate=20.0,
                                             rate_0=0.1, rate_1=1.0, skip_prob=0.0,
                                             ampl_range=(0.5, 2), decay_time=2, noise_std=0.1,
                                             verbose=True):
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
        print('Generating mixed-selective neural signals...')
        
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
                selected_feat_arrays, selected_weights,
                duration, sampling_rate,
                rate_0, rate_1, skip_prob,
                ampl_range, decay_time, noise_std,
                seed=seed + j if seed is not None else None
            )
            
        all_signals.append(signal)
    
    return np.vstack(all_signals), selectivity_matrix


def generate_synthetic_exp_with_mixed_selectivity(n_discrete_feats=4, n_continuous_feats=4, 
                                                  n_neurons=50, n_multifeatures=2,
                                                  create_discrete_pairs=True,
                                                  selectivity_prob=0.8, multi_select_prob=0.5,
                                                  weights_mode='random', duration=1200,
                                                  seed=42, fps=20, verbose=True):
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
        print(f'Generating {n_discrete_feats} discrete features...')
    for i in range(n_discrete_feats):
        binary_series = generate_binary_time_series(length, avg_islands=10, 
                                                   avg_duration=int(5 * fps))
        features_dict[f'd_feat_{i}'] = binary_series
    
    # Generate continuous features
    if verbose:
        print(f'Generating {n_continuous_feats} continuous features...')
    for i in range(n_continuous_feats):
        fbm_series = generate_fbm_time_series(length, hurst=0.3, seed=seed + i + 100)
        features_dict[f'c_feat_{i}'] = fbm_series
        
        # Create discretized pairs if requested
        if create_discrete_pairs:
            disc_series, _ = discretize_via_roi(fbm_series, seed=seed + i + 200)
            features_dict[f'd_feat_from_c{i}'] = disc_series
    
    # Create multifeatures (e.g., place fields from x,y)
    multifeature_map = {}
    if n_multifeatures > 0 and n_continuous_feats >= 2:
        if verbose:
            print(f'Creating {n_multifeatures} multifeatures...')
        # Create spatial features
        if n_multifeatures >= 1:
            features_dict['x'] = features_dict['c_feat_0']
            features_dict['y'] = features_dict['c_feat_1']
            multifeature_map[('x', 'y')] = 'place'
        # Create additional multifeatures if requested
        if n_multifeatures >= 2 and n_continuous_feats >= 4:
            features_dict['speed'] = np.abs(features_dict['c_feat_2'])
            features_dict['head_direction'] = features_dict['c_feat_3']
            multifeature_map[('speed', 'head_direction')] = 'locomotion'
    
    # Generate selectivity patterns
    all_feature_names = list(features_dict.keys())
    n_total_features = len(all_feature_names)
    
    if verbose:
        print(f'Generating selectivity patterns for {n_neurons} neurons...')
    selectivity_matrix = generate_multiselectivity_patterns(
        n_neurons, n_total_features, 
        selectivity_prob=selectivity_prob,
        multi_select_prob=multi_select_prob,
        weights_mode=weights_mode,
        seed=seed + 300
    )
    
    # Generate neural signals
    calcium_signals, _ = generate_synthetic_data_mixed_selectivity(
        features_dict, n_neurons, selectivity_matrix,
        duration=duration, seed=seed + 400, sampling_rate=fps,
        rate_0=0.1, rate_1=1.0, skip_prob=0.1,
        ampl_range=(0.5, 2), decay_time=2, noise_std=0.1,
        verbose=verbose
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
    for mf_tuple, mf_name in multifeature_map.items():
        # Get component TimeSeries
        component_ts = []
        for component_name in mf_tuple:
            if component_name in dynamic_features and not dynamic_features[component_name].discrete:
                component_ts.append(dynamic_features[component_name])
        
        # Create MultiTimeSeries if all components are continuous
        if len(component_ts) == len(mf_tuple):
            dynamic_features[mf_tuple] = aggregate_multiple_ts(*component_ts)
    
    # Create experiment
    exp = Experiment('SyntheticMixedSelectivity',
                     calcium_signals,
                     None,
                     {},
                     {'fps': fps},
                     dynamic_features,
                     reconstruct_spikes=None)
    
    # Prepare selectivity info
    selectivity_info = {
        'matrix': selectivity_matrix,
        'feature_names': all_feature_names,
        'multifeature_map': multifeature_map
    }
    
    return exp, selectivity_info


def generate_synthetic_exp(n_dfeats=20, n_cfeats=20, nneurons=500, seed=0, fps=20):
    # Split neurons between those responding to discrete and continuous features
    # For odd numbers, give the extra neuron to the first group
    n_neurons_discrete = (nneurons + 1) // 2
    n_neurons_continuous = nneurons // 2
    
    dfeats, calcium1, gt = generate_synthetic_data(n_dfeats,
                                                   n_neurons_discrete,
                                                   duration=1200,
                                                   hurst=0.3,
                                                   ftype='d',
                                                   seed=seed,
                                                   rate_0=0.1,
                                                   rate_1=1.0,
                                                   skip_prob=0.1,
                                                   noise_std=0.1,
                                                   sampling_rate=fps)

    cfeats, calcium2, gt2 = generate_synthetic_data(n_dfeats,
                                                    n_neurons_continuous,
                                                    duration=1200,
                                                    hurst=0.3,
                                                    ftype='c',
                                                    seed=seed,
                                                    rate_0=0.1,
                                                    rate_1=1.0,
                                                    skip_prob=0.1,
                                                    noise_std=0.1,
                                                    sampling_rate=fps)

    discr_ts = {f'd_feat_{i}': TimeSeries(dfeats[i, :], discrete=True) for i in range(len(dfeats))}
    cont_ts = {f'c_feat_{i}': TimeSeries(cfeats[i, :], discrete=False) for i in range(len(cfeats))}

    exp = Experiment('Synthetic',
                     np.vstack([calcium1, calcium2]),
                     None,
                     {},
                     {'fps': fps},
                     {**discr_ts, **cont_ts},
                     reconstruct_spikes=None)

    return exp
