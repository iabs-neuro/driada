"""
High-level experiment generators for synthetic neural data.

This module contains functions that generate complete synthetic experiments
by combining various types of neural data (manifold-based, feature-selective, mixed).
"""

import numpy as np
import tqdm
from .core import validate_peak_rate, generate_pseudo_calcium_signal, generate_pseudo_calcium_multisignal
from .time_series import (
    generate_binary_time_series, generate_fbm_time_series,
    select_signal_roi, delete_one_islands, apply_poisson_to_binary_series
)
from .manifold_circular import generate_circular_manifold_data
from .manifold_spatial_2d import generate_2d_manifold_data  
from .manifold_spatial_3d import generate_3d_manifold_data
from .mixed_selectivity import (
    generate_multiselectivity_patterns,
    generate_synthetic_data_mixed_selectivity
)
from ..exp_base import Experiment
from ...information.info_base import TimeSeries, MultiTimeSeries


def generate_synthetic_data(nfeats, nneurons, ftype='c', duration=600, seed=42, sampling_rate=20.0,
                            rate_0=0.1, rate_1=1.0, skip_prob=0.0, hurst=0.5, ampl_range=(0.5, 2), decay_time=2,
                            avg_islands=10, avg_duration=5, noise_std=0.1, verbose=True, pregenerated_features=None,
                            apply_random_neuron_shifts=False):
    """
    Generate synthetic neural data with feature-selective neurons.
    
    Parameters
    ----------
    nfeats : int
        Number of features.
    nneurons : int
        Number of neurons.
    ftype : str
        Feature type: 'c' for continuous, 'd' for discrete.
    duration : float
        Duration in seconds.
    seed : int
        Random seed.
    sampling_rate : float
        Sampling rate in Hz.
    rate_0 : float
        Baseline firing rate.
    rate_1 : float
        Active firing rate.
    skip_prob : float
        Probability of skipping islands.
    hurst : float
        Hurst parameter for FBM.
    ampl_range : tuple
        Amplitude range for calcium events.
    decay_time : float
        Calcium decay time.
    avg_islands : int
        Average number of islands for discrete features.
    avg_duration : int
        Average duration of islands.
    noise_std : float
        Noise standard deviation.
    verbose : bool
        Print progress.
    pregenerated_features : list, optional
        Use provided features instead of generating new ones.
    apply_random_neuron_shifts : bool
        Apply random shifts to break correlations.
        
    Returns
    -------
    features : ndarray
        Feature time series (nfeats x length).
    signals : ndarray
        Neural signals (nneurons x length).
    ground_truth : ndarray
        Ground truth matrix (nfeats x nneurons).
    """
    gt = np.zeros((nfeats, nneurons))
    length = int(duration * sampling_rate)
    
    # Handle edge case of 0 neurons
    if nneurons == 0:
        return np.array([]), np.array([]).reshape(0, length), gt

    # Use pregenerated features if provided, otherwise generate new ones
    if pregenerated_features is not None:
        print('Using pregenerated features...')
        all_feats = pregenerated_features
        if len(all_feats) != nfeats:
            raise ValueError(f'Number of pregenerated features ({len(all_feats)}) does not match nfeats ({nfeats})')
    else:
        print('Generating features...')
        all_feats = []
        for i in tqdm.tqdm(np.arange(nfeats)):
            if ftype == 'c':
                # Generate the series with unique seed for each feature
                feature_seed = seed + i if seed is not None else None
                fbm_series = generate_fbm_time_series(length, hurst, seed=feature_seed)
                all_feats.append(fbm_series)

            elif ftype == 'd':
                # Generate binary series
                binary_series = generate_binary_time_series(length, avg_islands, avg_duration * sampling_rate)
                all_feats.append(binary_series)

            else:
                raise ValueError('unknown feature flag')

    print('Generating signals...')
    if nfeats > 0:
        fois = np.random.choice(np.arange(nfeats), size=nneurons)
        gt[fois, np.arange(nneurons)] = 1  # add info about ground truth feature-signal connections
    else:
        # If no features, neurons won't be selective to any feature
        fois = np.full(nneurons, -1)  # Use -1 to indicate no feature selection
    all_signals = []

    for j in tqdm.tqdm(np.arange(nneurons)):
        foi = fois[j]
        
        # Handle case where there are no features
        if foi == -1 or nfeats == 0:
            # Generate random baseline activity
            binary_series = generate_binary_time_series(length, avg_islands // 2, avg_duration * sampling_rate // 2)
        elif ftype == 'c':
            csignal = all_feats[foi].copy()  # Make a copy to avoid modifying the original
            
            # Apply random per-neuron shift to break correlations
            if apply_random_neuron_shifts:
                # Apply a unique random shift for this neuron
                neuron_shift = np.random.randint(0, length)
                csignal = np.roll(csignal, neuron_shift)
                if verbose and j < 3:  # Print for first 3 neurons only
                    print(f'      Neuron {j}: Applied shift={neuron_shift} to continuous feature {foi}')
            
            loc, lower_border, upper_border = select_signal_roi(csignal, seed=seed)
            # Generate binary series from a continuous one
            binary_series = np.zeros(length)
            binary_series[np.where((csignal >= lower_border) & (csignal <= upper_border))] = 1

        elif ftype == 'd':
            binary_series = all_feats[foi].copy()  # Make a copy
            
            # Apply random per-neuron shift to break correlations
            if apply_random_neuron_shifts:
                # Apply a unique random shift for this neuron
                neuron_shift = np.random.randint(0, length)
                binary_series = np.roll(binary_series, neuron_shift)
                if verbose and j < 3:  # Print for first 3 neurons only
                    print(f'      Neuron {j}: Applied shift={neuron_shift} to discrete feature {foi}')

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
        # Do not modify seed during feature generation
        # if seed is not None:
        #     seed += 1  # save reproducibility, but break degeneracy

    return np.vstack(all_feats), np.vstack(all_signals), gt


def generate_synthetic_exp(n_dfeats=20, n_cfeats=20, nneurons=500, seed=0, fps=20, with_spikes=False, duration=1200):
    """
    Generate a synthetic experiment with neurons selective to discrete and continuous features.
    
    Parameters
    ----------
    n_dfeats : int, optional
        Number of discrete features. Default: 20.
    n_cfeats : int, optional
        Number of continuous features. Default: 20.
    nneurons : int, optional
        Total number of neurons. Default: 500.
    seed : int, optional
        Random seed for reproducibility. Default: 0.
    fps : float, optional
        Frames per second. Default: 20.
    with_spikes : bool, optional
        If True, reconstruct spikes from calcium using wavelet method. Default: False.
    duration : int, optional
        Duration of the experiment in seconds. Default: 1200.
        
    Returns
    -------
    exp : Experiment
        Synthetic experiment object with calcium signals and optionally spike data.
    """
    # Set the numpy random seed at the beginning of the function
    if seed is not None:
        np.random.seed(seed)
    # Split neurons between those responding to discrete and continuous features
    # For odd numbers, give the extra neuron to the first group
    # But if one type has 0 features, allocate all neurons to the other type
    if n_dfeats == 0:
        n_neurons_discrete = 0
        n_neurons_continuous = nneurons
    elif n_cfeats == 0:
        n_neurons_discrete = nneurons
        n_neurons_continuous = 0
    else:
        n_neurons_discrete = (nneurons + 1) // 2
        n_neurons_continuous = nneurons // 2
    
    dfeats, calcium1, gt = generate_synthetic_data(n_dfeats,
                                                   n_neurons_discrete,
                                                   duration=duration,
                                                   hurst=0.3,
                                                   ftype='d',
                                                   seed=seed,
                                                   rate_0=0.1,
                                                   rate_1=1.0,
                                                   skip_prob=0.1,
                                                   noise_std=0.1,
                                                   sampling_rate=fps)

    cfeats, calcium2, gt2 = generate_synthetic_data(n_cfeats,  # Fixed: was n_dfeats
                                                    n_neurons_continuous,
                                                    duration=duration,
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

    # Combine calcium signals, handling empty arrays
    if n_neurons_discrete == 0:
        all_calcium = calcium2
    elif n_neurons_continuous == 0:
        all_calcium = calcium1
    else:
        all_calcium = np.vstack([calcium1, calcium2])
    
    # Create experiment
    if with_spikes:
        # Create experiment with spike reconstruction
        exp = Experiment('Synthetic',
                         all_calcium,
                         None,
                         {},
                         {'fps': fps},
                         {**discr_ts, **cont_ts},
                         reconstruct_spikes='wavelet')
    else:
        # Create experiment without spikes
        exp = Experiment('Synthetic',
                         all_calcium,
                         None,
                         {},
                         {'fps': fps},
                         {**discr_ts, **cont_ts},
                         reconstruct_spikes=None)

    return exp


def generate_mixed_population_exp(n_neurons=100, manifold_fraction=0.6,
                                  manifold_type='2d_spatial', manifold_params=None,
                                  n_discrete_features=3, n_continuous_features=3,
                                  feature_params=None, correlation_mode='independent',
                                  correlation_strength=0.3, duration=600, fps=20.0,
                                  seed=None, verbose=True, return_info=False):
    """
    Generate synthetic experiment with mixed population of manifold and feature-selective cells.
    
    This function creates a neural population combining spatial cells (place cells, head direction)
    with feature-selective cells responding to behavioral variables. The mixing ratio and
    correlations between spatial and behavioral activities can be configured.
    
    Parameters
    ----------
    n_neurons : int
        Total number of neurons in the population.
    manifold_fraction : float
        Fraction of neurons that are manifold cells (0.0-1.0).
        Remaining neurons will be feature-selective.
    manifold_type : str
        Type of manifold: 'circular', '2d_spatial', '3d_spatial'.
    manifold_params : dict, optional
        Parameters for manifold generation. If None, uses defaults.
    n_discrete_features : int
        Number of discrete behavioral features.
    n_continuous_features : int
        Number of continuous behavioral features.
    feature_params : dict, optional
        Parameters for feature generation. If None, uses defaults.
    correlation_mode : str
        How to correlate spatial and behavioral activities:
        - 'independent': No correlation between spatial and behavioral
        - 'spatial_correlated': Behavioral features modulated by spatial position
        - 'feature_correlated': Spatial activity modulated by behavioral features
    correlation_strength : float
        Strength of correlation (0.0-1.0) when correlation_mode is not 'independent'.
    duration : float
        Duration of experiment in seconds.
    fps : float
        Sampling rate in Hz.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress messages.
    return_info : bool
        If True, return (exp, info) tuple. If False (default), return only exp.
        
    Returns
    -------
    exp : Experiment
        Experiment object with mixed population.
    info : dict (only if return_info=True)
        Dictionary containing:
        - 'population_composition': Details about neuron allocation
        - 'manifold_info': Information about manifold cells
        - 'feature_selectivity': Information about feature-selective cells
        - 'spatial_data': Spatial trajectory data
        - 'behavioral_features': Behavioral feature data
        - 'correlation_applied': Correlation mode used
        
    Examples
    --------
    >>> # Generate population with 60% place cells, 40% feature-selective
    >>> exp, info = generate_mixed_population_exp(
    ...     n_neurons=50,
    ...     manifold_fraction=0.6,
    ...     manifold_type='2d_spatial',
    ...     correlation_mode='spatial_correlated'
    ... )
    
    >>> # Check population composition
    >>> print(f"Manifold cells: {info['population_composition']['n_manifold']}")
    >>> print(f"Feature-selective: {info['population_composition']['n_feature_selective']}")
    
    Notes
    -----
    The function integrates existing manifold and feature generators to create
    realistic mixed populations. Spatial correlations can model scenarios where
    behavioral variables depend on location (e.g., speed varying with position)
    or where spatial coding is modulated by behavioral state.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate parameters
    if not 0.0 <= manifold_fraction <= 1.0:
        raise ValueError(f"manifold_fraction must be between 0.0 and 1.0, got {manifold_fraction}")
    
    if manifold_type not in ['circular', '2d_spatial', '3d_spatial']:
        raise ValueError(f"manifold_type must be 'circular', '2d_spatial', or '3d_spatial', got {manifold_type}")
    
    if correlation_mode not in ['independent', 'spatial_correlated', 'feature_correlated']:
        raise ValueError(f"Invalid correlation_mode: {correlation_mode}")
    
    if not 0.0 <= correlation_strength <= 1.0:
        raise ValueError(f"correlation_strength must be between 0.0 and 1.0, got {correlation_strength}")

    selectivity_prob = feature_params.get('selectivity_prob', 1.0) if feature_params else 1.0
    # Calculate population allocation
    n_manifold = int(n_neurons * manifold_fraction)
    n_feature_selective = int((n_neurons - n_manifold) * selectivity_prob)
    
    if verbose:
        print(f'Generating mixed population: {n_neurons} total neurons')
        print(f'  Manifold cells ({manifold_type}): {n_manifold}')
        print(f'  Expected feature-selective cells: {n_feature_selective}')
        print(f'  Correlation mode: {correlation_mode}')
    
    # Set default parameters
    if manifold_params is None:
        manifold_params = {
            'field_sigma': 0.1,
            'baseline_rate': 0.1,
            'peak_rate': 1.0,  # Realistic for calcium imaging
            'noise_std': 0.05,
            'decay_time': 2.0,
            'calcium_noise_std': 0.1
        }
    
    if feature_params is None:
        feature_params = {
            'rate_0': 0.1,
            'rate_1': 1.0,
            'skip_prob': 0.1,
            'hurst': 0.3,
            'ampl_range': (0.5, 2.0),
            'decay_time': 2.0,
            'noise_std': 0.1
        }
    
    # Initialize containers
    all_calcium_signals = []
    dynamic_features = {}
    manifold_info = {}
    spatial_data = None
    feature_selectivity = None
    
    # Generate manifold cells
    if n_manifold > 0:
        if verbose:
            print(f'  Generating {n_manifold} {manifold_type} manifold cells...')
        
        manifold_seed = seed if seed is None else seed + 1000
        
        if manifold_type == 'circular':
            calcium_manifold, head_direction, preferred_dirs, firing_rates = \
                generate_circular_manifold_data(
                    n_manifold, duration, fps,
                    kappa=manifold_params.get('kappa', 4.0),
                    step_std=manifold_params.get('step_std', 0.1),
                    baseline_rate=manifold_params['baseline_rate'],
                    peak_rate=manifold_params['peak_rate'],
                    noise_std=manifold_params['noise_std'],
                    decay_time=manifold_params['decay_time'],
                    calcium_noise_std=manifold_params['calcium_noise_std'],
                    seed=manifold_seed,
                    verbose=verbose
                )
            
            # Add circular features
            dynamic_features['head_direction'] = TimeSeries(head_direction, discrete=False)
            dynamic_features['circular_angle'] = MultiTimeSeries([
                TimeSeries(np.cos(head_direction), discrete=False),
                TimeSeries(np.sin(head_direction), discrete=False)
            ])
            
            spatial_data = head_direction
            manifold_info = {
                'manifold_type': 'circular',
                'head_direction': head_direction,
                'preferred_directions': preferred_dirs,
                'firing_rates': firing_rates
            }
            
        elif manifold_type == '2d_spatial':
            calcium_manifold, positions, centers, firing_rates = \
                generate_2d_manifold_data(
                    n_manifold, duration, fps,
                    field_sigma=manifold_params['field_sigma'],
                    step_size=manifold_params.get('step_size', 0.02),
                    momentum=manifold_params.get('momentum', 0.8),
                    baseline_rate=manifold_params['baseline_rate'],
                    peak_rate=manifold_params['peak_rate'],
                    noise_std=manifold_params['noise_std'],
                    decay_time=manifold_params['decay_time'],
                    calcium_noise_std=manifold_params['calcium_noise_std'],
                    grid_arrangement=manifold_params.get('grid_arrangement', True),
                    seed=manifold_seed,
                    verbose=verbose
                )
            
            # Add spatial features
            dynamic_features['x_position'] = TimeSeries(positions[0, :], discrete=False)
            dynamic_features['y_position'] = TimeSeries(positions[1, :], discrete=False)
            dynamic_features['position_2d'] = MultiTimeSeries([
                TimeSeries(positions[0, :], discrete=False),
                TimeSeries(positions[1, :], discrete=False)
            ])
            
            spatial_data = positions
            manifold_info = {
                'manifold_type': '2d_spatial',
                'positions': positions,
                'place_field_centers': centers,
                'firing_rates': firing_rates
            }
            
        elif manifold_type == '3d_spatial':
            calcium_manifold, positions, centers, firing_rates = \
                generate_3d_manifold_data(
                    n_manifold, duration, fps,
                    field_sigma=manifold_params['field_sigma'],
                    step_size=manifold_params.get('step_size', 0.02),
                    momentum=manifold_params.get('momentum', 0.8),
                    baseline_rate=manifold_params['baseline_rate'],
                    peak_rate=manifold_params['peak_rate'],
                    noise_std=manifold_params['noise_std'],
                    decay_time=manifold_params['decay_time'],
                    calcium_noise_std=manifold_params['calcium_noise_std'],
                    grid_arrangement=manifold_params.get('grid_arrangement', True),
                    seed=manifold_seed,
                    verbose=verbose
                )
            
            # Add 3D spatial features
            dynamic_features['x_position'] = TimeSeries(positions[0, :], discrete=False)
            dynamic_features['y_position'] = TimeSeries(positions[1, :], discrete=False)
            dynamic_features['z_position'] = TimeSeries(positions[2, :], discrete=False)
            dynamic_features['position_3d'] = MultiTimeSeries([
                TimeSeries(positions[0, :], discrete=False),
                TimeSeries(positions[1, :], discrete=False),
                TimeSeries(positions[2, :], discrete=False)
            ])
            
            spatial_data = positions
            manifold_info = {
                'manifold_type': '3d_spatial',
                'positions': positions,
                'place_field_centers': centers,
                'firing_rates': firing_rates
            }
        
        all_calcium_signals.append(calcium_manifold)
    
    # Generate behavioral features
    behavioral_features_data = {}
    
    if n_discrete_features > 0 or n_continuous_features > 0:
        if verbose:
            print(f'  Generating behavioral features: {n_discrete_features} discrete, {n_continuous_features} continuous')
        
        length = int(duration * fps)
        feature_seed = seed if seed is None else seed + 2000
        
        # Generate discrete features
        for i in range(n_discrete_features):
            binary_series = generate_binary_time_series(
                length, 
                avg_islands=feature_params.get('avg_islands', 10),
                avg_duration=int(feature_params.get('avg_duration', 5) * fps)
            )
            
            feat_name = f'd_feat_{i}'
            behavioral_features_data[feat_name] = binary_series
            dynamic_features[feat_name] = TimeSeries(binary_series, discrete=True)
            if feature_seed is not None:
                feature_seed += 1
        
        # Generate continuous features
        for i in range(n_continuous_features):
            fbm_series = generate_fbm_time_series(
                length, 
                hurst=feature_params['hurst'], 
                seed=feature_seed
            )
            
            feat_name = f'c_feat_{i}'
            behavioral_features_data[feat_name] = fbm_series
            dynamic_features[feat_name] = TimeSeries(fbm_series, discrete=False)
            if feature_seed is not None:
                feature_seed += 1
    
    # Apply correlation if requested
    if correlation_mode == 'spatial_correlated' and spatial_data is not None:
        if verbose:
            print(f'  Applying spatial correlation (strength={correlation_strength})')
        
        # Modulate behavioral features based on spatial position
        for feat_name, feat_data in behavioral_features_data.items():
            if 'c_feat' in feat_name:  # Only continuous features
                # Use average position as spatial signal
                if spatial_data.ndim == 1:  # Circular case
                    spatial_signal = np.sin(spatial_data)  # Project to [-1, 1]
                else:  # 2D/3D spatial case
                    spatial_signal = np.mean(spatial_data, axis=0)  # Average position
                
                # Normalize spatial signal
                spatial_signal = (spatial_signal - np.mean(spatial_signal)) / np.std(spatial_signal)
                
                # Apply correlation
                correlated_feat = (1 - correlation_strength) * feat_data + \
                                  correlation_strength * spatial_signal * np.std(feat_data)
                
                behavioral_features_data[feat_name] = correlated_feat
                dynamic_features[feat_name] = TimeSeries(correlated_feat, discrete=False)
    
    elif correlation_mode == 'independent':
        # Ensure true independence by regenerating features with different seeds
        if verbose:
            print(f'  Ensuring feature independence by regenerating behavioral features...')
        
        # Use completely different seeds for independent features
        independent_seed = seed + 10000 if seed is not None else None
        
        # Regenerate discrete features with new seeds
        for i in range(n_discrete_features):
            if independent_seed is not None:
                np.random.seed(independent_seed + i * 100)
            
            # Generate new binary series with different temporal pattern
            binary_series = generate_binary_time_series(
                length, 
                avg_islands=feature_params.get('avg_islands', 10) + np.random.randint(-3, 4),  # Vary parameters
                avg_duration=int(feature_params.get('avg_duration', 5) * fps * np.random.uniform(0.5, 1.5))
            )
            
            feat_name = f'd_feat_{i}'
            behavioral_features_data[feat_name] = binary_series
            dynamic_features[feat_name] = TimeSeries(binary_series, discrete=True)
        
        # Regenerate continuous features with new seeds
        for i in range(n_continuous_features):
            if independent_seed is not None:
                np.random.seed(independent_seed + 1000 + i * 100)
            
            # Use random low Hurst parameter (0.2-0.4) to break temporal autocorrelation and ensure independence
            # Anti-persistent behavior (H < 0.5) breaks accidental spatial correlations
            # Varying H across features prevents systematic correlations
            low_hurst = np.random.uniform(0.2, 0.4)
            
            # Apply random circular shift to break correlation with spatial trajectory
            # Random shift between 1/4 and 3/4 of the series length
            roll_shift = np.random.randint(length // 4, 3 * length // 4)
            
            if verbose:
                print(f'    Feature c_feat_{i}: Using Hurst={low_hurst:.3f}, roll_shift={roll_shift} for independence')
            
            fbm_series = generate_fbm_time_series(
                length, 
                hurst=low_hurst, 
                seed=independent_seed + 1000 + i * 100 if independent_seed is not None else None,
                roll_shift=roll_shift
            )
            
            feat_name = f'c_feat_{i}'
            behavioral_features_data[feat_name] = fbm_series
            dynamic_features[feat_name] = TimeSeries(fbm_series, discrete=False)
    
    # Generate feature-selective cells
    if n_feature_selective > 0:
        if verbose:
            print(f'  Generating {n_feature_selective} feature-selective cells...')
        
        feature_seed = seed if seed is None else seed + 3000
        
        # Prepare features for synthetic data generation
        discrete_feats = [behavioral_features_data[f'd_feat_{i}'] 
                         for i in range(n_discrete_features)]
        continuous_feats = [behavioral_features_data[f'c_feat_{i}'] 
                           for i in range(n_continuous_features)]
        
        all_feats = discrete_feats + continuous_feats
        
        if len(all_feats) == 0:
            # No features - generate baseline neurons
            calcium_features = np.random.normal(0, feature_params['noise_std'], 
                                               (n_feature_selective, int(duration * fps)))
            gt_features = np.zeros((0, n_feature_selective))
        else:
            # Check if mixed selectivity is requested
            use_mixed_selectivity = feature_params.get('multi_select_prob', 0) > 0
            
            if use_mixed_selectivity:
                # Use mixed selectivity generation
                selectivity_seed = None if feature_seed is None else feature_seed + 500
                
                # Generate selectivity patterns
                selectivity_matrix = generate_multiselectivity_patterns(
                    n_feature_selective, 
                    n_discrete_features + n_continuous_features,
                    mode='random',
                    selectivity_prob=feature_params.get('selectivity_prob', 0.8),
                    multi_select_prob=feature_params.get('multi_select_prob', 0.4),
                    weights_mode='random',
                    seed=selectivity_seed
                )
                
                # Create features dictionary
                features_dict = {}
                for i in range(n_discrete_features):
                    features_dict[f'd_feat_{i}'] = behavioral_features_data[f'd_feat_{i}']
                for i in range(n_continuous_features):
                    features_dict[f'c_feat_{i}'] = behavioral_features_data[f'c_feat_{i}']
                
                # Generate mixed selective signals
                calcium_features, gt_features = generate_synthetic_data_mixed_selectivity(
                    features_dict, n_feature_selective, selectivity_matrix,
                    duration=duration,
                    seed=feature_seed,
                    sampling_rate=fps,
                    rate_0=feature_params['rate_0'],
                    rate_1=feature_params['rate_1'],
                    skip_prob=feature_params['skip_prob'],
                    ampl_range=feature_params['ampl_range'],
                    decay_time=feature_params['decay_time'],
                    noise_std=feature_params['noise_std'],
                    verbose=False
                )
            else:
                # Original code for single selectivity
                # Generate neurons for discrete features
                all_calcium_parts = []
                all_gt_parts = []
                
                if n_discrete_features > 0:
                    # Generate neurons selective to discrete features
                    discrete_seed = None if feature_seed is None else feature_seed + 10
                    # Pass pregenerated discrete features
                    discrete_feat_list = [behavioral_features_data[f'd_feat_{i}'] 
                                        for i in range(n_discrete_features)]
                    feats_d, calcium_d, gt_d = generate_synthetic_data(
                        n_discrete_features, n_feature_selective // 2 if n_continuous_features > 0 else n_feature_selective,
                        ftype='d',
                        duration=duration,
                        seed=discrete_seed,
                        sampling_rate=fps,
                        rate_0=feature_params['rate_0'],
                        rate_1=feature_params['rate_1'],
                        skip_prob=feature_params['skip_prob'],
                        ampl_range=feature_params['ampl_range'],
                        decay_time=feature_params['decay_time'],
                        noise_std=feature_params['noise_std'],
                        verbose=verbose,
                        pregenerated_features=discrete_feat_list,
                        apply_random_neuron_shifts=(correlation_mode == 'independent')
                    )
                    all_calcium_parts.append(calcium_d)
                    # Adjust gt_d indices to account for all features
                    gt_d_adjusted = np.zeros((n_discrete_features + n_continuous_features, gt_d.shape[1]))
                    gt_d_adjusted[:n_discrete_features, :] = gt_d
                    all_gt_parts.append(gt_d_adjusted)
                
                if n_continuous_features > 0:
                    # Generate neurons selective to continuous features
                    remaining_neurons = n_feature_selective - (len(all_calcium_parts[0]) if all_calcium_parts else 0)
                    continuous_seed = None if feature_seed is None else feature_seed + 100
                    # Pass pregenerated continuous features
                    continuous_feat_list = [behavioral_features_data[f'c_feat_{i}'] 
                                          for i in range(n_continuous_features)]
                    feats_c, calcium_c, gt_c = generate_synthetic_data(
                        n_continuous_features, remaining_neurons,
                        ftype='c',
                        duration=duration,
                        seed=continuous_seed,
                        sampling_rate=fps,
                        rate_0=feature_params['rate_0'],
                        rate_1=feature_params['rate_1'],
                        skip_prob=feature_params['skip_prob'],
                        hurst=feature_params['hurst'],
                        ampl_range=feature_params['ampl_range'],
                        decay_time=feature_params['decay_time'],
                        noise_std=feature_params['noise_std'],
                        verbose=verbose,
                        pregenerated_features=continuous_feat_list,
                        apply_random_neuron_shifts=(correlation_mode == 'independent')
                    )
                    all_calcium_parts.append(calcium_c)
                    # Adjust gt_c indices to account for discrete features
                    gt_c_adjusted = np.zeros((n_discrete_features + n_continuous_features, gt_c.shape[1]))
                    gt_c_adjusted[n_discrete_features:, :] = gt_c
                    all_gt_parts.append(gt_c_adjusted)
                
                # Combine calcium signals and ground truth
                if len(all_calcium_parts) == 1:
                    calcium_features = all_calcium_parts[0]
                    gt_features = all_gt_parts[0]
                else:
                    calcium_features = np.vstack(all_calcium_parts)
                    # Combine ground truth matrices
                    gt_features = np.zeros((n_discrete_features + n_continuous_features, calcium_features.shape[0]))
                    neuron_idx = 0
                    for gt_part in all_gt_parts:
                        n_neurons_part = gt_part.shape[1] if len(gt_part.shape) > 1 else 0
                        if n_neurons_part > 0:
                            gt_features[:, neuron_idx:neuron_idx + n_neurons_part] = gt_part
                            neuron_idx += n_neurons_part
        
        # Apply feature correlation if requested
        if correlation_mode == 'feature_correlated' and spatial_data is not None and n_manifold > 0:
            if verbose:
                print(f'  Applying feature correlation to manifold cells (strength={correlation_strength})')
            
            # Modulate manifold cells based on behavioral features
            if len(all_feats) > 0:
                # Use first continuous feature as modulation signal
                modulation_signal = None
                for feat_name, feat_data in behavioral_features_data.items():
                    if 'c_feat' in feat_name:
                        modulation_signal = feat_data
                        break
                
                if modulation_signal is not None:
                    # Normalize modulation signal
                    mod_norm = (modulation_signal - np.mean(modulation_signal)) / np.std(modulation_signal)
                    
                    # Apply to manifold calcium signals
                    for i in range(n_manifold):
                        baseline = np.mean(calcium_manifold[i])
                        modulated = calcium_manifold[i] + correlation_strength * mod_norm * baseline * 0.2
                        calcium_manifold[i] = np.maximum(0, modulated)  # Ensure non-negative
        
        all_calcium_signals.append(calcium_features)
        feature_selectivity = gt_features
    
    # Combine all calcium signals
    if len(all_calcium_signals) == 1:
        combined_calcium = all_calcium_signals[0]
    else:
        combined_calcium = np.vstack(all_calcium_signals)
    
    # Create static features
    static_features = {
        'fps': fps,
        't_rise_sec': 0.5,
        't_off_sec': manifold_params.get('decay_time', 2.0)
    }
    
    # Create experiment
    exp = Experiment(
        'MixedPopulation',
        combined_calcium,
        None,  # No spike data
        {},    # No identificators
        static_features,
        dynamic_features,
        reconstruct_spikes=None
    )
    
    # Prepare comprehensive info dictionary
    info = {
        'population_composition': {
            'n_manifold': n_manifold,
            'n_feature_selective': n_feature_selective,
            'manifold_type': manifold_type,
            'manifold_indices': list(range(n_manifold)),
            'feature_indices': list(range(n_manifold, n_neurons)),
            'manifold_fraction': manifold_fraction
        },
        'manifold_info': manifold_info,
        'feature_selectivity': feature_selectivity,
        'spatial_data': spatial_data,
        'behavioral_features': behavioral_features_data,
        'correlation_applied': correlation_mode,
        'correlation_strength': correlation_strength if correlation_mode != 'independent' else 0.0,
        'parameters': {
            'manifold_params': manifold_params,
            'feature_params': feature_params,
            'n_discrete_features': n_discrete_features,
            'n_continuous_features': n_continuous_features
        }
    }
    
    if verbose:
        print(f'  Mixed population generated successfully!')
        print(f'  Total calcium traces: {combined_calcium.shape}')
        print(f'  Total features: {len(dynamic_features)}')
    
    if return_info:
        return exp, info
    else:
        return exp