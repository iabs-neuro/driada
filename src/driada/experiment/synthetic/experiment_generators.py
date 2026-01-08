"""
High-level experiment generators for synthetic neural data.

This module contains functions that generate complete synthetic experiments
by combining various types of neural data (manifold-based, feature-selective, mixed).
"""

import numpy as np
import tqdm
from .core import generate_pseudo_calcium_signal
from .time_series import (
    generate_binary_time_series,
    generate_fbm_time_series,
    select_signal_roi,
    delete_one_islands,
    apply_poisson_to_binary_series,
)
from .manifold_circular import generate_circular_manifold_data
from .manifold_spatial_2d import generate_2d_manifold_data
from .mixed_selectivity import (
    generate_multiselectivity_patterns,
    generate_synthetic_data_mixed_selectivity,
)
from ..exp_base import Experiment
from ...information.info_base import TimeSeries, MultiTimeSeries
from ...utils.data import check_nonnegative


def generate_synthetic_data(
    nfeats,
    nneurons,
    ftype="c",
    duration=600,
    seed=42,
    sampling_rate=20.0,
    rate_0=0.1,
    rate_1=1.0,
    skip_prob=0.0,
    hurst=0.5,
    ampl_range=(0.5, 2),
    decay_time=2,
    avg_islands=10,
    avg_duration=5,
    noise_std=0.1,
    verbose=True,
    pregenerated_features=None,
    apply_random_neuron_shifts=False,
):
    """
    Generate synthetic neural data with simple threshold-based selectivity.

    NOTE: This is a technical/legacy generator using binary ON/OFF responses.
    For scientific simulations with realistic tuning curves (von Mises, Gaussian),
    use generate_tuned_selectivity_exp() instead.

    This function is useful for:
    - Technical validation of INTENSE algorithm
    - Simple sanity checks
    - Backward compatibility with older code

    Creates synthetic calcium imaging data where each neuron is selective to
    one feature (either continuous or discrete). Features are generated using
    fractional Brownian motion (continuous) or binary island patterns (discrete).

    Parameters
    ----------
    nfeats : int
        Number of features. Must be non-negative.
    nneurons : int
        Number of neurons. Must be non-negative.
    ftype : str
        Feature type: 'c' for continuous, 'd' for discrete.
    duration : float
        Duration in seconds. Must be positive.
    seed : int
        Random seed for reproducibility.
    sampling_rate : float
        Sampling rate in Hz. Must be positive.
    rate_0 : float
        Baseline firing rate in Hz. Must be non-negative.
    rate_1 : float
        Active firing rate in Hz. Must be non-negative.
    skip_prob : float
        Probability of skipping islands. Must be in [0, 1].
    hurst : float
        Hurst parameter for FBM (0-1). 0.5 = random walk.
    ampl_range : tuple
        (min, max) amplitude range for calcium events.
    decay_time : float
        Calcium decay time constant in seconds. Must be positive.
    avg_islands : int
        Average number of islands for discrete features. Must be positive.
    avg_duration : int
        Average duration of islands in seconds. Must be positive.
    noise_std : float
        Noise standard deviation. Must be non-negative.
    verbose : bool
        Print progress messages.
    pregenerated_features : list, optional
        Pre-generated feature arrays to use instead of generating new ones.
        Must have length equal to nfeats if provided.
    apply_random_neuron_shifts : bool
        Apply random circular shifts to break correlations between neurons.

    Returns
    -------
    features : ndarray
        Feature time series of shape (nfeats, n_timepoints).
        Empty array if nfeats=0.
    signals : ndarray
        Neural calcium signals of shape (nneurons, n_timepoints).
        Shape (0, n_timepoints) if nneurons=0.
    ground_truth : ndarray
        Ground truth connectivity matrix of shape (nfeats, nneurons).
        Binary matrix where gt[i,j]=1 means neuron j responds to feature i.

    Raises
    ------
    ValueError
        If ftype is not 'c' or 'd'.
        If pregenerated_features length doesn't match nfeats.
        If any numeric parameters are out of valid ranges.

    Side Effects
    ------------
    - Modifies numpy random state when seed is used (affects global random number generation)
    - Prints progress messages to stdout when verbose=True:
      - "Using pregenerated features..." or "Generating features..."
      - "Generating signals..."
      - Progress bars via tqdm for feature and signal generation
    - Shows progress bars for neuron shifts when verbose=True and apply_random_neuron_shifts=True
    - Allocates potentially large arrays (memory intensive):
      - features array: (nfeats, duration*sampling_rate) float64
      - signals array: (nneurons, duration*sampling_rate) float64
      - ground_truth matrix: (nfeats, nneurons) float64
    - May consume significant memory for large nfeats/nneurons/duration combinations

    Notes
    -----
    - Each neuron randomly assigned to one feature (uniform distribution)
    - When nfeats=0: neurons show baseline activity only
    - When nneurons=0: returns empty arrays with correct dimensions
    - Features use incremental seeds for reproducibility
    - Some features may have no assigned neurons if nfeats > nneurons

    Examples
    --------
    >>> # Generate 10 neurons selective to 3 continuous features
    >>> features, signals, gt = generate_synthetic_data(
    ...     nfeats=3, nneurons=10, ftype='c', duration=100, seed=0, verbose=False
    ... )
    >>> features.shape
    (3, 2000)
    >>> signals.shape
    (10, 2000)
    >>> int(np.sum(gt))  # Each neuron assigned to one feature
    10

    >>> # Use pre-generated discrete features
    >>> prefeats = [np.random.randint(0, 2, 1000) for _ in range(2)]
    >>> f, s, gt = generate_synthetic_data(
    ...     nfeats=2, nneurons=5, ftype='d',
    ...     pregenerated_features=prefeats, verbose=False
    ... )"""
    # Input validation
    check_nonnegative(
        nfeats=nfeats,
        nneurons=nneurons,
        duration=duration,
        sampling_rate=sampling_rate,
        rate_0=rate_0,
        rate_1=rate_1,
        skip_prob=skip_prob,
        hurst=hurst,
        decay_time=decay_time,
        noise_std=noise_std,
        avg_islands=avg_islands,
        avg_duration=avg_duration,
    )

    # Additional validation for ranges
    if not 0 <= skip_prob <= 1:
        raise ValueError(f"skip_prob must be in [0, 1], got {skip_prob}")
    if not 0 <= hurst <= 1:
        raise ValueError(f"hurst must be in [0, 1], got {hurst}")
    if len(ampl_range) != 2 or ampl_range[0] > ampl_range[1]:
        raise ValueError(f"ampl_range must be (min, max) with min <= max, got {ampl_range}")
    check_nonnegative(ampl_min=ampl_range[0], ampl_max=ampl_range[1])
    if ftype not in ["c", "d"]:
        raise ValueError(f"ftype must be 'c' or 'd', got '{ftype}'")

    gt = np.zeros((nfeats, nneurons))
    length = int(duration * sampling_rate)

    # Handle edge case of 0 neurons
    if nneurons == 0:
        if nfeats == 0:
            return np.array([]).reshape(0, length), np.array([]).reshape(0, length), gt
        else:
            # Still need to return features even with 0 neurons
            if pregenerated_features is not None:
                return np.vstack(pregenerated_features), np.array([]).reshape(0, length), gt
            else:
                # Generate features for consistency
                if verbose:
                    print("Generating features...")
                all_feats = []
                feature_iterator = tqdm.tqdm(range(nfeats), disable=not verbose)
                for i in feature_iterator:
                    if ftype == "c":
                        feature_seed = seed + i if seed is not None else None
                        fbm_series = generate_fbm_time_series(length, hurst, seed=feature_seed)
                        all_feats.append(fbm_series)
                    else:
                        # Use seed for reproducibility
                        feature_seed = seed + i if seed is not None else None
                        if feature_seed is not None:
                            np.random.seed(feature_seed)
                        binary_series = generate_binary_time_series(
                            length, avg_islands, avg_duration * sampling_rate
                        )
                        all_feats.append(binary_series)
                return np.vstack(all_feats), np.array([]).reshape(0, length), gt

    # Use pregenerated features if provided, otherwise generate new ones
    if pregenerated_features is not None:
        if verbose:
            print("Using pregenerated features...")
        all_feats = pregenerated_features
        if len(all_feats) != nfeats:
            raise ValueError(
                f"Number of pregenerated features ({len(all_feats)}) does not match nfeats ({nfeats})"
            )
    else:
        if verbose:
            print("Generating features...")
        all_feats = []
        for i in tqdm.tqdm(np.arange(nfeats), disable=not verbose):
            if ftype == "c":
                # Generate the series with unique seed for each feature
                feature_seed = seed + i if seed is not None else None
                fbm_series = generate_fbm_time_series(length, hurst, seed=feature_seed)
                all_feats.append(fbm_series)

            elif ftype == "d":
                # Generate binary series
                binary_series = generate_binary_time_series(
                    length, avg_islands, avg_duration * sampling_rate
                )
                all_feats.append(binary_series)

            else:
                raise ValueError(f"Unknown feature type: {ftype}")

    if verbose:
        print("Generating signals...")

    # Handle feature selection for neurons
    if nfeats > 0:
        fois = np.random.choice(np.arange(nfeats), size=nneurons)
        gt[fois, np.arange(nneurons)] = 1  # add info about ground truth feature-signal connections
    else:
        # If no features, neurons won't be selective to any feature
        fois = np.full(nneurons, -1)  # Use -1 to indicate no feature selection
    all_signals = []

    for j in tqdm.tqdm(np.arange(nneurons), disable=not verbose):
        foi = fois[j]

        # Handle case where there are no features
        if foi == -1 or nfeats == 0:
            # Generate random baseline activity
            binary_series = generate_binary_time_series(
                length, avg_islands // 2, avg_duration * sampling_rate // 2
            )
        elif ftype == "c":
            csignal = all_feats[foi].copy()  # Make a copy to avoid modifying the original

            # Apply random per-neuron shift to break correlations
            if apply_random_neuron_shifts:
                # Apply a unique random shift for this neuron
                neuron_shift = np.random.randint(0, length)
                csignal = np.roll(csignal, neuron_shift)
                if verbose and j < 3:  # Print for first 3 neurons only
                    print(
                        f"      Neuron {j}: Applied shift={neuron_shift} to continuous feature {foi}"
                    )

            loc, lower_border, upper_border = select_signal_roi(csignal, seed=seed)
            # Generate binary series from a continuous one
            binary_series = np.zeros(length)
            binary_series[np.where((csignal >= lower_border) & (csignal <= upper_border))] = 1

        elif ftype == "d":
            binary_series = all_feats[foi].copy()  # Make a copy

            # Apply random per-neuron shift to break correlations
            if apply_random_neuron_shifts:
                # Apply a unique random shift for this neuron
                neuron_shift = np.random.randint(0, length)
                binary_series = np.roll(binary_series, neuron_shift)
                if verbose and j < 3:  # Print for first 3 neurons only
                    print(
                        f"      Neuron {j}: Applied shift={neuron_shift} to discrete feature {foi}"
                    )

        else:
            raise ValueError(f"Unknown feature type: {ftype}")

        # randomly skip some on periods
        mod_binary_series = delete_one_islands(binary_series, skip_prob)

        # Apply Poisson process
        poisson_series = apply_poisson_to_binary_series(
            mod_binary_series, rate_0 / sampling_rate, rate_1 / sampling_rate
        )

        # Generate pseudo-calcium
        pseudo_calcium_signal = generate_pseudo_calcium_signal(
            duration=duration,
            events=poisson_series,
            sampling_rate=sampling_rate,
            amplitude_range=ampl_range,
            decay_time=decay_time,
            noise_std=noise_std,
        )

        all_signals.append(pseudo_calcium_signal)

    # Return features and signals
    if nfeats == 0:
        features = np.array([]).reshape(0, length)
    else:
        features = np.vstack(all_feats)

    if nneurons == 0:
        signals = np.array([]).reshape(0, length)
    else:
        signals = np.vstack(all_signals)

    return features, signals, gt


def generate_synthetic_exp(
    n_dfeats=20,
    n_cfeats=20,
    nneurons=500,
    seed=0,
    fps=20,
    with_spikes=False,
    duration=1200,
    **kwargs,
):
    """
    Generate a synthetic experiment with neurons selective to discrete and continuous features.

    This is a convenience wrapper around generate_tuned_selectivity_exp() that provides
    a simpler API for basic synthetic data. Ground truth is always available via
    exp.ground_truth.

    Parameters
    ----------
    n_dfeats : int, optional
        Number of discrete event features. Default: 20.
    n_cfeats : int, optional
        Number of continuous FBM features. Default: 20.
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
    **kwargs : dict, optional
        Additional parameters passed to generate_tuned_selectivity_exp.
        Supported: verbose, baseline_rate, peak_rate, decay_time, calcium_noise, hurst.

    Returns
    -------
    exp : Experiment
        Synthetic experiment object with calcium signals. Ground truth accessible
        via exp.ground_truth.

    Examples
    --------
    >>> exp = generate_synthetic_exp(n_dfeats=10, n_cfeats=10, nneurons=100, verbose=False)
    >>> exp.n_cells
    100
    >>> exp.ground_truth is not None
    True

    Notes
    -----
    This is a thin wrapper around generate_tuned_selectivity_exp(). For full
    control over population configuration, use generate_tuned_selectivity_exp()
    directly.
    """
    from .principled_selectivity import generate_tuned_selectivity_exp

    # Split neurons evenly between discrete and continuous
    if n_dfeats == 0:
        n_discrete_neurons = 0
        n_continuous_neurons = nneurons
    elif n_cfeats == 0:
        n_discrete_neurons = nneurons
        n_continuous_neurons = 0
    else:
        n_discrete_neurons = (nneurons + 1) // 2
        n_continuous_neurons = nneurons // 2

    # Build population configuration
    population = []

    if n_discrete_neurons > 0 and n_dfeats > 0:
        population.append(
            {
                "name": "event_cells",
                "count": n_discrete_neurons,
                "features": [f"event_{i}" for i in range(n_dfeats)],
            }
        )

    if n_continuous_neurons > 0 and n_cfeats > 0:
        population.append(
            {
                "name": "fbm_cells",
                "count": n_continuous_neurons,
                "features": [f"fbm_{i}" for i in range(n_cfeats)],
            }
        )

    # Extract supported kwargs
    tuned_kwargs = {}
    for key in ["verbose", "baseline_rate", "peak_rate", "decay_time", "calcium_noise", "hurst"]:
        if key in kwargs:
            tuned_kwargs[key] = kwargs[key]

    # Generate experiment using canonical generator
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=duration,
        fps=fps,
        n_discrete_features=n_dfeats,
        seed=seed,
        **tuned_kwargs,
    )

    # TODO: Add spike reconstruction if requested
    # Currently not supported via the thin wrapper
    if with_spikes:
        import warnings
        warnings.warn(
            "with_spikes=True is not yet supported in the thin wrapper. "
            "Spikes not reconstructed.",
            UserWarning,
        )

    return exp


def generate_mixed_population_exp(
    n_neurons=100,
    manifold_fraction=0.6,
    manifold_type="2d_spatial",
    n_discrete_features=3,
    n_continuous_features=3,
    duration=600,
    fps=20.0,
    seed=None,
    verbose=True,
    return_info=False,
):
    """
    Generate synthetic experiment with mixed population of manifold and feature-selective cells.

    This is a convenience wrapper around generate_tuned_selectivity_exp() that provides
    a simpler API for common mixed population scenarios. Ground truth is always
    available via exp.ground_truth.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons in the population. Default: 100.
    manifold_fraction : float
        Fraction of neurons that are manifold cells (0.0-1.0). Default: 0.6.
        Remaining neurons are split between event and FBM cells.
    manifold_type : str
        Type of manifold: 'circular' (head direction) or '2d_spatial' (place cells).
        Default: '2d_spatial'.
    n_discrete_features : int
        Number of discrete event features. Default: 3.
    n_continuous_features : int
        Number of continuous FBM features. Default: 3.
    duration : float
        Duration of experiment in seconds. Default: 600.
    fps : float
        Sampling rate in Hz. Default: 20.0.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress messages. Default: True.
    return_info : bool
        If True, return (exp, info) tuple for backward compatibility.
        Ground truth is always available via exp.ground_truth regardless.
        Default: False.

    Returns
    -------
    exp : Experiment
        Experiment object with mixed population. Ground truth accessible via
        exp.ground_truth containing expected_pairs, tuning_parameters, etc.
    info : dict (only if return_info=True)
        Dictionary containing ground_truth for backward compatibility.

    Raises
    ------
    ValueError
        If manifold_fraction not in [0.0, 1.0].
        If manifold_type not in ['circular', '2d_spatial'].
        If n_neurons < 1.

    Examples
    --------
    >>> # Generate population with 60% place cells, 40% feature-selective
    >>> exp = generate_mixed_population_exp(
    ...     n_neurons=50,
    ...     manifold_fraction=0.6,
    ...     manifold_type='2d_spatial',
    ...     verbose=False
    ... )
    >>> exp.n_cells
    50
    >>> len(exp.ground_truth['expected_pairs']) > 0
    True

    Notes
    -----
    This is a thin wrapper around generate_tuned_selectivity_exp(). For full
    control over population configuration, use generate_tuned_selectivity_exp()
    directly.
    """
    # Input validation
    if not 0.0 <= manifold_fraction <= 1.0:
        raise ValueError(
            f"manifold_fraction must be between 0.0 and 1.0, got {manifold_fraction}"
        )
    if manifold_type not in ["circular", "2d_spatial"]:
        raise ValueError(
            f"manifold_type must be 'circular' or '2d_spatial', got {manifold_type}"
        )
    if n_neurons < 1:
        raise ValueError(f"n_neurons must be at least 1, got {n_neurons}")

    from .principled_selectivity import generate_tuned_selectivity_exp

    # Calculate population allocation
    n_manifold = int(n_neurons * manifold_fraction)
    n_feature = n_neurons - n_manifold
    n_event = n_feature // 2
    n_fbm = n_feature - n_event

    # Build population configuration
    population = []

    # Manifold cells
    if n_manifold > 0:
        if manifold_type == "2d_spatial":
            population.append(
                {"name": "place_cells", "count": n_manifold, "features": ["position_2d"]}
            )
        elif manifold_type == "circular":
            population.append(
                {"name": "hd_cells", "count": n_manifold, "features": ["head_direction"]}
            )

    # Event cells (discrete features)
    if n_event > 0 and n_discrete_features > 0:
        population.append(
            {
                "name": "event_cells",
                "count": n_event,
                "features": [f"event_{i}" for i in range(n_discrete_features)],
            }
        )

    # FBM cells (continuous features)
    if n_fbm > 0 and n_continuous_features > 0:
        population.append(
            {
                "name": "fbm_cells",
                "count": n_fbm,
                "features": [f"fbm_{i}" for i in range(n_continuous_features)],
            }
        )

    # Generate experiment using canonical generator
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=duration,
        fps=fps,
        n_discrete_features=n_discrete_features,
        seed=seed,
        verbose=verbose,
    )

    # Backward compatibility: return_info gives (exp, info) tuple
    if return_info:
        # Build backward-compatible info structure
        n_feature = n_event + n_fbm
        info = {
            "ground_truth": exp.ground_truth,
            "population_composition": {
                "n_manifold": n_manifold,
                "n_feature_selective": n_feature,
                "manifold_type": manifold_type,
                "manifold_indices": list(range(n_manifold)),
                "feature_indices": list(range(n_manifold, n_manifold + n_feature)),
                "manifold_fraction": manifold_fraction,
            },
        }
        return exp, info
    return exp
