"""
Circular manifold generation for head direction cells.

This module contains functions for generating synthetic neural data on circular
manifolds, typically used to model head direction cells.
"""

import numpy as np
from .core import validate_peak_rate, generate_pseudo_calcium_signal
from .utils import get_effective_decay_time
from ..exp_base import Experiment
from ...information.info_base import TimeSeries, MultiTimeSeries
from ...utils.data import check_positive, check_nonnegative


def generate_circular_random_walk(length, step_std=0.1, seed=None):
    """
    Generate a random walk on a circle (head direction trajectory).

    Simulates angular motion by accumulating Gaussian-distributed steps
    and wrapping to [0, 2π). Useful for modeling head direction in
    navigation experiments.

    Parameters
    ----------
    length : int
        Number of time points. Must be non-negative.
    step_std : float, optional
        Standard deviation of angular steps in radians. Must be non-negative.
        Default is 0.1 radians (~5.7 degrees). Typical values: 0.05-0.5.
    seed : int, optional
        Random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    angles : ndarray
        Array of angles in radians [0, 2π). Shape: (length,).
        Returns empty array if length is 0.

    Raises
    ------
    ValueError
        If length is negative or step_std is negative.
    TypeError
        If inputs are not numeric.

    Notes
    -----
    The walk follows: angles[t] = (Σ(i=0 to t) N(0, step_std)) mod 2π
    where N(0, step_std) represents Gaussian noise.    """
    # Input validation
    if not isinstance(length, (int, np.integer)):
        raise TypeError("length must be an integer")
    check_nonnegative(length=length, step_std=step_std)
    
    if seed is not None:
        np.random.seed(seed)

    # Handle edge case
    if length == 0:
        return np.array([])
    
    # Generate random steps
    steps = np.random.normal(0, step_std, length)

    # Cumulative sum to get trajectory
    angles = np.cumsum(steps)

    # Wrap to [0, 2π)
    angles = angles % (2 * np.pi)

    return angles


def von_mises_tuning_curve(angles, preferred_direction, kappa):
    """
    Calculate neural response using Von Mises tuning curve.

    Implements a normalized Von Mises (circular Gaussian) tuning curve,
    commonly used to model head direction cells and other neurons with
    circular selectivity.

    Parameters
    ----------
    angles : ndarray
        Current head directions in radians. Can be any real values
        (automatically handles periodicity).
    preferred_direction : float
        Preferred direction of the neuron in radians.
    kappa : float
        Concentration parameter (inverse width of tuning curve).
        Higher kappa = narrower tuning. Typical values: 2-8.
        kappa=0 gives uniform response, negative kappa inverts tuning.

    Returns
    -------
    response : ndarray
        Neural response (firing rate modulation) normalized to max=1.
        Same shape as input angles. Values in range [exp(-kappa), 1].

    Raises
    ------
    ValueError
        If kappa is NaN or infinity.
    TypeError
        If inputs are not numeric types.

    Notes
    -----
    The response follows: response = exp(κ * (cos(θ - θ_pref) - 1))
    This is a Von Mises distribution normalized to peak at 1.    """
    # Input validation
    angles = np.asarray(angles)
    if not np.issubdtype(angles.dtype, np.number):
        raise TypeError("angles must be numeric")
    if not isinstance(preferred_direction, (int, float)):
        raise TypeError("preferred_direction must be numeric")
    if not isinstance(kappa, (int, float)):
        raise TypeError("kappa must be numeric")
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")
    
    # Von Mises distribution normalized to max=1
    response = np.exp(kappa * (np.cos(angles - preferred_direction) - 1))
    return response


def generate_circular_manifold_neurons(
    n_neurons,
    head_direction,
    kappa=4.0,
    baseline_rate=0.1,
    peak_rate=1.0,
    noise_std=0.05,
    seed=None,
):
    """
    Generate population of head direction cells with Von Mises tuning.

    Creates a population of neurons with uniformly distributed preferred
    directions on the circle, each responding to head direction with
    Von Mises tuning curves. Includes realistic noise and ensures
    non-negative firing rates.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population. Must be positive.
    head_direction : ndarray
        Head direction trajectory in radians. Shape: (n_timepoints,).
    kappa : float, optional
        Concentration parameter for Von Mises tuning curves.
        Typical values: 2-8 (higher = narrower tuning). Default is 4.0.
    baseline_rate : float, optional
        Baseline firing rate when far from preferred direction in Hz.
        Default is 0.1 Hz (realistic for sparse firing neurons).
    peak_rate : float, optional
        Peak firing rate at preferred direction in Hz.
        Default is 1.0 Hz (realistic for calcium imaging).
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to firing rates.
        Default is 0.05. Must be non-negative.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints) with firing rates in Hz.
        All values are non-negative.
    preferred_directions : ndarray
        Preferred direction for each neuron in radians [0, 2π).
        Shape: (n_neurons,).

    Raises
    ------
    ValueError
        If peak_rate is negative, NaN, or infinity.
        If n_neurons is not positive.
        If noise_std is negative.
    TypeError
        If peak_rate is not numeric.

    Notes
    -----
    Preferred directions are uniformly distributed with small jitter
    (σ=0.1 rad) to break symmetry. Firing rates are computed as:
    rate = baseline + (peak - baseline) * von_mises_response + noise    """
    # Input validation
    check_positive(n_neurons=n_neurons)
    check_nonnegative(noise_std=noise_std)
    
    # Validate firing rate
    validate_peak_rate(peak_rate, context="generate_circular_manifold_neurons")

    if seed is not None:
        np.random.seed(seed)

    head_direction = np.asarray(head_direction)
    n_timepoints = len(head_direction)

    # Uniformly distribute preferred directions around the circle
    preferred_directions = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)

    # Add small random jitter to break perfect symmetry
    JITTER_STD = 0.1  # radians, approximately 5.7 degrees
    jitter = np.random.normal(0, JITTER_STD, n_neurons)
    preferred_directions = (preferred_directions + jitter) % (2 * np.pi)

    # Generate firing rates for each neuron
    firing_rates = np.zeros((n_neurons, n_timepoints))

    for i in range(n_neurons):
        # Von Mises tuning curve
        tuning_response = von_mises_tuning_curve(
            head_direction, preferred_directions[i], kappa
        )

        # Scale to desired firing rate range
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * tuning_response

        # Add noise
        noise = np.random.normal(0, noise_std, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)  # Ensure non-negative

        firing_rates[i, :] = firing_rate

    return firing_rates, preferred_directions


def generate_circular_manifold_data(
    n_neurons,
    duration=600,
    sampling_rate=20.0,
    kappa=4.0,
    step_std=0.1,
    baseline_rate=0.1,
    peak_rate=1.0,
    noise_std=0.05,
    decay_time=2.0,
    calcium_noise_std=0.1,
    seed=None,
    verbose=True,
):
    """
    Generate synthetic data with neurons on circular manifold (head direction cells).

    Creates a complete dataset with head direction trajectory, neural responses
    with Von Mises tuning, and realistic calcium imaging signals including noise.

    Parameters
    ----------
    n_neurons : int
        Number of neurons. Must be positive.
    duration : float, optional
        Duration in seconds. Must be positive. Default is 600.
    sampling_rate : float, optional
        Sampling rate in Hz. Must be positive. Default is 20.0.
    kappa : float, optional
        Von Mises concentration parameter (tuning width).
        Default is 4.0. Higher values give narrower tuning.
    step_std : float, optional
        Standard deviation of head direction random walk steps in radians.
        Must be non-negative. Default is 0.1.
    baseline_rate : float, optional
        Baseline firing rate in Hz. Must be non-negative. Default is 0.1.
    peak_rate : float, optional
        Peak firing rate at preferred direction in Hz. Default is 1.0.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float, optional
        Noise in firing rates. Must be non-negative. Default is 0.05.
    decay_time : float, optional
        Calcium decay time constant in seconds. Must be positive. Default is 2.0.
    calcium_noise_std : float, optional
        Noise in calcium signal. Must be non-negative. Default is 0.1.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default is True.

    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    head_direction : ndarray
        Head direction trajectory in radians [0, 2π).
    preferred_directions : ndarray
        Preferred direction for each neuron in radians.
    firing_rates : ndarray
        Underlying firing rates in Hz.

    Raises
    ------
    ValueError
        If any positive parameters are not positive.
        If any non-negative parameters are negative.

    Notes
    -----
    The generation process:
    1. Creates random walk trajectory for head direction
    2. Generates neural responses with Von Mises tuning
    3. Converts firing rates to spike probabilities
    4. Samples spikes using binomial distribution
    5. Convolves spikes with calcium kernel and adds noise    """
    # Input validation
    check_positive(n_neurons=n_neurons, duration=duration, sampling_rate=sampling_rate,
                  decay_time=decay_time)
    check_nonnegative(step_std=step_std, baseline_rate=baseline_rate, 
                     noise_std=noise_std, calcium_noise_std=calcium_noise_std)
    
    if seed is not None:
        np.random.seed(seed)

    n_timepoints = int(duration * sampling_rate)

    if verbose:
        print(f"Generating circular manifold data: {n_neurons} neurons, {duration}s")

    # Generate head direction trajectory
    if verbose:
        print("  Generating head direction trajectory...")
    head_direction = generate_circular_random_walk(n_timepoints, step_std, seed)

    # Generate neural responses
    if verbose:
        print("  Generating neural responses with Von Mises tuning...")
    firing_rates, preferred_directions = generate_circular_manifold_neurons(
        n_neurons,
        head_direction,
        kappa,
        baseline_rate,
        peak_rate,
        noise_std,
        seed=(seed + 1) if seed is not None else None,
    )

    # Convert firing rates to calcium signals
    if verbose:
        print("  Converting to calcium signals...")
    calcium_signals = np.zeros((n_neurons, n_timepoints))

    for i in range(n_neurons):
        # Generate Poisson events from firing rates
        prob_spike = firing_rates[i, :] / sampling_rate
        prob_spike = np.clip(prob_spike, 0, 1)  # Ensure valid probability
        events = np.random.binomial(1, prob_spike)

        # Convert to calcium using existing function
        calcium_signal = generate_pseudo_calcium_signal(
            events=events,
            duration=duration,
            sampling_rate=sampling_rate,
            amplitude_range=(0.5, 2.0),
            decay_time=decay_time,
            noise_std=calcium_noise_std,
        )
        calcium_signals[i, :] = calcium_signal

    if verbose:
        print("  Done!")

    return calcium_signals, head_direction, preferred_directions, firing_rates


def generate_circular_manifold_exp(
    n_neurons=100,
    duration=600,
    fps=20.0,
    kappa=4.0,
    step_std=0.1,
    baseline_rate=0.1,
    peak_rate=1.0,
    noise_std=0.05,
    decay_time=2.0,
    calcium_noise_std=0.1,
    add_mixed_features=False,
    seed=None,
    verbose=True,
    return_info=False,
):
    """
    Generate complete experiment with circular manifold (head direction cells).

    Creates a synthetic experiment with head direction cells arranged on a 
    circular manifold. Neurons have Von Mises tuning curves with uniformly 
    distributed preferred directions.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons. Must be positive. Default is 100.
    duration : float, optional
        Duration in seconds. Must be positive. Default is 600.
    fps : float, optional
        Sampling rate (frames per second). Must be positive. Default is 20.0.
    kappa : float, optional
        Von Mises concentration parameter (tuning width).
        Higher values give narrower tuning. Must be positive.
        Typical values: 2-8. Default is 4.0.
    step_std : float, optional
        Head direction random walk step size in radians.
        Must be non-negative. Default is 0.1 (~5.7 degrees).
    baseline_rate : float, optional
        Baseline firing rate in Hz. Must be non-negative. Default is 0.1.
    peak_rate : float, optional
        Peak firing rate at preferred direction in Hz. Must be positive
        and greater than baseline_rate. Default is 1.0.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float, optional
        Standard deviation of firing rate noise. Must be non-negative.
        Default is 0.05.
    decay_time : float, optional
        Calcium indicator decay time constant in seconds.
        Must be positive. Default is 2.0.
        For short experiments (≤30s), automatically limited to 0.5s.
    calcium_noise_std : float, optional
        Standard deviation of calcium signal noise.
        Must be non-negative. Default is 0.1.
    add_mixed_features : bool, optional
        Whether to add circular_angle MultiTimeSeries with cos/sin 
        representation of head direction. Useful for algorithms that
        cannot handle circular variables directly. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    verbose : bool, optional
        Print progress messages. Default is True.
    return_info : bool, optional
        If True, return additional information dictionary.
        Default is False.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object containing:
        - calcium signals as main data
        - static features: fps, decay times, manifold parameters
        - dynamic features: head_direction (and circular_angle if requested)
        - firing_rates stored as exp.firing_rates attribute
    info : dict, optional
        Only returned if return_info=True. Contains:
        - 'manifold_type': "circular"
        - 'n_neurons': number of neurons
        - 'head_direction': trajectory array
        - 'preferred_directions': array of preferred directions
        - 'firing_rates': underlying firing rates
        - 'parameters': dict of all generation parameters

    Raises
    ------
    ValueError
        If any positive parameters are not positive.
        If any non-negative parameters are negative.
        If baseline_rate >= peak_rate.

    Notes
    -----
    The experiment generation process:
    1. Creates random walk trajectory for head direction
    2. Generates neurons with Von Mises tuning curves
    3. Converts firing rates to realistic calcium signals
    4. Packages data into DRIADA Experiment format
    
    The effective decay time is automatically adjusted for short
    experiments to ensure proper shuffle mask generation.
    
    Side effect: The firing_rates array is stored as an attribute
    on the returned Experiment object (exp.firing_rates).

    Examples
    --------
    >>> # Basic usage
    >>> exp = generate_circular_manifold_exp(n_neurons=10, duration=10, verbose=False)
    
    >>> # With circular angle representation
    >>> exp = generate_circular_manifold_exp(
    ...     n_neurons=10,
    ...     duration=10,
    ...     add_mixed_features=True,
    ...     kappa=6.0,  # Narrower tuning
    ...     verbose=False
    ... )
    
    >>> # Get additional information
    >>> exp, info = generate_circular_manifold_exp(
    ...     n_neurons=10,
    ...     duration=10,
    ...     return_info=True,
    ...     verbose=False
    ... )
    >>> len(info['preferred_directions']) == 10
    True
    """
    # Input validation
    check_positive(n_neurons=n_neurons, duration=duration, fps=fps, kappa=kappa, 
                  peak_rate=peak_rate, decay_time=decay_time)
    check_nonnegative(step_std=step_std, baseline_rate=baseline_rate, 
                     noise_std=noise_std, calcium_noise_std=calcium_noise_std)
    
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")
    
    if baseline_rate >= peak_rate:
        raise ValueError(f"baseline_rate ({baseline_rate}) must be less than peak_rate ({peak_rate})")
    
    # Calculate effective decay time for shuffle mask
    effective_decay_time = get_effective_decay_time(decay_time, duration, verbose)

    # Generate data
    calcium, head_direction, preferred_directions, firing_rates = (
        generate_circular_manifold_data(
            n_neurons=n_neurons,
            duration=duration,
            sampling_rate=fps,
            kappa=kappa,
            step_std=step_std,
            baseline_rate=baseline_rate,
            peak_rate=peak_rate,
            noise_std=noise_std,
            decay_time=decay_time,
            calcium_noise_std=calcium_noise_std,
            seed=seed,
            verbose=verbose,
        )
    )

    # Create static features
    static_features = {
        "fps": fps,
        "t_rise_sec": 0.04,
        "t_off_sec": effective_decay_time,  # Use effective decay time for shuffle mask
        "manifold_type": "circular",
        "kappa": kappa,
        "baseline_rate": baseline_rate,
        "peak_rate": peak_rate,
    }

    # Create dynamic features
    head_direction_ts = TimeSeries(data=head_direction, discrete=False)

    dynamic_features = {"head_direction": head_direction_ts}

    # Add circular_angle MultiTimeSeries if requested
    if add_mixed_features:
        # Create circular_angle as MultiTimeSeries with cos and sin components
        # This is the proper representation for circular variables
        cos_component = np.cos(head_direction)
        sin_component = np.sin(head_direction)
        circular_angle_mts = MultiTimeSeries(
            [
                TimeSeries(data=cos_component, discrete=False),
                TimeSeries(data=sin_component, discrete=False),
            ]
        )
        dynamic_features["circular_angle"] = circular_angle_mts

    # Store additional information
    static_features["preferred_directions"] = preferred_directions

    # Create experiment
    exp = Experiment(
        signature="circular_manifold_exp",
        calcium=calcium,
        spikes=None,  # Will be extracted from calcium if needed
        static_features=static_features,
        dynamic_features=dynamic_features,
        exp_identificators={
            "manifold": "circular",
            "n_neurons": n_neurons,
            "duration": duration,
        },
        verbose=verbose,
    )

    # Store firing rates as additional data
    exp.firing_rates = firing_rates

    # Create info dictionary if requested
    if return_info:
        info = {
            "manifold_type": "circular",
            "n_neurons": n_neurons,
            "head_direction": head_direction,
            "preferred_directions": preferred_directions,
            "firing_rates": firing_rates,
            "parameters": {
                "kappa": kappa,
                "step_std": step_std,
                "baseline_rate": baseline_rate,
                "peak_rate": peak_rate,
                "noise_std": noise_std,
                "decay_time": decay_time,
                "calcium_noise_std": calcium_noise_std,
            },
        }
        return exp, info

    return exp
