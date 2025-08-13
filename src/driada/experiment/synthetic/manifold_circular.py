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


def generate_circular_random_walk(length, step_std=0.1, seed=None):
    """
    Generate a random walk on a circle (head direction trajectory).

    Parameters
    ----------
    length : int
        Number of time points.
    step_std : float
        Standard deviation of angular steps in radians.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    angles : ndarray
        Array of angles in radians [0, 2π).
    """
    if seed is not None:
        np.random.seed(seed)

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

    Parameters
    ----------
    angles : ndarray
        Current head directions in radians.
    preferred_direction : float
        Preferred direction of the neuron in radians.
    kappa : float
        Concentration parameter (inverse width of tuning curve).
        Higher kappa = narrower tuning.

    Returns
    -------
    response : ndarray
        Neural response (firing rate modulation).
    """
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

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population.
    head_direction : ndarray
        Head direction trajectory in radians.
    kappa : float
        Concentration parameter for Von Mises tuning curves.
        Typical values: 2-8 (higher = narrower tuning).
    baseline_rate : float
        Baseline firing rate when far from preferred direction.
        Default is 0.1 Hz (realistic for sparse firing neurons).
    peak_rate : float
        Peak firing rate at preferred direction.
        Default is 1.0 Hz (realistic for calcium imaging).
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float
        Standard deviation of noise in firing rates.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints) with firing rates.
    preferred_directions : ndarray
        Preferred direction for each neuron in radians.
    """
    # Validate firing rate
    validate_peak_rate(peak_rate, context="generate_circular_manifold_neurons")

    if seed is not None:
        np.random.seed(seed)

    n_timepoints = len(head_direction)

    # Uniformly distribute preferred directions around the circle
    preferred_directions = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)

    # Add small random jitter to break perfect symmetry
    jitter = np.random.normal(0, 0.1, n_neurons)
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

    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    duration : float
        Duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    kappa : float
        Von Mises concentration parameter (tuning width).
    step_std : float
        Standard deviation of head direction random walk steps.
    baseline_rate : float
        Baseline firing rate. Default is 0.1 Hz.
    peak_rate : float
        Peak firing rate at preferred direction. Default is 1.0 Hz.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float
        Noise in firing rates.
    decay_time : float
        Calcium decay time constant.
    calcium_noise_std : float
        Noise in calcium signal.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    head_direction : ndarray
        Head direction trajectory.
    preferred_directions : ndarray
        Preferred direction for each neuron.
    firing_rates : ndarray
        Underlying firing rates.
    """
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
        seed=(seed + 1) if seed else None,
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

    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    duration : float
        Duration in seconds.
    fps : float
        Sampling rate (frames per second).
    kappa : float
        Von Mises concentration parameter.
    step_std : float
        Head direction random walk step size.
    baseline_rate : float
        Baseline firing rate. Default is 0.1 Hz.
    peak_rate : float
        Peak firing rate. Default is 1.0 Hz.
    noise_std : float
        Firing rate noise.
    decay_time : float
        Calcium decay time.
    calcium_noise_std : float
        Calcium signal noise.
    add_mixed_features : bool
        Whether to add circular_angle MultiTimeSeries (cos/sin representation).
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with circular manifold data.
    """
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
