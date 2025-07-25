"""
2D spatial manifold generation for place cells.

This module contains functions for generating synthetic neural data on 2D spatial
manifolds, typically used to model hippocampal place cells.
"""

import numpy as np
from .core import validate_peak_rate, generate_pseudo_calcium_signal
from .utils import get_effective_decay_time
from ..exp_base import Experiment
from ...information.info_base import TimeSeries, MultiTimeSeries


def generate_2d_random_walk(length, bounds=(0, 1), step_size=0.02, momentum=0.8, seed=None):
    """
    Generate a 2D random walk trajectory within bounded region.
    
    Parameters
    ----------
    length : int
        Number of time points.
    bounds : tuple
        (min, max) bounds for x and y coordinates.
    step_size : float
        Step size for random walk.
    momentum : float
        Momentum factor (0-1) for smoother trajectories.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    positions : ndarray
        Shape (2, length) with x, y coordinates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.zeros((2, length))
    velocity = np.zeros(2)
    
    # Initialize at random position
    positions[:, 0] = np.random.uniform(bounds[0], bounds[1], 2)
    
    for t in range(1, length):
        # Random walk with momentum
        velocity = momentum * velocity + (1 - momentum) * np.random.randn(2) * step_size
        
        # Update position
        new_pos = positions[:, t-1] + velocity
        
        # Bounce off walls
        for dim in range(2):
            if new_pos[dim] < bounds[0]:
                new_pos[dim] = bounds[0]
                velocity[dim] = -velocity[dim]
            elif new_pos[dim] > bounds[1]:
                new_pos[dim] = bounds[1]
                velocity[dim] = -velocity[dim]
        
        positions[:, t] = new_pos
    
    return positions


def gaussian_place_field(positions, center, sigma=0.1):
    """
    Calculate neural response using 2D Gaussian place field.
    
    Parameters
    ----------
    positions : ndarray
        Shape (2, n_timepoints) with x, y coordinates.
    center : ndarray
        Shape (2,) with place field center coordinates.
    sigma : float
        Width of the place field.
        
    Returns
    -------
    response : ndarray
        Neural response (firing rate modulation).
    """
    # Calculate squared distance from center
    dx = positions[0, :] - center[0]
    dy = positions[1, :] - center[1]
    dist_sq = dx**2 + dy**2
    
    # Gaussian response
    response = np.exp(-dist_sq / (2 * sigma**2))
    
    return response


def generate_2d_manifold_neurons(n_neurons, positions, field_sigma=0.1,
                                baseline_rate=0.1, peak_rate=1.0,
                                noise_std=0.05, grid_arrangement=True,
                                bounds=(0, 1), seed=None):
    """
    Generate population of place cells with 2D Gaussian place fields.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    positions : ndarray
        Shape (2, n_timepoints) with x, y positions.
    field_sigma : float
        Width of place fields. Default is 0.1.
    baseline_rate : float
        Baseline firing rate. Default is 0.1 Hz.
    peak_rate : float
        Peak firing rate at place field center. Default is 1.0 Hz.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float
        Noise in firing rates.
    grid_arrangement : bool
        If True, arrange place fields in a grid. Otherwise random.
    bounds : tuple
        (min, max) bounds for place field centers.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints) with firing rates.
    place_field_centers : ndarray
        Shape (n_neurons, 2) with place field centers.
    """
    # Validate firing rate
    validate_peak_rate(peak_rate, context="generate_2d_manifold_neurons")
    
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = positions.shape[1]
    
    # Generate place field centers
    if grid_arrangement:
        # Arrange in a grid
        n_per_side = int(np.ceil(np.sqrt(n_neurons)))
        x_centers = np.linspace(bounds[0] + 0.1, bounds[1] - 0.1, n_per_side)
        y_centers = np.linspace(bounds[0] + 0.1, bounds[1] - 0.1, n_per_side)
        
        centers = []
        for x in x_centers:
            for y in y_centers:
                centers.append([x, y])
                if len(centers) >= n_neurons:
                    break
            if len(centers) >= n_neurons:
                break
        
        place_field_centers = np.array(centers[:n_neurons])
        
        # Add small jitter
        jitter = np.random.normal(0, 0.02, place_field_centers.shape)
        place_field_centers += jitter
        place_field_centers = np.clip(place_field_centers, bounds[0], bounds[1])
    else:
        # Random placement
        place_field_centers = np.random.uniform(bounds[0], bounds[1], (n_neurons, 2))
    
    # Generate firing rates
    firing_rates = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Gaussian place field
        place_response = gaussian_place_field(positions, place_field_centers[i], field_sigma)
        
        # Scale to desired firing rate range
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response
        
        # Add noise
        noise = np.random.normal(0, noise_std, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)
        
        firing_rates[i, :] = firing_rate
    
    return firing_rates, place_field_centers


def generate_2d_manifold_data(n_neurons, duration=600, sampling_rate=20.0,
                             field_sigma=0.1, step_size=0.02, momentum=0.8,
                             baseline_rate=0.1, peak_rate=1.0,
                             noise_std=0.05, grid_arrangement=True,
                             decay_time=2.0, calcium_noise_std=0.1,
                             bounds=(0, 1), seed=None, verbose=True):
    """
    Generate synthetic data with neurons on 2D spatial manifold (place cells).
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    duration : float
        Duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    field_sigma : float
        Width of place fields.
    step_size : float
        Step size for random walk.
    momentum : float
        Momentum for smoother trajectories.
    baseline_rate : float
        Baseline firing rate. Default is 0.1 Hz.
    peak_rate : float
        Peak firing rate. Default is 1.0 Hz.
    noise_std : float
        Firing rate noise.
    grid_arrangement : bool
        Arrange place fields in grid.
    decay_time : float
        Calcium decay time.
    calcium_noise_std : float
        Calcium signal noise.
    bounds : tuple
        Spatial bounds.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
        
    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    positions : ndarray
        Position trajectory (2 x n_timepoints).
    place_field_centers : ndarray
        Place field centers (n_neurons x 2).
    firing_rates : ndarray
        Underlying firing rates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = int(duration * sampling_rate)
    
    if verbose:
        print(f'Generating 2D spatial manifold data: {n_neurons} neurons, {duration}s')
    
    # Generate spatial trajectory
    if verbose:
        print('  Generating 2D random walk trajectory...')
    positions = generate_2d_random_walk(n_timepoints, bounds, step_size, momentum, seed)
    
    # Generate neural responses
    if verbose:
        print('  Generating neural responses with place fields...')
    firing_rates, place_field_centers = generate_2d_manifold_neurons(
        n_neurons, positions, field_sigma,
        baseline_rate, peak_rate, noise_std,
        grid_arrangement, bounds,
        seed=(seed + 1) if seed else None
    )
    
    # Convert to calcium signals
    if verbose:
        print('  Converting to calcium signals...')
    calcium_signals = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Generate Poisson events
        prob_spike = firing_rates[i, :] / sampling_rate
        prob_spike = np.clip(prob_spike, 0, 1)
        events = np.random.binomial(1, prob_spike)
        
        # Convert to calcium
        calcium_signal = generate_pseudo_calcium_signal(
            events=events,
            duration=duration,
            sampling_rate=sampling_rate,
            amplitude_range=(0.5, 2.0),
            decay_time=decay_time,
            noise_std=calcium_noise_std
        )
        calcium_signals[i, :] = calcium_signal
    
    if verbose:
        print('  Done!')
    
    return calcium_signals, positions, place_field_centers, firing_rates


def generate_2d_manifold_exp(n_neurons=100, duration=600, fps=20.0,
                            field_sigma=0.1, step_size=0.02, momentum=0.8,
                            baseline_rate=0.1, peak_rate=1.0,
                            noise_std=0.05, grid_arrangement=True,
                            decay_time=2.0, calcium_noise_std=0.1,
                            bounds=(0, 1), seed=None, verbose=True, return_info=False):
    """
    Generate complete experiment with 2D spatial manifold (place cells).
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    duration : float
        Duration in seconds.
    fps : float
        Sampling rate.
    field_sigma : float
        Place field width.
    step_size : float
        Random walk step size.
    momentum : float
        Trajectory smoothness.
    baseline_rate : float
        Baseline firing rate. Default is 0.1 Hz.
    peak_rate : float
        Peak firing rate. Default is 1.0 Hz.
    noise_std : float
        Firing rate noise.
    grid_arrangement : bool
        Grid arrangement of place fields.
    decay_time : float
        Calcium decay time.
    calcium_noise_std : float
        Calcium noise.
    bounds : tuple
        Spatial bounds.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
    return_info : bool
        If True, return (exp, info) tuple with additional information.
        
    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with 2D spatial manifold data.
    info : dict, optional
        If return_info=True, dictionary with manifold info.
    """
    # Calculate effective decay time for shuffle mask
    effective_decay_time = get_effective_decay_time(decay_time, duration, verbose)
    
    # Generate data
    calcium, positions, place_field_centers, firing_rates = generate_2d_manifold_data(
        n_neurons=n_neurons,
        duration=duration,
        sampling_rate=fps,
        field_sigma=field_sigma,
        step_size=step_size,
        momentum=momentum,
        baseline_rate=baseline_rate,
        peak_rate=peak_rate,
        noise_std=noise_std,
        grid_arrangement=grid_arrangement,
        decay_time=decay_time,
        calcium_noise_std=calcium_noise_std,
        bounds=bounds,
        seed=seed,
        verbose=verbose
    )
    
    # Create static features
    static_features = {
        'fps': fps,
        't_rise_sec': 0.04,
        't_off_sec': effective_decay_time,  # Use effective decay time for shuffle mask
        'manifold_type': '2d_spatial',
        'field_sigma': field_sigma,
        'baseline_rate': baseline_rate,
        'peak_rate': peak_rate,
        'grid_arrangement': grid_arrangement,
    }
    
    # Create dynamic features
    position_ts = MultiTimeSeries(
        [TimeSeries(positions[0, :], discrete=False),
         TimeSeries(positions[1, :], discrete=False)]
    )
    
    # Also create separate x, y features
    x_ts = TimeSeries(
        data=positions[0, :],
        discrete=False
    )
    
    y_ts = TimeSeries(
        data=positions[1, :],
        discrete=False
    )
    
    dynamic_features = {
        'position_2d': position_ts,
        'x': x_ts,
        'y': y_ts
    }
    
    # Store additional information
    static_features['place_field_centers'] = place_field_centers
    
    # Create experiment
    exp = Experiment(
        signature='2d_spatial_manifold_exp',
        calcium=calcium,
        spikes=None,
        static_features=static_features,
        dynamic_features=dynamic_features,
        exp_identificators={
            'manifold': '2d_spatial',
            'n_neurons': n_neurons,
            'duration': duration
        }
    )
    
    # Store firing rates
    exp.firing_rates = firing_rates
    
    # Create info dictionary if requested
    if return_info:
        info = {
            'manifold_type': '2d_spatial',
            'n_neurons': n_neurons,
            'positions': positions,
            'place_field_centers': place_field_centers,
            'firing_rates': firing_rates,
            'parameters': {
                'field_sigma': field_sigma,
                'step_size': step_size,
                'momentum': momentum,
                'baseline_rate': baseline_rate,
                'peak_rate': peak_rate,
                'noise_std': noise_std,
                'grid_arrangement': grid_arrangement,
                'decay_time': decay_time,
                'calcium_noise_std': calcium_noise_std,
                'bounds': bounds
            }
        }
        return exp, info
    
    return exp