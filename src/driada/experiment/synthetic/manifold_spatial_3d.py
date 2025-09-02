"""
3D spatial manifold generation for place cells.

This module contains functions for generating synthetic neural data on 3D spatial
manifolds, typically used to model place cells in 3D environments.
"""

import numpy as np
from .core import validate_peak_rate, generate_pseudo_calcium_signal
from .utils import get_effective_decay_time
from ..exp_base import Experiment
from ...information.info_base import TimeSeries, MultiTimeSeries
from ...utils.data import check_positive, check_nonnegative


def generate_3d_random_walk(
    length, bounds=(0, 1), step_size=0.02, momentum=0.8, seed=None
):
    """
    Generate a 3D random walk trajectory within bounded region.

    Creates a smooth random walk in 3D space using momentum-based updates
    with reflective boundary conditions. The walker starts at a random
    position and moves with inertia, bouncing off walls elastically.

    Parameters
    ----------
    length : int
        Number of time points. Must be positive.
    bounds : tuple, optional
        (min, max) bounds for x, y, and z coordinates. Default is (0, 1).
        Must have bounds[0] < bounds[1].
    step_size : float, optional
        Step size for random walk. Must be positive. Default is 0.02.
        Typical values: 0.01-0.1 relative to bounds size.
    momentum : float, optional
        Momentum factor for smoother trajectories. Must be in [0, 1].
        Default is 0.8. Higher values give smoother, more continuous paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    positions : ndarray
        Shape (3, length) with x, y, z coordinates. Each row contains
        x, y, and z coordinates respectively.

    Raises
    ------
    ValueError
        If length is not positive, step_size is not positive, momentum is
        not in [0, 1], or bounds[0] >= bounds[1].
    TypeError
        If inputs are not numeric types.

    Notes
    -----
    The walk follows:
    velocity = momentum * velocity + (1 - momentum) * N(0, step_size)
    position[t] = position[t-1] + velocity
    
    When hitting boundaries, the relevant velocity component is reversed
    to simulate elastic collision.    """
    # Input validation
    if not isinstance(length, (int, np.integer)):
        raise TypeError("length must be an integer")
    check_positive(length=length, step_size=step_size)
    
    if not isinstance(momentum, (int, float)):
        raise TypeError("momentum must be numeric")
    if not 0 <= momentum <= 1:
        raise ValueError("momentum must be in range [0, 1]")
    
    if len(bounds) != 2:
        raise ValueError("bounds must be a tuple of (min, max)")
    if bounds[0] >= bounds[1]:
        raise ValueError("bounds must have min < max")
    
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.zeros((3, length))
    velocity = np.zeros(3)

    # Initialize at random position
    positions[:, 0] = np.random.uniform(bounds[0], bounds[1], 3)

    for t in range(1, length):
        # Random walk with momentum
        velocity = momentum * velocity + (1 - momentum) * np.random.randn(3) * step_size

        # Update position
        new_pos = positions[:, t - 1] + velocity

        # Bounce off walls
        for dim in range(3):
            if new_pos[dim] < bounds[0]:
                new_pos[dim] = bounds[0]
                velocity[dim] = -velocity[dim]
            elif new_pos[dim] > bounds[1]:
                new_pos[dim] = bounds[1]
                velocity[dim] = -velocity[dim]

        positions[:, t] = new_pos

    return positions


def gaussian_place_field_3d(positions, center, sigma=0.1):
    """
    Calculate neural response using 3D Gaussian place field.

    Implements an isotropic 3D Gaussian receptive field commonly used to
    model place cells in volumetric environments. The response peaks at
    the field center and falls off with a Gaussian profile.

    Parameters
    ----------
    positions : ndarray
        Shape (3, n_timepoints) with x, y, z coordinates OR
        Shape (n_positions, 3) with positions in rows.
    center : ndarray
        Shape (3,) with place field center coordinates [x, y, z].
    sigma : float, optional
        Width (standard deviation) of the place field. Must be positive.
        Default is 0.1. Larger values give wider fields.

    Returns
    -------
    response : ndarray
        Neural response (firing rate modulation) in range [0, 1].
        Shape matches input positions format. Maximum value is 1.0
        at field center.

    Raises
    ------
    ValueError
        If sigma is not positive, or if center shape is not (3,).
    TypeError
        If inputs are not numeric arrays.

    Notes
    -----
    The response follows a 3D Gaussian:
    response = exp(-((x-cx)² + (y-cy)² + (z-cz)²) / (2σ²))
    where (cx, cy, cz) is the field center and σ is the width.
    
    This function flexibly handles both common position formats:
    - (3, n_timepoints): positions as columns
    - (n_positions, 3): positions as rows    """
    # Input validation
    positions = np.asarray(positions)
    center = np.asarray(center)
    
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    
    check_positive(sigma=sigma)
    
    # Handle both input formats
    if positions.shape[0] == 3 and positions.shape[1] != 3:
        # Format: (3, n_timepoints)
        dx = positions[0, :] - center[0]
        dy = positions[1, :] - center[1]
        dz = positions[2, :] - center[2]
    else:
        # Format: (n_positions, 3)
        dx = positions[:, 0] - center[0]
        dy = positions[:, 1] - center[1]
        dz = positions[:, 2] - center[2]

    dist_sq = dx**2 + dy**2 + dz**2

    # Gaussian response
    response = np.exp(-dist_sq / (2 * sigma**2))

    return response


def generate_3d_manifold_neurons(
    n_neurons,
    positions,
    field_sigma=0.1,
    baseline_rate=0.1,
    peak_rate=1.0,
    noise_std=0.05,
    grid_arrangement=True,
    bounds=(0, 1),
    seed=None,
):
    """
    Generate population of place cells with 3D Gaussian place fields.

    Creates a population of neurons with place fields either arranged in a
    regular 3D grid or randomly distributed. Each neuron responds maximally
    when the animal is at its place field center in 3D space.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population. Must be positive.
    positions : ndarray
        Shape (3, n_timepoints) with x, y, z positions.
    field_sigma : float, optional
        Width of place fields. Must be positive. Default is 0.1.
    baseline_rate : float, optional
        Baseline firing rate in Hz. Must be non-negative. Default is 0.1.
        Should be less than peak_rate.
    peak_rate : float, optional
        Peak firing rate at place field center in Hz. Default is 1.0.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float, optional
        Standard deviation of Gaussian noise in firing rates.
        Must be non-negative. Default is 0.05.
    grid_arrangement : bool, optional
        If True, arrange place fields in a 3D grid. Otherwise random.
        Default is True.
    bounds : tuple, optional
        (min, max) bounds for place field centers. Default is (0, 1).
        Must have bounds[0] < bounds[1].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints) with firing rates in Hz.
        All values are non-negative.
    place_field_centers : ndarray
        Shape (n_neurons, 3) with x, y, z coordinates of place field centers.

    Raises
    ------
    ValueError
        If peak_rate is invalid, n_neurons is not positive, field_sigma
        is not positive, noise_std is negative, baseline_rate > peak_rate,
        or bounds are invalid.
    TypeError
        If inputs are not correct types.

    Notes
    -----
    Grid arrangement places neurons on a cubic grid with 0.1 margin from
    boundaries. The grid uses n^(1/3) neurons per side and adds small
    jitter (std=0.02) to break regularity.
    
    Firing rates are computed as:
    rate = baseline + (peak - baseline) * gaussian_place_field_3d + noise    """
    # Input validation
    check_positive(n_neurons=n_neurons, field_sigma=field_sigma)
    check_nonnegative(baseline_rate=baseline_rate, noise_std=noise_std)
    
    # Validate firing rate
    validate_peak_rate(peak_rate, context="generate_3d_manifold_neurons")
    
    # Check parameter relationships
    if baseline_rate > peak_rate:
        raise ValueError(f"baseline_rate ({baseline_rate}) must be <= peak_rate ({peak_rate})")
    
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be (min, max) with min < max")

    if seed is not None:
        np.random.seed(seed)

    positions = np.asarray(positions)
    if positions.shape[0] != 3:
        raise ValueError("positions must have shape (3, n_timepoints)")
    n_timepoints = positions.shape[1]

    # Generate place field centers
    if grid_arrangement:
        # Arrange in a 3D grid
        n_per_side = int(np.ceil(n_neurons ** (1 / 3)))
        x_centers = np.linspace(bounds[0] + 0.1, bounds[1] - 0.1, n_per_side)
        y_centers = np.linspace(bounds[0] + 0.1, bounds[1] - 0.1, n_per_side)
        z_centers = np.linspace(bounds[0] + 0.1, bounds[1] - 0.1, n_per_side)

        centers = []
        for x in x_centers:
            for y in y_centers:
                for z in z_centers:
                    centers.append([x, y, z])
                    if len(centers) >= n_neurons:
                        break
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
        place_field_centers = np.random.uniform(bounds[0], bounds[1], (n_neurons, 3))

    # Generate firing rates
    firing_rates = np.zeros((n_neurons, n_timepoints))

    for i in range(n_neurons):
        # Gaussian place field
        place_response = gaussian_place_field_3d(
            positions, place_field_centers[i], field_sigma
        )

        # Scale to desired firing rate range
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response

        # Add noise
        noise = np.random.normal(0, noise_std, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)

        firing_rates[i, :] = firing_rate

    return firing_rates, place_field_centers


def generate_3d_manifold_data(
    n_neurons,
    duration=600,
    sampling_rate=20.0,
    field_sigma=0.1,
    step_size=0.02,
    momentum=0.8,
    baseline_rate=0.1,
    peak_rate=1.0,
    noise_std=0.05,
    grid_arrangement=True,
    decay_time=2.0,
    calcium_noise_std=0.1,
    bounds=(0, 1),
    seed=None,
    verbose=True,
):
    """
    Generate synthetic data with neurons on 3D spatial manifold (place cells).

    Creates a complete dataset including 3D spatial trajectory, place cell
    responses, and realistic calcium imaging signals. Useful for testing
    3D spatial coding analyses.

    Parameters
    ----------
    n_neurons : int
        Number of neurons. Must be positive.
    duration : float, optional
        Duration in seconds. Must be positive. Default is 600.
    sampling_rate : float, optional
        Sampling rate in Hz. Must be positive. Default is 20.0.
    field_sigma : float, optional
        Width of place fields. Must be positive. Default is 0.1.
    step_size : float, optional
        Step size for random walk. Must be positive. Default is 0.02.
    momentum : float, optional
        Momentum for smoother trajectories. Must be in [0, 1]. Default is 0.8.
    baseline_rate : float, optional
        Baseline firing rate in Hz. Must be non-negative. Default is 0.1.
    peak_rate : float, optional
        Peak firing rate in Hz. Default is 1.0.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float, optional
        Firing rate noise. Must be non-negative. Default is 0.05.
    grid_arrangement : bool, optional
        If True, arrange place fields in 3D grid. Default is True.
    decay_time : float, optional
        Calcium decay time constant in seconds. Must be positive. Default is 2.0.
    calcium_noise_std : float, optional
        Calcium signal noise. Must be non-negative. Default is 0.1.
    bounds : tuple, optional
        Spatial bounds (min, max) for all dimensions. Default is (0, 1).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default is True.

    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    positions : ndarray
        Position trajectory (3 x n_timepoints) with x, y, z coordinates.
    place_field_centers : ndarray
        Place field centers (n_neurons x 3) with x, y, z coordinates.
    firing_rates : ndarray
        Underlying firing rates in Hz (n_neurons x n_timepoints).

    Raises
    ------
    ValueError
        If any positive parameters are not positive, if any non-negative
        parameters are negative, or if momentum is not in [0, 1].

    Notes
    -----
    The generation process:
    1. Creates 3D random walk trajectory with momentum
    2. Generates place cell responses based on 3D distance to place fields
    3. Converts firing rates to spike probabilities
    4. Samples spikes using binomial distribution
    5. Convolves spikes with calcium kernel and adds noise    """
    # Input validation
    check_positive(n_neurons=n_neurons, duration=duration, sampling_rate=sampling_rate,
                  field_sigma=field_sigma, step_size=step_size, decay_time=decay_time)
    check_nonnegative(baseline_rate=baseline_rate, noise_std=noise_std,
                     calcium_noise_std=calcium_noise_std)
    
    if not isinstance(momentum, (int, float)):
        raise TypeError("momentum must be numeric")
    if not 0 <= momentum <= 1:
        raise ValueError("momentum must be in range [0, 1]")
    
    if seed is not None:
        np.random.seed(seed)

    n_timepoints = int(duration * sampling_rate)

    if verbose:
        print(f"Generating 3D spatial manifold data: {n_neurons} neurons, {duration}s")

    # Generate spatial trajectory
    if verbose:
        print("  Generating 3D random walk trajectory...")
    positions = generate_3d_random_walk(n_timepoints, bounds, step_size, momentum, seed)

    # Generate neural responses
    if verbose:
        print("  Generating neural responses with 3D place fields...")
    firing_rates, place_field_centers = generate_3d_manifold_neurons(
        n_neurons,
        positions,
        field_sigma,
        baseline_rate,
        peak_rate,
        noise_std,
        grid_arrangement,
        bounds,
        seed=(seed + 1) if seed is not None else None,
    )

    # Convert to calcium signals
    if verbose:
        print("  Converting to calcium signals...")
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
            noise_std=calcium_noise_std,
        )
        calcium_signals[i, :] = calcium_signal

    if verbose:
        print("  Done!")

    return calcium_signals, positions, place_field_centers, firing_rates


def generate_3d_manifold_exp(
    n_neurons=125,
    duration=600,
    fps=20.0,
    field_sigma=0.1,
    step_size=0.02,
    momentum=0.8,
    baseline_rate=0.1,
    peak_rate=1.0,
    noise_std=0.05,
    grid_arrangement=True,
    decay_time=2.0,
    calcium_noise_std=0.1,
    bounds=(0, 1),
    seed=None,
    verbose=True,
    return_info=False,
):
    """
    Generate complete experiment with 3D spatial manifold (place cells).

    Creates a DRIADA Experiment object with synthetic 3D place cell data,
    including calcium imaging signals and 3D behavioral trajectory.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons. Must be positive. Default is 125 (5x5x5 grid).
    duration : float, optional
        Duration in seconds. Must be positive. Default is 600.
    fps : float, optional
        Sampling rate (frames per second). Must be positive. Default is 20.0.
    field_sigma : float, optional
        Place field width. Must be positive. Default is 0.1.
    step_size : float, optional
        Random walk step size. Must be positive. Default is 0.02.
    momentum : float, optional
        Trajectory smoothness factor. Must be in [0, 1]. Default is 0.8.
    baseline_rate : float, optional
        Baseline firing rate in Hz. Must be non-negative. Default is 0.1.
    peak_rate : float, optional
        Peak firing rate in Hz. Default is 1.0.
        Values >2 Hz may cause calcium signal saturation.
    noise_std : float, optional
        Firing rate noise. Must be non-negative. Default is 0.05.
    grid_arrangement : bool, optional
        If True, arrange place fields in 3D grid. Default is True.
    decay_time : float, optional
        Calcium decay time in seconds. Must be positive. Default is 2.0.
    calcium_noise_std : float, optional
        Calcium signal noise. Must be non-negative. Default is 0.1.
    bounds : tuple, optional
        Spatial bounds (min, max) for all dimensions. Default is (0, 1).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default is True.
    return_info : bool, optional
        If True, return (exp, info) tuple with additional information.
        Default is False.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with 3D spatial manifold data.
    info : dict, optional
        Only returned if return_info=True. Contains:
        - manifold_type: '3d_spatial'
        - n_neurons: Number of neurons
        - positions: 3D trajectory (n_frames, 3)
        - place_field_centers: 3D place field centers (n_neurons, 3)
        - firing_rates: Raw firing rates (n_neurons, n_frames)
        - parameters: Dictionary of all parameters used

    Raises
    ------
    ValueError
        If any positive parameters are not positive, if any non-negative
        parameters are negative, if momentum is not in [0, 1], or if
        bounds are invalid.
    TypeError
        If inputs are not correct types.

    Notes
    -----
    The experiment includes:
    - Static features: fps, decay time, manifold info, place field centers
    - Dynamic features: position_3d (MultiTimeSeries), x, y, and z (TimeSeries)
    - Calcium signals with realistic noise and dynamics
    - Underlying firing rates attached to exp object
    
    For short experiments (duration ≤ 30s), the decay time is automatically
    adjusted to prevent shuffle mask issues.    """
    # Calculate effective decay time for shuffle mask
    effective_decay_time = get_effective_decay_time(decay_time, duration, verbose)

    # Generate data
    calcium, positions, place_field_centers, firing_rates = generate_3d_manifold_data(
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
        verbose=verbose,
    )

    # Create static features
    static_features = {
        "fps": fps,
        "t_rise_sec": 0.04,
        "t_off_sec": effective_decay_time,  # Use effective decay time for shuffle mask
        "manifold_type": "3d_spatial",
        "field_sigma": field_sigma,
        "baseline_rate": baseline_rate,
        "peak_rate": peak_rate,
        "grid_arrangement": grid_arrangement,
    }

    # Create dynamic features
    position_ts = MultiTimeSeries(
        [
            TimeSeries(positions[0, :], discrete=False),
            TimeSeries(positions[1, :], discrete=False),
            TimeSeries(positions[2, :], discrete=False),
        ]
    )

    # Also create separate x, y, z features
    x_ts = TimeSeries(data=positions[0, :], discrete=False)

    y_ts = TimeSeries(data=positions[1, :], discrete=False)

    z_ts = TimeSeries(data=positions[2, :], discrete=False)

    dynamic_features = {"position_3d": position_ts, "x": x_ts, "y": y_ts, "z": z_ts}

    # Store additional information
    static_features["place_field_centers"] = place_field_centers

    # Create experiment
    exp = Experiment(
        signature="3d_spatial_manifold_exp",
        calcium=calcium,
        spikes=None,
        static_features=static_features,
        dynamic_features=dynamic_features,
        exp_identificators={
            "manifold": "3d_spatial",
            "n_neurons": n_neurons,
            "duration": duration,
        },
    )

    # Store firing rates
    exp.firing_rates = firing_rates

    # Create info dictionary if requested
    if return_info:
        info = {
            "manifold_type": "3d_spatial",
            "n_neurons": n_neurons,
            "positions": positions,
            "place_field_centers": place_field_centers,
            "firing_rates": firing_rates,
            "parameters": {
                "field_sigma": field_sigma,
                "step_size": step_size,
                "momentum": momentum,
                "baseline_rate": baseline_rate,
                "peak_rate": peak_rate,
                "noise_std": noise_std,
                "grid_arrangement": grid_arrangement,
                "decay_time": decay_time,
                "calcium_noise_std": calcium_noise_std,
                "bounds": bounds,
            },
        }
        return exp, info

    return exp
