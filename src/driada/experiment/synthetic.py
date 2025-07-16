import numpy as np
from fbm import FBM
import itertools
import tqdm
from scipy import stats
from scipy.special import i0
from .exp_base import *
from ..information.info_base import TimeSeries, MultiTimeSeries, aggregate_multiple_ts


# Circular manifold generation functions for head direction cells
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


def generate_circular_manifold_neurons(n_neurons, head_direction, kappa=4.0, 
                                      baseline_rate=0.1, peak_rate=2.0,
                                      noise_std=0.05, seed=None):
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
    peak_rate : float
        Peak firing rate at preferred direction.
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
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = len(head_direction)
    
    # Uniformly distribute preferred directions around the circle
    preferred_directions = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
    
    # Add small random jitter to break perfect symmetry
    jitter = np.random.normal(0, 0.1, n_neurons)
    preferred_directions = (preferred_directions + jitter) % (2*np.pi)
    
    # Generate firing rates for each neuron
    firing_rates = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Von Mises tuning curve
        tuning_response = von_mises_tuning_curve(head_direction, 
                                               preferred_directions[i], 
                                               kappa)
        
        # Scale to desired firing rate range
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * tuning_response
        
        # Add noise
        noise = np.random.normal(0, noise_std, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)  # Ensure non-negative
        
        firing_rates[i, :] = firing_rate
    
    return firing_rates, preferred_directions


def generate_circular_manifold_data(n_neurons, duration=600, sampling_rate=20.0,
                                   kappa=4.0, step_std=0.1,
                                   baseline_rate=0.1, peak_rate=2.0,
                                   noise_std=0.05, 
                                   decay_time=2.0, calcium_noise_std=0.1,
                                   seed=None, verbose=True):
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
        Baseline firing rate.
    peak_rate : float
        Peak firing rate at preferred direction.
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
        print(f'Generating circular manifold data: {n_neurons} neurons, {duration}s')
    
    # Generate head direction trajectory
    if verbose:
        print('  Generating head direction trajectory...')
    head_direction = generate_circular_random_walk(n_timepoints, step_std, seed)
    
    # Generate neural responses
    if verbose:
        print('  Generating neural responses with Von Mises tuning...')
    firing_rates, preferred_directions = generate_circular_manifold_neurons(
        n_neurons, head_direction, kappa,
        baseline_rate, peak_rate, noise_std,
        seed=(seed + 1) if seed else None
    )
    
    # Convert firing rates to calcium signals
    if verbose:
        print('  Converting to calcium signals...')
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
            noise_std=calcium_noise_std
        )
        calcium_signals[i, :] = calcium_signal
    
    if verbose:
        print('  Done!')
    
    return calcium_signals, head_direction, preferred_directions, firing_rates


# 2D spatial manifold generation functions for place cells
def generate_2d_random_walk(length, bounds=(0, 1), step_size=0.02, momentum=0.8, seed=None):
    """
    Generate a 2D random walk trajectory with momentum.
    
    Parameters
    ----------
    length : int
        Number of time points.
    bounds : tuple
        (min, max) boundaries for the 2D space.
    step_size : float
        Step size for movement.
    momentum : float
        Momentum factor (0-1) for smoother trajectories.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    positions : ndarray
        Shape (length, 2) with (x, y) positions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.zeros((length, 2))
    velocity = np.zeros(2)
    
    # Start at random position
    positions[0] = np.random.uniform(bounds[0], bounds[1], 2)
    
    for t in range(1, length):
        # Random acceleration
        acceleration = np.random.randn(2) * step_size
        
        # Update velocity with momentum
        velocity = momentum * velocity + (1 - momentum) * acceleration
        
        # Update position
        new_pos = positions[t-1] + velocity
        
        # Bounce off walls
        for dim in range(2):
            if new_pos[dim] < bounds[0]:
                new_pos[dim] = bounds[0]
                velocity[dim] *= -0.5
            elif new_pos[dim] > bounds[1]:
                new_pos[dim] = bounds[1]
                velocity[dim] *= -0.5
        
        positions[t] = new_pos
    
    return positions


def gaussian_place_field(positions, center, sigma=0.1):
    """
    Calculate neural response using 2D Gaussian place field.
    
    Parameters
    ----------
    positions : ndarray
        Shape (n_timepoints, 2) with (x, y) positions.
    center : ndarray
        (x, y) center of the place field.
    sigma : float
        Width of the Gaussian place field.
        
    Returns
    -------
    response : ndarray
        Neural response at each position.
    """
    distances_squared = np.sum((positions - center)**2, axis=1)
    response = np.exp(-distances_squared / (2 * sigma**2))
    return response


def generate_2d_manifold_neurons(n_neurons, positions, field_sigma=0.1,
                                baseline_rate=0.1, peak_rate=2.0,
                                noise_std=0.05, grid_arrangement=True,
                                seed=None):
    """
    Generate population of place cells with 2D Gaussian tuning.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    positions : ndarray
        Shape (n_timepoints, 2) with (x, y) trajectory.
    field_sigma : float
        Width of place fields.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate at place field center.
    noise_std : float
        Noise in firing rates.
    grid_arrangement : bool
        If True, arrange place fields on a grid. If False, random positions.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints).
    place_field_centers : ndarray
        Shape (n_neurons, 2) with place field centers.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = len(positions)
    
    # Determine place field centers
    if grid_arrangement:
        # Arrange on a grid - ensure good coverage
        grid_size = int(np.ceil(np.sqrt(n_neurons)))
        # Use full space for better coverage
        x_grid = np.linspace(0.05, 0.95, grid_size)
        y_grid = np.linspace(0.05, 0.95, grid_size)
        centers = []
        for x in x_grid:
            for y in y_grid:
                if len(centers) < n_neurons:
                    centers.append([x, y])
        place_field_centers = np.array(centers[:n_neurons])
        
        # Add small jitter
        jitter = np.random.normal(0, 0.01, (n_neurons, 2))
        place_field_centers += jitter
        place_field_centers = np.clip(place_field_centers, 0.02, 0.98)
    else:
        # Random positions - better spread
        place_field_centers = np.random.uniform(0.05, 0.95, (n_neurons, 2))
    
    # Generate firing rates
    firing_rates = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Gaussian place field response
        place_response = gaussian_place_field(positions, place_field_centers[i], field_sigma)
        
        # Scale to firing rates
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response
        
        # Add noise
        noise = np.random.normal(0, noise_std, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)
        
        firing_rates[i] = firing_rate
    
    return firing_rates, place_field_centers


def generate_2d_manifold_data(n_neurons, duration=600, sampling_rate=20.0,
                             field_sigma=0.1, step_size=0.02, momentum=0.8,
                             baseline_rate=0.1, peak_rate=2.0,
                             noise_std=0.05, decay_time=2.0, 
                             calcium_noise_std=0.1,
                             grid_arrangement=True,
                             n_environments=1,
                             seed=None, verbose=True):
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
        Step size for trajectory.
    momentum : float
        Momentum for smoother trajectories.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate in place field.
    noise_std : float
        Noise in firing rates.
    decay_time : float
        Calcium decay time.
    calcium_noise_std : float
        Calcium signal noise.
    grid_arrangement : bool
        If True, arrange place fields on grid.
    n_environments : int
        Number of different environments (for remapping).
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
        
    Returns
    -------
    calcium_signals : ndarray
        Shape (n_neurons, n_timepoints).
    positions : ndarray or list
        Positions for each environment.
    place_field_centers : ndarray or list
        Place field centers for each environment.
    firing_rates : ndarray
        Underlying firing rates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = int(duration * sampling_rate)
    
    if verbose:
        print(f'Generating 2D manifold data: {n_neurons} neurons, {duration}s, {n_environments} environment(s)')
    
    # Generate data for each environment
    all_calcium = []
    all_positions = []
    all_centers = []
    all_rates = []
    
    timepoints_per_env = n_timepoints // n_environments
    
    for env in range(n_environments):
        if verbose:
            print(f'  Environment {env+1}/{n_environments}:')
        
        # Generate trajectory
        if verbose:
            print('    Generating 2D trajectory...')
        positions = generate_2d_random_walk(
            timepoints_per_env, 
            step_size=step_size,
            momentum=momentum,
            seed=(seed + env * 100) if seed else None
        )
        
        # Generate neural responses
        if verbose:
            print('    Generating place cell responses...')
        
        # For remapping, shuffle place field centers
        if env > 0:
            # Partial remapping: some cells keep their fields, others remap
            remap_fraction = 0.5
            n_remap = int(n_neurons * remap_fraction)
            remap_indices = np.random.choice(n_neurons, n_remap, replace=False)
        
        firing_rates, centers = generate_2d_manifold_neurons(
            n_neurons, positions, field_sigma,
            baseline_rate, peak_rate, noise_std,
            grid_arrangement,
            seed=(seed + env * 200) if seed else None
        )
        
        # Apply remapping if not first environment
        if env > 0:
            # Keep some place fields, randomize others
            centers = all_centers[0].copy()
            centers[remap_indices] = np.random.uniform(0.1, 0.9, (n_remap, 2))
            
            # Recalculate firing rates with new centers
            firing_rates = np.zeros((n_neurons, timepoints_per_env))
            for i in range(n_neurons):
                place_response = gaussian_place_field(positions, centers[i], field_sigma)
                firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response
                noise = np.random.normal(0, noise_std, timepoints_per_env)
                firing_rates[i] = np.maximum(0, firing_rate + noise)
        
        all_positions.append(positions)
        all_centers.append(centers)
        all_rates.append(firing_rates)
    
    # Concatenate all environments
    positions = np.vstack(all_positions) if n_environments > 1 else all_positions[0]
    firing_rates = np.hstack(all_rates) if n_environments > 1 else all_rates[0]
    
    # Convert to calcium signals
    if verbose:
        print('  Converting to calcium signals...')
    
    calcium_signals = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Generate events from firing rates
        prob_spike = firing_rates[i] / sampling_rate
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
        calcium_signals[i] = calcium_signal
    
    if verbose:
        print('  Done!')
    
    # Return appropriate format
    if n_environments == 1:
        return calcium_signals, positions, all_centers[0], firing_rates
    else:
        return calcium_signals, all_positions, all_centers, firing_rates


def generate_2d_manifold_exp(n_neurons=100, duration=600, fps=20.0,
                            field_sigma=0.1, step_size=0.02, momentum=0.8,
                            baseline_rate=0.1, peak_rate=2.0,
                            noise_std=0.05, decay_time=2.0,
                            calcium_noise_std=0.1,
                            grid_arrangement=True,
                            n_environments=1,
                            add_head_direction=False,
                            seed=None, verbose=True):
    """
    Generate synthetic experiment with place cells on 2D spatial manifold.
    
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
        Movement step size.
    momentum : float
        Movement momentum.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate.
    noise_std : float
        Firing rate noise.
    decay_time : float
        Calcium decay time.
    calcium_noise_std : float
        Calcium signal noise.
    grid_arrangement : bool
        Arrange fields on grid.
    n_environments : int
        Number of environments.
    add_head_direction : bool
        If True, also add head direction as a feature.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
        
    Returns
    -------
    exp : Experiment
        Experiment object.
    info : dict
        Information about the generated data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate 2D manifold data
    calcium_signals, positions, centers, firing_rates = generate_2d_manifold_data(
        n_neurons, duration, fps,
        field_sigma, step_size, momentum,
        baseline_rate, peak_rate,
        noise_std, decay_time, calcium_noise_std,
        grid_arrangement, n_environments,
        seed, verbose
    )
    
    # Create dynamic features
    dynamic_features = {}
    
    if n_environments == 1:
        # Single environment
        dynamic_features['x_position'] = TimeSeries(positions[:, 0], discrete=False)
        dynamic_features['y_position'] = TimeSeries(positions[:, 1], discrete=False)
        
        # Also add as multifeature for manifold analysis
        dynamic_features['position_2d'] = MultiTimeSeries([
            TimeSeries(positions[:, 0], discrete=False),
            TimeSeries(positions[:, 1], discrete=False)
        ])
    else:
        # Multiple environments - concatenate all
        all_x = np.concatenate([pos[:, 0] for pos in positions])
        all_y = np.concatenate([pos[:, 1] for pos in positions])
        
        dynamic_features['x_position'] = TimeSeries(all_x, discrete=False)
        dynamic_features['y_position'] = TimeSeries(all_y, discrete=False)
        
        dynamic_features['position_2d'] = MultiTimeSeries([
            TimeSeries(all_x, discrete=False),
            TimeSeries(all_y, discrete=False)
        ])
        
        # Add environment indicator
        env_indicator = np.concatenate([
            np.full(len(pos), env_idx) 
            for env_idx, pos in enumerate(positions)
        ])
        dynamic_features['environment'] = TimeSeries(env_indicator, discrete=True)
    
    # Optionally add head direction
    if add_head_direction:
        if verbose:
            print('  Adding head direction feature...')
        
        # Calculate head direction from trajectory
        if n_environments == 1:
            velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        else:
            all_positions = np.vstack(positions)
            velocities = np.diff(all_positions, axis=0, prepend=all_positions[0:1])
        
        head_direction = np.arctan2(velocities[:, 1], velocities[:, 0])
        head_direction = (head_direction + 2 * np.pi) % (2 * np.pi)
        
        dynamic_features['head_direction'] = TimeSeries(head_direction, discrete=False)
        
        # Add circular representation
        dynamic_features['head_direction_circular'] = MultiTimeSeries([
            TimeSeries(np.cos(head_direction), discrete=False),
            TimeSeries(np.sin(head_direction), discrete=False)
        ])
    
    # Create static features
    static_features = {
        'fps': fps,
        't_rise_sec': 0.5,
        't_off_sec': decay_time
    }
    
    # Create experiment
    exp = Experiment(
        'SpatialManifold2D',
        calcium_signals,
        None,
        {},
        static_features,
        dynamic_features,
        reconstruct_spikes=None
    )
    
    # Prepare info
    info = {
        'positions': positions,
        'place_field_centers': centers,
        'firing_rates': firing_rates,
        'field_sigma': field_sigma,
        'manifold_type': '2d_spatial',
        'n_neurons': n_neurons,
        'n_environments': n_environments
    }
    
    return exp, info


# 3D spatial manifold generation functions for flying/swimming animals
def generate_3d_random_walk(length, bounds=(0, 1), step_size=0.02, momentum=0.8, seed=None):
    """
    Generate a 3D random walk trajectory with momentum.
    
    Parameters
    ----------
    length : int
        Number of time points.
    bounds : tuple
        (min, max) boundaries for the 3D space.
    step_size : float
        Step size for movement.
    momentum : float
        Momentum factor (0-1) for smoother trajectories.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    positions : ndarray
        Shape (length, 3) with (x, y, z) positions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.zeros((length, 3))
    velocity = np.zeros(3)
    
    # Start at random position
    positions[0] = np.random.uniform(bounds[0], bounds[1], 3)
    
    for t in range(1, length):
        # Random acceleration
        acceleration = np.random.randn(3) * step_size
        
        # Update velocity with momentum
        velocity = momentum * velocity + (1 - momentum) * acceleration
        
        # Update position
        new_pos = positions[t-1] + velocity
        
        # Bounce off walls
        for dim in range(3):
            if new_pos[dim] < bounds[0]:
                new_pos[dim] = bounds[0]
                velocity[dim] *= -0.5
            elif new_pos[dim] > bounds[1]:
                new_pos[dim] = bounds[1]
                velocity[dim] *= -0.5
        
        positions[t] = new_pos
    
    return positions


def gaussian_place_field_3d(positions, center, sigma=0.1):
    """
    Calculate neural response using 3D Gaussian place field.
    
    Parameters
    ----------
    positions : ndarray
        Shape (n_timepoints, 3) with (x, y, z) positions.
    center : ndarray
        (x, y, z) center of the place field.
    sigma : float
        Width of the Gaussian place field.
        
    Returns
    -------
    response : ndarray
        Neural response at each position.
    """
    distances_squared = np.sum((positions - center)**2, axis=1)
    response = np.exp(-distances_squared / (2 * sigma**2))
    return response


def generate_3d_manifold_neurons(n_neurons, positions, field_sigma=0.1,
                                baseline_rate=0.1, peak_rate=2.0,
                                noise_std=0.05, grid_arrangement=True,
                                seed=None):
    """
    Generate population of 3D place cells with Gaussian tuning.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    positions : ndarray
        Shape (n_timepoints, 3) with (x, y, z) trajectory.
    field_sigma : float
        Width of place fields.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate at place field center.
    noise_std : float
        Noise in firing rates.
    grid_arrangement : bool
        If True, arrange place fields on a 3D grid. If False, random positions.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints).
    place_field_centers : ndarray
        Shape (n_neurons, 3) with place field centers.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = len(positions)
    
    # Determine place field centers
    if grid_arrangement:
        # Arrange on a 3D grid
        grid_size = int(np.ceil(n_neurons**(1/3)))
        # Use full space for better coverage
        x_grid = np.linspace(0.05, 0.95, grid_size)
        y_grid = np.linspace(0.05, 0.95, grid_size)
        z_grid = np.linspace(0.05, 0.95, grid_size)
        centers = []
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    if len(centers) < n_neurons:
                        centers.append([x, y, z])
        place_field_centers = np.array(centers[:n_neurons])
        
        # Add small jitter
        jitter = np.random.normal(0, 0.01, (n_neurons, 3))
        place_field_centers += jitter
        place_field_centers = np.clip(place_field_centers, 0.02, 0.98)
    else:
        # Random positions - better spread
        place_field_centers = np.random.uniform(0.05, 0.95, (n_neurons, 3))
    
    # Generate firing rates
    firing_rates = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Gaussian place field response
        place_response = gaussian_place_field_3d(positions, place_field_centers[i], field_sigma)
        
        # Scale to firing rates
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response
        
        # Add noise
        noise = np.random.normal(0, noise_std, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)
        
        firing_rates[i] = firing_rate
    
    return firing_rates, place_field_centers


def generate_3d_manifold_data(n_neurons, duration=600, sampling_rate=20.0,
                             field_sigma=0.1, step_size=0.02, momentum=0.8,
                             baseline_rate=0.1, peak_rate=2.0,
                             noise_std=0.05, decay_time=2.0, 
                             calcium_noise_std=0.1,
                             grid_arrangement=True,
                             n_environments=1,
                             seed=None, verbose=True):
    """
    Generate synthetic data with neurons on 3D spatial manifold (3D place cells).
    
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
        Step size for movement.
    momentum : float
        Momentum for smoother trajectories.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate at place field center.
    noise_std : float
        Noise in firing rates.
    decay_time : float
        Calcium decay time constant.
    calcium_noise_std : float
        Noise in calcium signal.
    grid_arrangement : bool
        If True, arrange fields on 3D grid.
    n_environments : int
        Number of different 3D environments.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
        
    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    positions : ndarray or list
        3D positions. If n_environments=1, shape (n_timepoints, 3).
        Otherwise list of arrays.
    place_field_centers : ndarray or list
        Place field centers. If n_environments=1, shape (n_neurons, 3).
        Otherwise list of arrays.
    firing_rates : ndarray
        Underlying firing rates (n_neurons x n_timepoints).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_timepoints = int(duration * sampling_rate)
    
    if verbose:
        print(f'Generating 3D spatial manifold data: {n_neurons} neurons, {duration}s')
    
    # Generate data for each environment
    all_positions = []
    all_centers = []
    all_rates = []
    
    for env in range(n_environments):
        if verbose and n_environments > 1:
            print(f'  Environment {env+1}/{n_environments}')
        
        # Generate 3D trajectory
        if verbose:
            print('  Generating 3D trajectory...')
        
        timepoints_per_env = n_timepoints // n_environments
        positions = generate_3d_random_walk(
            timepoints_per_env, 
            bounds=(0, 1),
            step_size=step_size,
            momentum=momentum,
            seed=(seed + env) if seed else None
        )
        
        # Generate place cells
        if verbose:
            print('  Generating 3D place cell responses...')
        
        # For multiple environments, randomly remap some place fields
        if env == 0 or n_environments == 1:
            firing_rates, centers = generate_3d_manifold_neurons(
                n_neurons, positions, field_sigma,
                baseline_rate, peak_rate, noise_std,
                grid_arrangement,
                seed=(seed + env + 100) if seed else None
            )
        else:
            # Partial remapping: ~50% of cells change their place fields
            remap_mask = np.random.random(n_neurons) < 0.5
            centers = all_centers[0].copy()
            
            # Randomly move remapped centers
            centers[remap_mask] = np.random.uniform(0.05, 0.95, (np.sum(remap_mask), 3))
            
            # Generate firing rates with new centers
            firing_rates = np.zeros((n_neurons, timepoints_per_env))
            for i in range(n_neurons):
                place_response = gaussian_place_field_3d(positions, centers[i], field_sigma)
                firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response
                noise = np.random.normal(0, noise_std, timepoints_per_env)
                firing_rates[i] = np.maximum(0, firing_rate + noise)
        
        all_positions.append(positions)
        all_centers.append(centers)
        all_rates.append(firing_rates)
    
    # Concatenate all environments
    positions = np.vstack(all_positions) if n_environments > 1 else all_positions[0]
    firing_rates = np.hstack(all_rates) if n_environments > 1 else all_rates[0]
    
    # Convert to calcium signals
    if verbose:
        print('  Converting to calcium signals...')
    
    calcium_signals = np.zeros((n_neurons, n_timepoints))
    
    for i in range(n_neurons):
        # Generate events from firing rates
        prob_spike = firing_rates[i] / sampling_rate
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
        calcium_signals[i] = calcium_signal
    
    if verbose:
        print('  Done!')
    
    # Return appropriate format
    if n_environments == 1:
        return calcium_signals, positions, all_centers[0], firing_rates
    else:
        return calcium_signals, all_positions, all_centers, firing_rates


def generate_3d_manifold_exp(n_neurons=125, duration=600, fps=20.0,
                            field_sigma=0.1, step_size=0.02, momentum=0.8,
                            baseline_rate=0.1, peak_rate=2.0,
                            noise_std=0.05, decay_time=2.0,
                            calcium_noise_std=0.1,
                            grid_arrangement=True,
                            n_environments=1,
                            add_head_direction=False,
                            seed=None, verbose=True):
    """
    Generate synthetic experiment with 3D place cells.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons (default 125 = 5x5x5 grid).
    duration : float
        Duration in seconds.
    fps : float
        Sampling rate.
    field_sigma : float
        Place field width.
    step_size : float
        Movement step size.
    momentum : float
        Movement momentum.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate.
    noise_std : float
        Firing rate noise.
    decay_time : float
        Calcium decay time.
    calcium_noise_std : float
        Calcium signal noise.
    grid_arrangement : bool
        Arrange fields on 3D grid.
    n_environments : int
        Number of environments.
    add_head_direction : bool
        If True, add 3D head direction (azimuth and elevation).
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
        
    Returns
    -------
    exp : Experiment
        Experiment object.
    info : dict
        Information about the generated data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate 3D manifold data
    calcium_signals, positions, centers, firing_rates = generate_3d_manifold_data(
        n_neurons, duration, fps,
        field_sigma, step_size, momentum,
        baseline_rate, peak_rate,
        noise_std, decay_time, calcium_noise_std,
        grid_arrangement, n_environments,
        seed, verbose
    )
    
    # Create dynamic features
    dynamic_features = {}
    
    if n_environments == 1:
        # Single environment
        dynamic_features['x_position'] = TimeSeries(positions[:, 0], discrete=False)
        dynamic_features['y_position'] = TimeSeries(positions[:, 1], discrete=False)
        dynamic_features['z_position'] = TimeSeries(positions[:, 2], discrete=False)
        
        # Also add as multifeature for manifold analysis
        dynamic_features['position_3d'] = MultiTimeSeries([
            TimeSeries(positions[:, 0], discrete=False),
            TimeSeries(positions[:, 1], discrete=False),
            TimeSeries(positions[:, 2], discrete=False)
        ])
    else:
        # Multiple environments - concatenate all
        all_x = np.concatenate([pos[:, 0] for pos in positions])
        all_y = np.concatenate([pos[:, 1] for pos in positions])
        all_z = np.concatenate([pos[:, 2] for pos in positions])
        
        dynamic_features['x_position'] = TimeSeries(all_x, discrete=False)
        dynamic_features['y_position'] = TimeSeries(all_y, discrete=False)
        dynamic_features['z_position'] = TimeSeries(all_z, discrete=False)
        
        dynamic_features['position_3d'] = MultiTimeSeries([
            TimeSeries(all_x, discrete=False),
            TimeSeries(all_y, discrete=False),
            TimeSeries(all_z, discrete=False)
        ])
        
        # Add environment indicator
        env_indicator = np.concatenate([
            np.full(len(pos), env_idx) 
            for env_idx, pos in enumerate(positions)
        ])
        dynamic_features['environment'] = TimeSeries(env_indicator, discrete=True)
    
    # Optionally add 3D head direction
    if add_head_direction:
        if verbose:
            print('  Adding 3D head direction features...')
        
        # Calculate head direction from trajectory
        if n_environments == 1:
            velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        else:
            all_positions = np.vstack(positions)
            velocities = np.diff(all_positions, axis=0, prepend=all_positions[0:1])
        
        # Azimuth (horizontal angle)
        azimuth = np.arctan2(velocities[:, 1], velocities[:, 0])
        azimuth = (azimuth + 2 * np.pi) % (2 * np.pi)
        
        # Elevation (vertical angle)
        horizontal_dist = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        elevation = np.arctan2(velocities[:, 2], horizontal_dist)
        elevation = np.clip(elevation, -np.pi/2, np.pi/2)  # -90 to +90 degrees
        
        dynamic_features['azimuth'] = TimeSeries(azimuth, discrete=False)
        dynamic_features['elevation'] = TimeSeries(elevation, discrete=False)
        
        # Add circular representations
        dynamic_features['azimuth_circular'] = MultiTimeSeries([
            TimeSeries(np.cos(azimuth), discrete=False),
            TimeSeries(np.sin(azimuth), discrete=False)
        ])
        
        # For elevation, use both sin/cos since it's not fully circular
        dynamic_features['elevation_circular'] = MultiTimeSeries([
            TimeSeries(np.cos(elevation), discrete=False),
            TimeSeries(np.sin(elevation), discrete=False)
        ])
    
    # Create static features
    static_features = {
        'fps': fps,
        't_rise_sec': 0.5,
        't_off_sec': decay_time
    }
    
    # Create experiment
    exp = Experiment(
        'SpatialManifold3D',
        calcium_signals,
        None,
        {},
        static_features,
        dynamic_features,
        reconstruct_spikes=None
    )
    
    # Prepare info
    info = {
        'positions': positions,
        'place_field_centers': centers,
        'firing_rates': firing_rates,
        'field_sigma': field_sigma,
        'manifold_type': '3d_spatial',
        'n_neurons': n_neurons,
        'n_environments': n_environments
    }
    
    return exp, info


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
        # Use amplitude_range to modulate event amplitudes instead of using binary values
        if len(event_times) > 0:
            event_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], len(event_times))
        else:
            event_amplitudes = np.array([])

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
    
    # Handle edge case of 0 neurons
    if nneurons == 0:
        return np.array([]), np.array([]).reshape(0, length), gt

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
                                                  seed=42, fps=20, verbose=True,
                                                  name_convention='str',
                                                  rate_0=0.1, rate_1=1.0, skip_prob=0.1,
                                                  ampl_range=(0.5, 2), decay_time=2, noise_std=0.1):
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
    
    # Create multifeatures from existing continuous features
    multifeatures_to_create = []
    if n_multifeatures > 0 and n_continuous_feats >= 2:
        if verbose:
            print(f'Creating {n_multifeatures} multifeatures...')
        
        # Get all continuous features
        continuous_feats = [f for f in features_dict.keys() if 'c_feat' in f]
        
        # Create multifeatures by pairing continuous features
        multi_idx = 0
        for i in range(0, min(n_multifeatures * 2, len(continuous_feats)), 2):
            if multi_idx >= n_multifeatures:
                break
            if i + 1 < len(continuous_feats):
                feat1 = continuous_feats[i]
                feat2 = continuous_feats[i + 1]
                
                if name_convention == 'str':
                    # String key for the multifeature
                    mf_name = f'multi{multi_idx}'
                    multifeatures_to_create.append((mf_name, (feat1, feat2)))
                else:  # 'tuple' convention (deprecated)
                    # Tuple key for the multifeature
                    # TODO: this need fixing
                    multifeatures_to_create.append(((feat1, feat2), (feat1, feat2)))
                
                multi_idx += 1
    
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
        rate_0=rate_0, rate_1=rate_1, skip_prob=skip_prob,
        ampl_range=ampl_range, decay_time=decay_time, noise_std=noise_std,
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
    for mf_key, mf_components in multifeatures_to_create:
        # Get component TimeSeries
        component_ts = []
        for component_name in mf_components:
            if component_name in dynamic_features and not dynamic_features[component_name].discrete:
                component_ts.append(dynamic_features[component_name])
        
        # Create MultiTimeSeries if all components are continuous
        if len(component_ts) == len(mf_components):
            dynamic_features[mf_key] = aggregate_multiple_ts(*component_ts)
    
    # Create experiment
    exp = Experiment('SyntheticMixedSelectivity',
                     calcium_signals,
                     None,
                     {},
                     {'fps': fps},
                     dynamic_features,
                     reconstruct_spikes=None)
    
    # Prepare selectivity info
    # Create multifeature map for return value
    multifeature_map = {}
    for i, (mf_key, mf_components) in enumerate(multifeatures_to_create):
        if isinstance(mf_key, str):
            # For string convention: components tuple -> multifeature name
            multifeature_map[mf_components] = mf_key
        else:
            # For tuple convention: components tuple -> generated name
            multifeature_map[mf_key] = f'multifeature_{i}'
    
    selectivity_info = {
        'matrix': selectivity_matrix,
        'feature_names': all_feature_names,
        'multifeature_map': multifeature_map
    }
    
    return exp, selectivity_info


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


def generate_circular_manifold_exp(n_neurons=100, duration=600, fps=20.0,
                                  kappa=4.0, step_std=0.1,
                                  baseline_rate=0.1, peak_rate=2.0,
                                  noise_std=0.05, decay_time=2.0,
                                  calcium_noise_std=0.1,
                                  add_mixed_features=False,
                                  n_extra_features=0,
                                  seed=None, verbose=True):
    """
    Generate synthetic experiment with head direction cells on circular manifold.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons.
    duration : float
        Duration in seconds.
    fps : float
        Sampling rate (frames per second).
    kappa : float
        Von Mises concentration (tuning width). Higher = narrower tuning.
        Typical values: 2-8.
    step_std : float
        Standard deviation of head direction random walk.
    baseline_rate : float
        Baseline firing rate.
    peak_rate : float
        Peak firing rate at preferred direction.
    noise_std : float
        Noise in firing rates.
    decay_time : float
        Calcium decay time constant.
    calcium_noise_std : float
        Noise in calcium signal.
    add_mixed_features : bool
        If True, add neurons with mixed selectivity to head direction
        and other features.
    n_extra_features : int
        Number of additional features to generate if add_mixed_features=True.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
        
    Returns
    -------
    exp : Experiment
        Experiment object with circular manifold data.
    info : dict
        Dictionary containing:
        - 'head_direction': Head direction trajectory
        - 'preferred_directions': Preferred directions of neurons
        - 'firing_rates': Underlying firing rates
        - 'kappa': Von Mises concentration used
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate circular manifold data
    calcium_signals, head_direction, preferred_directions, firing_rates = \
        generate_circular_manifold_data(
            n_neurons, duration, fps,
            kappa, step_std,
            baseline_rate, peak_rate,
            noise_std, decay_time, calcium_noise_std,
            seed, verbose
        )
    
    # Create dynamic features
    dynamic_features = {
        'head_direction': TimeSeries(head_direction, discrete=False)
    }
    
    # Add circular representation as multifeature (cos, sin)
    # This allows INTENSE to properly detect circular selectivity
    cos_head = np.cos(head_direction)
    sin_head = np.sin(head_direction)
    circular_components = [
        TimeSeries(cos_head, discrete=False),
        TimeSeries(sin_head, discrete=False)
    ]
    dynamic_features['circular_angle'] = MultiTimeSeries(circular_components)
    
    # Optionally add extra features and mixed selectivity
    if add_mixed_features and n_extra_features > 0:
        if verbose:
            print(f'Adding {n_extra_features} extra features...')
        
        n_timepoints = int(duration * fps)
        
        # Add continuous features
        for i in range(n_extra_features):
            fbm_series = generate_fbm_time_series(n_timepoints, hurst=0.3, 
                                                seed=(seed + 100 + i) if seed else None)
            dynamic_features[f'c_feat_{i}'] = TimeSeries(fbm_series, discrete=False)
        
        # TODO: Add mixed selectivity by modulating some neurons with extra features
        # This would involve modifying the calcium signals based on the extra features
        # For now, we keep pure head direction selectivity
    
    # Create static features
    static_features = {
        'fps': fps,
        't_rise_sec': 0.5,  # Standard GCaMP rise time
        't_off_sec': decay_time
    }
    
    # Create experiment
    exp = Experiment(
        'CircularManifold',
        calcium_signals,
        None,  # No spike data
        {},    # No identificators
        static_features,
        dynamic_features,
        reconstruct_spikes=None
    )
    
    # Prepare info dictionary
    info = {
        'head_direction': head_direction,
        'preferred_directions': preferred_directions,
        'firing_rates': firing_rates,
        'kappa': kappa,
        'manifold_type': 'circular',
        'n_neurons': n_neurons
    }
    
    return exp, info


def generate_mixed_population_exp(n_neurons=100, manifold_fraction=0.6,
                                  manifold_type='2d_spatial', manifold_params=None,
                                  n_discrete_features=3, n_continuous_features=3,
                                  feature_params=None, correlation_mode='independent',
                                  correlation_strength=0.3, duration=600, fps=20.0,
                                  seed=None, verbose=True):
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
        
    Returns
    -------
    exp : Experiment
        Experiment object with mixed population.
    info : dict
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
    
    # Calculate population allocation
    n_manifold = int(n_neurons * manifold_fraction)
    n_feature_selective = n_neurons - n_manifold
    
    if verbose:
        print(f'Generating mixed population: {n_neurons} total neurons')
        print(f'  Manifold cells ({manifold_type}): {n_manifold}')
        print(f'  Feature-selective cells: {n_feature_selective}')
        print(f'  Correlation mode: {correlation_mode}')
    
    # Set default parameters
    if manifold_params is None:
        manifold_params = {
            'field_sigma': 0.1,
            'baseline_rate': 0.1,
            'peak_rate': 2.0,
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
                    n_environments=1,
                    seed=manifold_seed,
                    verbose=verbose
                )
            
            # Add spatial features
            dynamic_features['x_position'] = TimeSeries(positions[:, 0], discrete=False)
            dynamic_features['y_position'] = TimeSeries(positions[:, 1], discrete=False)
            dynamic_features['position_2d'] = MultiTimeSeries([
                TimeSeries(positions[:, 0], discrete=False),
                TimeSeries(positions[:, 1], discrete=False)
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
                    n_environments=1,
                    seed=manifold_seed,
                    verbose=verbose
                )
            
            # Add 3D spatial features
            dynamic_features['x_position'] = TimeSeries(positions[:, 0], discrete=False)
            dynamic_features['y_position'] = TimeSeries(positions[:, 1], discrete=False)
            dynamic_features['z_position'] = TimeSeries(positions[:, 2], discrete=False)
            dynamic_features['position_3d'] = MultiTimeSeries([
                TimeSeries(positions[:, 0], discrete=False),
                TimeSeries(positions[:, 1], discrete=False),
                TimeSeries(positions[:, 2], discrete=False)
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
                    spatial_signal = np.mean(spatial_data, axis=1)  # Average position
                
                # Normalize spatial signal
                spatial_signal = (spatial_signal - np.mean(spatial_signal)) / np.std(spatial_signal)
                
                # Apply correlation
                correlated_feat = (1 - correlation_strength) * feat_data + \
                                  correlation_strength * spatial_signal * np.std(feat_data)
                
                behavioral_features_data[feat_name] = correlated_feat
                dynamic_features[feat_name] = TimeSeries(correlated_feat, discrete=False)
    
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
                        verbose=False
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
                        verbose=False
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
    
    return exp, info
