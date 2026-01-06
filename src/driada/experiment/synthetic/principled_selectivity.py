"""
Principled selectivity synthetic data generation.

This module provides functions to generate synthetic neural data with biologically
meaningful tuning curves for testing INTENSE analysis. Unlike ROI-based discretization,
this approach uses proper tuning curves (von Mises, Gaussian, sigmoid) to create
realistic feature-selective responses.

Key features:
- Head direction cells with von Mises tuning
- Place cells with Gaussian place fields
- Speed cells with sigmoid tuning
- Event cells with binary response
- Mixed selectivity neurons with OR/AND combination modes
- Ground truth structure for validation

Example
-------
>>> population = [
...     {"name": "hd_cells", "count": 4, "features": ["head_direction"]},
...     {"name": "place_cells", "count": 4, "features": ["x", "y"], "combination": "and"},
...     {"name": "speed_cells", "count": 4, "features": ["speed"]},
...     {"name": "event_cells", "count": 4, "features": ["event_0"]},
...     {"name": "mixed", "count": 4, "features": ["head_direction", "event_0"]},
...     {"name": "nonselective", "count": 4, "features": []},
... ]
>>> exp, ground_truth = generate_tuned_selectivity_exp(population)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d

from driada.experiment.exp_base import Experiment
from driada.information.info_base import MultiTimeSeries, TimeSeries

from .core import generate_pseudo_calcium_signal
from .manifold_circular import von_mises_tuning_curve
from .manifold_spatial_2d import generate_2d_random_walk
from .time_series import generate_binary_time_series

# Default tuning parameters for different feature types
TUNING_DEFAULTS = {
    "head_direction": {"kappa": 2.0},  # von Mises concentration
    "x": {"sigma": 0.25},  # Gaussian width
    "y": {"sigma": 0.25},  # Gaussian width
    "speed": {"slope": 12.0},  # Sigmoid slope
}


def sigmoid_tuning_curve(
    x: np.ndarray,
    threshold: float,
    slope: float,
    max_response: float = 1.0,
) -> np.ndarray:
    """
    Sigmoid tuning curve for monotonic response to linear features.

    Used for speed cells that increase firing rate with running speed.

    Parameters
    ----------
    x : array-like
        Input values (e.g., running speed normalized to [0, 1]).
    threshold : float
        Value at which response is 50% of maximum.
    slope : float
        Steepness of the sigmoid. Higher = sharper transition.
    max_response : float, optional
        Maximum response value. Default: 1.0.

    Returns
    -------
    response : ndarray
        Response values in [0, max_response].

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 100)
    >>> response = sigmoid_tuning_curve(x, threshold=0.5, slope=10.0)
    >>> response[50]  # At threshold, response is 50%
    0.5
    """
    x = np.asarray(x)
    return max_response / (1 + np.exp(-slope * (x - threshold)))


def compute_speed_from_positions(
    positions: np.ndarray,
    fps: float,
    smooth_sigma: float = 3,
) -> np.ndarray:
    """
    Compute instantaneous speed from 2D position trajectory.

    Parameters
    ----------
    positions : ndarray
        Shape (2, n_timepoints) with x, y coordinates.
    fps : float
        Sampling rate in Hz.
    smooth_sigma : float, optional
        Gaussian smoothing sigma in frames. Default: 3.

    Returns
    -------
    speed : ndarray
        Instantaneous speed, smoothed. Shape (n_timepoints,).

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.random.randn(2, 100) * 0.1
    >>> positions = np.cumsum(positions, axis=1)
    >>> speed = compute_speed_from_positions(positions, fps=20)
    >>> speed.shape
    (100,)
    """
    dx = np.diff(positions[0, :])
    dy = np.diff(positions[1, :])
    speed = np.sqrt(dx**2 + dy**2) * fps
    # Pad to match original length
    speed = np.concatenate([[speed[0]], speed])
    # Smooth for more realistic speed signal
    speed = gaussian_filter1d(speed, sigma=smooth_sigma)
    return speed


def compute_head_direction_from_positions(
    positions: np.ndarray,
    smooth_sigma: float = 5,
) -> np.ndarray:
    """
    Compute head direction from 2D position trajectory.

    Assumes animal faces in direction of movement.

    Parameters
    ----------
    positions : ndarray
        Shape (2, n_timepoints) with x, y coordinates.
    smooth_sigma : float, optional
        Gaussian smoothing sigma in frames. Default: 5.

    Returns
    -------
    head_direction : ndarray
        Head direction in radians [0, 2*pi). Shape (n_timepoints,).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> positions = np.random.randn(2, 100) * 0.1
    >>> positions = np.cumsum(positions, axis=1)
    >>> hd = compute_head_direction_from_positions(positions)
    >>> hd.shape
    (100,)
    >>> 0 <= hd.min() and hd.max() < 2 * np.pi
    True
    """
    # Smooth positions first for cleaner velocity estimate
    x_smooth = gaussian_filter1d(positions[0, :], sigma=smooth_sigma)
    y_smooth = gaussian_filter1d(positions[1, :], sigma=smooth_sigma)

    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    # Compute angle from velocity
    angles = np.arctan2(dy, dx)
    # Pad to match original length
    angles = np.concatenate([[angles[0]], angles])
    # Convert to [0, 2*pi)
    angles = angles % (2 * np.pi)
    # Additional smoothing using circular mean (via sin/cos)
    sin_smooth = gaussian_filter1d(np.sin(angles), sigma=smooth_sigma)
    cos_smooth = gaussian_filter1d(np.cos(angles), sigma=smooth_sigma)
    head_direction = np.arctan2(sin_smooth, cos_smooth) % (2 * np.pi)
    return head_direction


def combine_responses(
    responses: List[np.ndarray],
    mode: str = "or",
) -> np.ndarray:
    """
    Combine multiple response traces using OR or AND logic.

    Parameters
    ----------
    responses : list of ndarray
        List of response traces, each in range [0, 1].
    mode : str, optional
        Combination mode: "or" (max) or "and" (min). Default: "or".

    Returns
    -------
    combined : ndarray
        Combined response trace.

    Examples
    --------
    >>> import numpy as np
    >>> r1 = np.array([0.2, 0.8, 0.1])
    >>> r2 = np.array([0.5, 0.3, 0.9])
    >>> combine_responses([r1, r2], mode="or")
    array([0.5, 0.8, 0.9])
    >>> combine_responses([r1, r2], mode="and")
    array([0.2, 0.3, 0.1])
    """
    if len(responses) == 0:
        raise ValueError("responses list cannot be empty")
    if len(responses) == 1:
        return responses[0]

    if mode == "or":
        return np.maximum.reduce(responses)
    elif mode == "and":
        return np.minimum.reduce(responses)
    else:
        raise ValueError(f"Unknown combination mode: {mode}. Use 'or' or 'and'.")


def _get_tuning_param(
    feature_name: str,
    param_name: str,
    user_params: Optional[Dict] = None,
    tuning_defaults: Optional[Dict] = None,
) -> float:
    """Get tuning parameter with fallback to defaults."""
    # Check user params first
    if user_params and param_name in user_params:
        return user_params[param_name]

    # Check custom defaults
    if tuning_defaults and feature_name in tuning_defaults:
        if param_name in tuning_defaults[feature_name]:
            return tuning_defaults[feature_name][param_name]

    # Fall back to module defaults
    if feature_name in TUNING_DEFAULTS:
        if param_name in TUNING_DEFAULTS[feature_name]:
            return TUNING_DEFAULTS[feature_name][param_name]

    raise ValueError(f"No default for {param_name} in feature {feature_name}")


def _generate_random_tuning_param(
    feature_name: str,
    param_name: str,
    rng: np.random.Generator,
) -> float:
    """Generate randomized per-neuron tuning parameter."""
    if feature_name == "head_direction" and param_name == "pref_dir":
        return rng.uniform(0, 2 * np.pi)
    elif feature_name in ["x", "y"] and param_name == "center":
        return rng.uniform(0.15, 0.85)
    elif feature_name == "speed" and param_name == "threshold":
        return rng.uniform(0.3, 0.7)
    else:
        raise ValueError(f"Unknown random param: {feature_name}.{param_name}")


def generate_tuned_selectivity_exp(
    population: List[Dict],
    tuning_defaults: Optional[Dict] = None,
    duration: float = 600,
    fps: float = 20,
    baseline_rate: float = 0.05,
    peak_rate: float = 1.2,
    decay_time: float = 2.0,
    calcium_noise: float = 0.08,
    n_discrete_features: int = 2,
    event_active_fraction: float = 0.08,
    event_avg_duration: float = 0.8,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Experiment, Dict]:
    """
    Generate synthetic experiment with principled tuning-based selectivity.

    Creates neurons with biologically meaningful tuning to various feature types.
    Each neuron group can respond to one or more features with configurable
    combination modes (OR/AND).

    Parameters
    ----------
    population : list of dict
        Population configuration. Each dict specifies a neuron group with keys:
        - "name" : str - Group name (e.g., "hd_cells", "place_cells")
        - "count" : int - Number of neurons in this group
        - "features" : list of str - Feature names this group responds to.
          Supported: "head_direction", "x", "y", "speed", "event_0", "event_1", etc.
        - "combination" : str, optional - How to combine multiple features:
          "or" (default) or "and"
        - "tuning_params" : dict, optional - Override default tuning parameters
    tuning_defaults : dict, optional
        Override module-level tuning defaults. Keys are feature names,
        values are dicts with parameter names and values.
    duration : float, optional
        Recording duration in seconds. Default: 600.
    fps : float, optional
        Sampling rate in Hz. Default: 20.
    baseline_rate : float, optional
        Baseline firing rate (spikes/frame). Default: 0.05.
    peak_rate : float, optional
        Peak firing rate during selectivity. Default: 1.2.
    decay_time : float, optional
        Calcium decay time constant in seconds. Default: 2.0.
    calcium_noise : float, optional
        Calcium signal noise standard deviation. Default: 0.08.
    n_discrete_features : int, optional
        Number of discrete event features to generate. Default: 2.
    event_active_fraction : float, optional
        Fraction of time each event is active. Default: 0.08.
    event_avg_duration : float, optional
        Average event duration in seconds. Default: 0.8.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default: True.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with neural signals and features.
    ground_truth : dict
        Ground truth information containing:
        - "expected_pairs" : list of (neuron_idx, feature_name) tuples
        - "neuron_types" : dict mapping neuron_idx to group name
        - "tuning_parameters" : dict with detailed tuning params per neuron
        - "population_config" : reference to input population config

    Examples
    --------
    >>> # Minimal example with default parameters
    >>> population = [
    ...     {"name": "hd_cells", "count": 4, "features": ["head_direction"]},
    ...     {"name": "nonselective", "count": 2, "features": []},
    ... ]
    >>> exp, gt = generate_tuned_selectivity_exp(population, duration=60, verbose=False)
    >>> exp.n_cells
    6
    >>> len(gt["expected_pairs"])
    4

    >>> # Advanced example with custom parameters
    >>> population = [
    ...     {"name": "narrow_hd", "count": 2, "features": ["head_direction"],
    ...      "tuning_params": {"kappa": 4.0}},  # Narrower tuning
    ...     {"name": "conjunctive", "count": 2, "features": ["x", "y"],
    ...      "combination": "and"},  # AND combination
    ... ]
    >>> exp, gt = generate_tuned_selectivity_exp(
    ...     population, tuning_defaults={"x": {"sigma": 0.3}}, verbose=False
    ... )
    """
    rng = np.random.default_rng(seed)

    n_frames = int(duration * fps)

    # Calculate total neurons
    n_neurons = sum(group["count"] for group in population)

    if verbose:
        print(f"Generating {n_neurons} neurons with principled selectivity...")

    # -------------------------------------------------------------------------
    # Generate behavioral features
    # -------------------------------------------------------------------------
    if verbose:
        print("Generating behavioral trajectory...")

    # 2D random walk for position
    positions = generate_2d_random_walk(
        n_frames,
        bounds=(0, 1),
        step_size=0.02,
        momentum=0.8,
        seed=seed,
    )

    # Derive head direction from movement
    head_direction = compute_head_direction_from_positions(positions)

    # Derive speed from movement
    speed = compute_speed_from_positions(positions, fps)
    # Normalize speed to [0, 1] for easier threshold setting
    speed_normalized = (speed - speed.min()) / (speed.max() - speed.min() + 1e-10)

    # Generate discrete event features
    discrete_features = {}
    for i in range(n_discrete_features):
        avg_islands = int(
            n_frames * event_active_fraction / (event_avg_duration * fps)
        )
        avg_duration_frames = int(event_avg_duration * fps)
        event_seed = seed + i + 100 if seed is not None else None
        binary_ts = generate_binary_time_series(
            n_frames, avg_islands, avg_duration_frames, seed=event_seed
        )
        discrete_features[f"event_{i}"] = binary_ts

    # Collect all available features
    available_features = {
        "head_direction": head_direction,
        "x": positions[0, :],
        "y": positions[1, :],
        "speed": speed_normalized,
    }
    available_features.update(discrete_features)

    if verbose:
        print(f"  Position range: x=[{positions[0].min():.2f}, {positions[0].max():.2f}], "
              f"y=[{positions[1].min():.2f}, {positions[1].max():.2f}]")
        print(f"  Speed range: [{speed.min():.3f}, {speed.max():.3f}]")
        print(f"  Discrete features: {list(discrete_features.keys())}")

    # -------------------------------------------------------------------------
    # Generate neural responses with ground truth
    # -------------------------------------------------------------------------
    if verbose:
        print("Generating neural responses...")

    firing_rates = np.zeros((n_neurons, n_frames))
    ground_truth = {
        "expected_pairs": [],
        "neuron_types": {},
        "tuning_parameters": {},
        "population_config": population,
    }

    neuron_idx = 0
    for group in population:
        group_name = group["name"]
        group_count = group["count"]
        group_features = group.get("features", [])
        combination_mode = group.get("combination", "or")
        user_tuning_params = group.get("tuning_params", {})

        if verbose:
            print(f"  {group_name}: neurons {neuron_idx}-{neuron_idx + group_count - 1}, "
                  f"features={group_features}, mode={combination_mode}")

        for i in range(group_count):
            current_idx = neuron_idx + i
            ground_truth["neuron_types"][current_idx] = group_name
            ground_truth["tuning_parameters"][current_idx] = {}

            if not group_features:
                # Non-selective neurons
                firing_rates[current_idx] = baseline_rate + rng.normal(0, 0.02, n_frames)
                firing_rates[current_idx] = np.maximum(0, firing_rates[current_idx])
                continue

            # Generate responses for each feature
            responses = []
            for feat_name in group_features:
                if feat_name not in available_features:
                    raise ValueError(f"Unknown feature: {feat_name}")

                feat_data = available_features[feat_name]
                neuron_tuning = {}

                if feat_name == "head_direction":
                    pref_dir = _generate_random_tuning_param("head_direction", "pref_dir", rng)
                    kappa = _get_tuning_param("head_direction", "kappa",
                                               user_tuning_params, tuning_defaults)
                    response = von_mises_tuning_curve(feat_data, pref_dir, kappa)
                    neuron_tuning = {"pref_dir": pref_dir, "kappa": kappa}

                elif feat_name in ["x", "y"]:
                    center = _generate_random_tuning_param(feat_name, "center", rng)
                    sigma = _get_tuning_param(feat_name, "sigma",
                                               user_tuning_params, tuning_defaults)
                    # Use 1D Gaussian for marginal response
                    response = np.exp(-0.5 * ((feat_data - center) / sigma) ** 2)
                    neuron_tuning = {"center": center, "sigma": sigma}

                elif feat_name == "speed":
                    threshold = _generate_random_tuning_param("speed", "threshold", rng)
                    slope = _get_tuning_param("speed", "slope",
                                               user_tuning_params, tuning_defaults)
                    response = sigmoid_tuning_curve(feat_data, threshold, slope)
                    neuron_tuning = {"threshold": threshold, "slope": slope}

                elif feat_name.startswith("event_"):
                    # Binary feature - response is the event itself
                    response = feat_data.astype(float)
                    neuron_tuning = {"binary": True}

                else:
                    raise ValueError(f"Unsupported feature type: {feat_name}")

                responses.append(response)
                ground_truth["tuning_parameters"][current_idx][feat_name] = neuron_tuning

            # Build expected pairs - use position_2d for place cells (x+y) instead of marginals
            has_x = "x" in group_features
            has_y = "y" in group_features
            if has_x and has_y:
                # Place cell: expect position_2d detection, not x/y marginals
                ground_truth["expected_pairs"].append((current_idx, "position_2d"))
            else:
                # Add individual features (excluding x/y which need both for place field)
                for feat_name in group_features:
                    if feat_name not in ["x", "y"]:
                        ground_truth["expected_pairs"].append((current_idx, feat_name))

            # Combine responses
            combined_response = combine_responses(responses, mode=combination_mode)
            firing_rates[current_idx] = baseline_rate + (peak_rate - baseline_rate) * combined_response

        neuron_idx += group_count

    # Add noise to all firing rates
    noise = rng.normal(0, 0.03, firing_rates.shape)
    firing_rates = np.maximum(0, firing_rates + noise)

    # -------------------------------------------------------------------------
    # Convert firing rates to calcium signals
    # -------------------------------------------------------------------------
    if verbose:
        print("Converting to calcium signals...")

    calcium_signals = np.zeros((n_neurons, n_frames))
    for idx in range(n_neurons):
        # Generate spikes from firing rates
        prob_spike = firing_rates[idx] / fps
        prob_spike = np.clip(prob_spike, 0, 1)
        events = rng.binomial(1, prob_spike)

        # Convert to calcium
        calcium_signals[idx] = generate_pseudo_calcium_signal(
            events=events,
            duration=duration,
            sampling_rate=fps,
            amplitude_range=(0.5, 2.0),
            decay_time=decay_time,
            noise_std=calcium_noise,
        )

    # -------------------------------------------------------------------------
    # Create Experiment object
    # -------------------------------------------------------------------------
    if verbose:
        print("Creating Experiment object...")

    # TimeSeries for continuous features
    hd_ts = TimeSeries(data=head_direction, discrete=False)
    x_ts = TimeSeries(data=positions[0, :], discrete=False)
    y_ts = TimeSeries(data=positions[1, :], discrete=False)
    speed_ts = TimeSeries(data=speed_normalized, discrete=False)

    # MultiTimeSeries for position
    position_mts = MultiTimeSeries(
        [
            TimeSeries(data=positions[0, :], discrete=False),
            TimeSeries(data=positions[1, :], discrete=False),
        ],
        allow_zero_columns=True,
    )

    # Build dynamic features
    dynamic_features = {
        "head_direction": hd_ts,
        "x": x_ts,
        "y": y_ts,
        "speed": speed_ts,
        "position_2d": position_mts,
    }

    # Add discrete features
    for feat_name, feat_data in discrete_features.items():
        dynamic_features[feat_name] = TimeSeries(data=feat_data, discrete=True)

    # Static features
    static_features = {
        "fps": fps,
        "t_rise_sec": 0.04,
        "t_off_sec": min(decay_time, duration / 10),
    }

    # Create experiment
    exp = Experiment(
        signature="tuned_selectivity_exp",
        calcium=calcium_signals,
        spikes=None,
        static_features=static_features,
        dynamic_features=dynamic_features,
        exp_identificators={
            "type": "principled_tuning",
            "n_neurons": n_neurons,
            "duration": duration,
        },
        verbose=False,
    )

    if verbose:
        print(f"  Created experiment: {n_neurons} neurons, {n_frames} frames")
        print(f"  Features: {list(dynamic_features.keys())}")
        print(f"  Expected significant pairs: {len(ground_truth['expected_pairs'])}")

    return exp, ground_truth


def ground_truth_to_selectivity_matrix(
    ground_truth: Dict,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Convert ground truth to selectivity matrix format.

    Converts the principled ground truth format to the matrix format
    used by the old mixed selectivity system, enabling compatibility
    with existing analysis tools.

    Parameters
    ----------
    ground_truth : dict
        Ground truth from generate_tuned_selectivity_exp().
        Must contain 'expected_pairs' key.
    feature_names : list, optional
        Ordered list of feature names for matrix columns.
        If None, extracted from expected_pairs (sorted alphabetically).

    Returns
    -------
    selectivity_info : dict
        Dictionary compatible with old format:
        - 'matrix': ndarray of shape (n_features, n_neurons)
        - 'feature_names': list of feature names

    Examples
    --------
    >>> population = [
    ...     {"name": "hd_cells", "count": 2, "features": ["head_direction"]},
    ...     {"name": "speed_cells", "count": 2, "features": ["speed"]},
    ... ]
    >>> exp, gt = generate_tuned_selectivity_exp(population, duration=30, verbose=False)
    >>> selectivity_info = ground_truth_to_selectivity_matrix(gt)
    >>> selectivity_info['matrix'].shape
    (2, 4)
    >>> selectivity_info['feature_names']
    ['head_direction', 'speed']
    """
    expected_pairs = ground_truth["expected_pairs"]
    neuron_types = ground_truth.get("neuron_types", {})

    # Get number of neurons (consider both expected_pairs and neuron_types
    # to include nonselective neurons that don't appear in expected_pairs)
    n_neurons = 0
    if expected_pairs:
        n_neurons = max(n_neurons, max(pair[0] for pair in expected_pairs) + 1)
    if neuron_types:
        n_neurons = max(n_neurons, max(neuron_types.keys()) + 1)

    # Extract unique features if not provided
    if feature_names is None:
        feature_names = sorted(set(pair[1] for pair in expected_pairs))

    n_features = len(feature_names)
    feat_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Build matrix
    matrix = np.zeros((n_features, n_neurons))
    for neuron_idx, feat_name in expected_pairs:
        if feat_name in feat_to_idx:
            matrix[feat_to_idx[feat_name], neuron_idx] = 1.0

    return {
        "matrix": matrix,
        "feature_names": feature_names,
    }
