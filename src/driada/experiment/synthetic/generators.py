"""
Experiment generators for synthetic neural data.

This module consolidates all experiment generation functions:

CANONICAL GENERATOR:
- generate_tuned_selectivity_exp: Main generator with tuning-based selectivity
- ground_truth_to_selectivity_matrix: Convert ground truth format

MIXED SELECTIVITY:
- generate_multiselectivity_patterns: Random selectivity assignment (Dirichlet)
- generate_synthetic_exp_with_mixed_selectivity: Thin wrapper for random assignment

MANIFOLD-SPECIFIC:
- generate_circular_manifold_neurons/data/exp: Head direction cells
- generate_2d_manifold_neurons/data/exp: Place cells

LEGACY/CONVENIENCE WRAPPERS:
- generate_synthetic_data: Legacy threshold-based generation
- generate_synthetic_exp: Convenience wrapper
- generate_mixed_population_exp: Mixed population convenience wrapper

All generators follow the unified pattern:
1. Generate behavioral trajectory (random walk)
2. Create feature time series
3. Generate neural responses using tuning curves
4. Convert to calcium signals
5. Package into Experiment object with ground truth
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tqdm

from driada.experiment.exp_base import Experiment
from driada.information.info_base import MultiTimeSeries, TimeSeries
from driada.information.circular_transform import circular_to_cos_sin
from driada.utils.data import check_positive, check_nonnegative, check_unit

from .core import DEFAULT_T_RISE, DEFAULT_SYNTHETIC_PARAMS, validate_peak_rate, generate_pseudo_calcium_signal
from .utils import get_effective_decay_time
from .time_series import (
    generate_binary_time_series,
    generate_fbm_time_series,
    generate_circular_random_walk,
    generate_2d_random_walk,
    select_signal_roi,
    delete_one_islands,
    apply_poisson_to_binary_series,
    discretize_via_roi,
)
from .tuning import (
    TUNING_DEFAULTS,
    von_mises_tuning_curve,
    gaussian_place_field,
    sigmoid_tuning_curve,
    threshold_response,
    compute_speed_from_positions,
    compute_head_direction_from_positions,
    combine_responses,
)


# =============================================================================
# Helper Functions (from principled_selectivity)
# =============================================================================


def _extract_synthetic_params(**kwargs) -> dict:
    """
    Extract and merge synthetic data parameters with defaults.

    Consolidates the common pattern of merging user kwargs with
    DEFAULT_SYNTHETIC_PARAMS and extracting individual parameters.

    Parameters
    ----------
    **kwargs : dict
        User-provided parameters to override defaults.

    Returns
    -------
    params : dict
        Merged parameters with all standard synthetic data keys:
        - duration: Recording duration in seconds
        - fps: Sampling rate in Hz
        - baseline_rate: Baseline firing rate
        - peak_rate: Peak firing rate
        - firing_noise: Firing rate noise std
        - decay_time: Calcium decay time constant
        - calcium_noise: Calcium signal noise std
        - amplitude_range: (min, max) calcium event amplitudes
    """
    params = {**DEFAULT_SYNTHETIC_PARAMS, **kwargs}
    return {
        "duration": params["duration"],
        "fps": params["fps"],
        "baseline_rate": params["baseline_rate"],
        "peak_rate": params["peak_rate"],
        "firing_noise": params["firing_noise"],
        "decay_time": params["decay_time"],
        "calcium_noise": params["calcium_noise"],
        "amplitude_range": params["amplitude_range"],
    }


def _firing_rates_to_calcium(
    firing_rates: np.ndarray,
    fps: float,
    duration: float,
    decay_time: float,
    calcium_noise: float,
    amplitude_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Convert firing rates to calcium signals via spike generation.

    This is a common operation used by multiple generators. It converts
    firing rates to binary spikes using a Poisson process, then convolves
    with calcium dynamics.

    Parameters
    ----------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints). Instantaneous firing rates in Hz.
    fps : float
        Sampling rate in Hz.
    duration : float
        Total duration in seconds.
    decay_time : float
        Calcium decay time constant in seconds.
    calcium_noise : float
        Noise standard deviation for calcium signal.
    amplitude_range : tuple of float
        (min, max) amplitude range for calcium events.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    calcium_signals : ndarray
        Shape (n_neurons, n_timepoints). Synthetic calcium signals.
    """
    n_neurons, n_timepoints = firing_rates.shape
    calcium_signals = np.zeros((n_neurons, n_timepoints))

    for idx in range(n_neurons):
        # Generate spikes from firing rates via Poisson process
        prob_spike = firing_rates[idx] / fps
        prob_spike = np.clip(prob_spike, 0, 1)
        events = rng.binomial(1, prob_spike)

        # Generate unique seed for this neuron's calcium signal
        neuron_seed = int(rng.integers(0, 2**31))

        # Convert to calcium signal
        calcium_signals[idx] = generate_pseudo_calcium_signal(
            events=events,
            duration=duration,
            sampling_rate=fps,
            amplitude_range=amplitude_range,
            decay_time=decay_time,
            noise_std=calcium_noise,
            seed=neuron_seed,
        )

    return calcium_signals


def _selectivity_matrix_to_config(
    selectivity_matrix: np.ndarray,
    feature_names: List[str],
    tuning_type: str = "threshold",
    combination_mode: str = "weighted_or",
    feature_name_map: Optional[Dict[str, str]] = None,
    baseline_rate: float = 0.1,
    peak_rate: float = 1.0,
) -> List[Dict]:
    """
    Convert a selectivity matrix to population config format.

    Used for random assignment mode - converts the output of
    generate_multiselectivity_patterns() to population config.

    Parameters
    ----------
    selectivity_matrix : ndarray
        Shape (n_features, n_neurons). Non-zero values indicate selectivity.
    feature_names : list of str
        Feature names corresponding to matrix rows.
    tuning_type : str, optional
        Tuning type for all neurons. Default: "threshold".
    combination_mode : str, optional
        How to combine multiple features. Default: "weighted_or".
    feature_name_map : dict, optional
        Mapping from feature_names to canonical names.
        E.g., {"my_discrete": "event_0", "my_continuous": "fbm_0"}.
    baseline_rate : float, optional
        Baseline rate for threshold tuning. Default: 0.1.
    peak_rate : float, optional
        Peak rate for threshold tuning. Default: 1.0.

    Returns
    -------
    population : list of dict
        Population config where each neuron is its own "group".
    """
    n_features, n_neurons = selectivity_matrix.shape
    population = []

    for neuron_idx in range(n_neurons):
        # Get features this neuron is selective to
        weights = selectivity_matrix[:, neuron_idx]
        selective_indices = [i for i in range(n_features) if weights[i] > 0]
        selective_weights = [weights[i] for i in selective_indices]

        # Map feature names if mapping provided
        if feature_name_map:
            selective_features = [
                feature_name_map.get(feature_names[i], feature_names[i])
                for i in selective_indices
            ]
        else:
            selective_features = [feature_names[i] for i in selective_indices]

        if not selective_features:
            # Non-selective neuron
            population.append({
                "name": f"nonselective_{neuron_idx}",
                "count": 1,
                "features": [],
            })
        else:
            # Selective neuron
            group_config = {
                "name": f"neuron_{neuron_idx}",
                "count": 1,
                "features": selective_features,
                "tuning_type": tuning_type,
                "combination": combination_mode,
            }
            # Add weights if multiple features
            if len(selective_features) > 1:
                group_config["weights"] = list(selective_weights)

            population.append(group_config)

    return population


def _get_tuning_param(
    feature_name: str,
    param_name: str,
    user_params: Optional[Dict] = None,
    tuning_defaults: Optional[Dict] = None,
) -> float:
    """Get tuning parameter with fallback to defaults.

    Looks up a tuning parameter by checking user overrides first, then
    custom defaults, then module-level ``TUNING_DEFAULTS``.

    Parameters
    ----------
    feature_name : str
        Name of the feature (e.g., ``"head_direction"``, ``"speed"``).
    param_name : str
        Name of the tuning parameter to retrieve (e.g., ``"kappa"``).
    user_params : dict, optional
        Per-group user overrides for tuning parameters.
    tuning_defaults : dict, optional
        Custom default overrides keyed by feature name.

    Returns
    -------
    float
        The resolved parameter value.

    Raises
    ------
    ValueError
        If the parameter cannot be found in any source.
    """
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
) -> Union[float, np.ndarray]:
    """Generate randomized per-neuron tuning parameter.

    Produces a random value appropriate for the given feature and parameter
    combination (e.g., a random preferred direction for head-direction cells).

    Parameters
    ----------
    feature_name : str
        Name of the feature (e.g., ``"head_direction"``, ``"x"``,
        ``"position_2d"``, ``"speed"``, ``"fbm_0"``).
    param_name : str
        Name of the tuning parameter to randomize (e.g., ``"pref_dir"``,
        ``"center"``, ``"threshold"``).
    rng : numpy.random.Generator
        NumPy random number generator instance.

    Returns
    -------
    float or numpy.ndarray
        The randomized parameter value. Returns an array for
        multi-dimensional parameters (e.g., 2D place-field center).

    Raises
    ------
    ValueError
        If the feature/parameter combination is not recognized.
    """
    if feature_name == "head_direction" and param_name == "pref_dir":
        return rng.uniform(0, 2 * np.pi)
    elif feature_name in ["x", "y"] and param_name == "center":
        return rng.uniform(0.15, 0.85)
    elif feature_name == "position_2d" and param_name == "center":
        # Return 2D center for true 2D place field
        return np.array([rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)])
    elif feature_name == "speed" and param_name == "threshold":
        return rng.uniform(0.3, 0.7)
    elif feature_name.startswith("fbm_") and param_name == "threshold":
        return rng.uniform(0.3, 0.7)
    else:
        raise ValueError(f"Unknown random param: {feature_name}.{param_name}")


# =============================================================================
# Canonical Generator (from principled_selectivity)
# =============================================================================


def generate_tuned_selectivity_exp(
    population: List[Dict],
    tuning_defaults: Optional[Dict] = None,
    duration: float = 600,
    fps: float = 20,
    baseline_rate: float = 0.05,
    peak_rate: float = 2.0,
    decay_time: float = 2.0,
    calcium_noise: float = 0.02,
    calcium_amplitude_range: Tuple[float, float] = (0.5, 2.0),
    n_discrete_features: int = 2,
    event_active_fraction: float = 0.08,
    event_avg_duration: float = 0.8,
    skip_prob: float = 0.0,
    hurst: float = 0.3,
    seed: Optional[int] = None,
    verbose: bool = True,
    reconstruct_spikes: Optional[str] = None,
) -> "Experiment":
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
          Supported features:
          * "head_direction" - von Mises tuning to heading direction
          * "position_2d" - True 2D Gaussian place field (recommended for place cells)
          * "x", "y" - 1D marginal Gaussian tuning to position axes
          * "speed" - sigmoid tuning to running speed
          * "event_0", "event_1", ... - binary response to discrete events
          * "fbm_0", "fbm_1", ... - sigmoid response to FBM continuous features
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
        Peak firing rate during selectivity. Default: 2.0.
    decay_time : float, optional
        Calcium decay time constant in seconds. Default: 2.0.
    calcium_noise : float, optional
        Calcium signal noise standard deviation. Default: 0.02.
    calcium_amplitude_range : tuple of float, optional
        Range for calcium event amplitudes (min, max). Default: (0.5, 2.0).
    n_discrete_features : int, optional
        Number of discrete event features to generate. Default: 2.
    event_active_fraction : float, optional
        Fraction of time each event is active. Default: 0.08.
    event_avg_duration : float, optional
        Average event duration in seconds. Default: 0.8.
    skip_prob : float, optional
        Probability of skipping an event (for event features). Default: 0.0.
    hurst : float, optional
        Hurst parameter for FBM features. Default: 0.3.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default: True.
    reconstruct_spikes : str, optional
        Spike reconstruction method to apply after generating calcium
        traces. If ``None``, no spike reconstruction is performed.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with neural signals and features.
        Ground truth is accessible via exp.ground_truth containing:
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
    >>> exp = generate_tuned_selectivity_exp(population, duration=60, verbose=False)
    >>> exp.n_cells
    6
    >>> len(exp.ground_truth["expected_pairs"])
    4

    >>> # Advanced example with custom parameters
    >>> population = [
    ...     {"name": "narrow_hd", "count": 2, "features": ["head_direction"],
    ...      "tuning_params": {"kappa": 4.0}},  # Narrower tuning
    ...     {"name": "conjunctive", "count": 2, "features": ["x", "y"],
    ...      "combination": "and"},  # AND combination
    ... ]
    >>> exp = generate_tuned_selectivity_exp(
    ...     population, tuning_defaults={"x": {"sigma": 0.3}}, verbose=False
    ... )
    """
    # Validate peak rate
    validate_peak_rate(peak_rate, context="generate_tuned_selectivity_exp")

    rng = np.random.default_rng(seed)

    n_frames = int(duration * fps)

    # Calculate total neurons
    n_neurons = sum(group["count"] for group in population)

    # Collect all feature names referenced in population config
    required_features = set()
    for group in population:
        for feat in group.get("features", []):
            required_features.add(feat)

    # Determine which behavioral features are needed
    needs_head_direction = "head_direction" in required_features
    needs_position_2d = "position_2d" in required_features
    needs_x = "x" in required_features
    needs_y = "y" in required_features
    needs_speed = "speed" in required_features
    # If any position-related feature is needed, we need the trajectory
    needs_trajectory = needs_head_direction or needs_position_2d or needs_x or needs_y or needs_speed

    if verbose:
        print(f"Generating {n_neurons} neurons with principled selectivity...")

    # -------------------------------------------------------------------------
    # Generate behavioral features (only if needed)
    # -------------------------------------------------------------------------
    positions = None
    head_direction = None
    speed_normalized = None

    if needs_trajectory:
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

    # Collect all required FBM features from population config
    fbm_features = {}
    all_features_needed = set()
    for group in population:
        for feat in group.get("features", []):
            all_features_needed.add(feat)

    for feat_name in all_features_needed:
        if feat_name.startswith("fbm_"):
            # Generate FBM time series for this feature
            fbm_idx = int(feat_name.split("_")[1])
            fbm_seed = seed + fbm_idx + 200 if seed is not None else None
            # Use function parameter hurst as default, allow override via tuning_defaults
            fbm_hurst = tuning_defaults.get("fbm", {}).get("hurst", hurst) if tuning_defaults else hurst
            fbm_ts = generate_fbm_time_series(n_frames, fbm_hurst, seed=fbm_seed)
            # Normalize to [0, 1] for easier threshold setting
            fbm_ts = (fbm_ts - fbm_ts.min()) / (fbm_ts.max() - fbm_ts.min() + 1e-10)
            fbm_features[feat_name] = fbm_ts

    # Collect all available features
    available_features = {}
    if needs_trajectory:
        available_features["head_direction"] = head_direction
        available_features["x"] = positions[0, :]
        available_features["y"] = positions[1, :]
        available_features["speed"] = speed_normalized
        available_features["positions_array"] = positions  # For position_2d handling
    available_features.update(discrete_features)
    available_features.update(fbm_features)

    if verbose:
        if needs_trajectory:
            print(f"  Position range: x=[{positions[0].min():.2f}, {positions[0].max():.2f}], "
                  f"y=[{positions[1].min():.2f}, {positions[1].max():.2f}]")
            print(f"  Speed range: [{speed.min():.3f}, {speed.max():.3f}]")
        print(f"  Discrete features: {list(discrete_features.keys())}")
        if fbm_features:
            print(f"  FBM features: {list(fbm_features.keys())}")

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
        combination_weights = group.get("weights", None)
        user_tuning_params = group.get("tuning_params", {})
        tuning_type = group.get("tuning_type", "default")  # "default" or "threshold"
        discretization_mode = group.get("discretization", "roi")  # For threshold mode

        if verbose:
            tuning_info = f", tuning={tuning_type}" if tuning_type != "default" else ""
            print(f"  {group_name}: neurons {neuron_idx}-{neuron_idx + group_count - 1}, "
                  f"features={group_features}, mode={combination_mode}{tuning_info}")

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
            # Generate seed for threshold discretization
            threshold_seed = seed + current_idx * 10 if seed is not None else None

            for feat_name in group_features:
                # Check if feature is available (position_2d uses positions_array)
                if feat_name != "position_2d" and feat_name not in available_features:
                    raise ValueError(f"Unknown feature: {feat_name}")

                neuron_tuning = {}

                # Handle threshold tuning mode (binary discretization)
                if tuning_type == "threshold":
                    # Get the feature data
                    if feat_name == "position_2d":
                        # For position_2d in threshold mode, use average position
                        pos_array = available_features["positions_array"]
                        feat_data = np.mean(pos_array, axis=0)
                    elif feat_name.startswith("event_"):
                        # Events are already binary
                        feat_data = available_features[feat_name]
                        response = feat_data.astype(float)
                        neuron_tuning = {"binary": True, "tuning_type": "threshold"}
                        responses.append(response)
                        ground_truth["tuning_parameters"][current_idx][feat_name] = neuron_tuning
                        continue
                    else:
                        feat_data = available_features[feat_name]

                    # Apply threshold response (discretization)
                    response = threshold_response(
                        feat_data,
                        discretization=discretization_mode,
                        threshold=user_tuning_params.get("threshold", 0.5),
                        seed=threshold_seed,
                    )
                    neuron_tuning = {
                        "tuning_type": "threshold",
                        "discretization": discretization_mode,
                    }
                    responses.append(response)
                    ground_truth["tuning_parameters"][current_idx][feat_name] = neuron_tuning
                    continue

                # Default tuning curves (original behavior)
                if feat_name == "head_direction":
                    feat_data = available_features[feat_name]
                    pref_dir = _generate_random_tuning_param("head_direction", "pref_dir", rng)
                    kappa = _get_tuning_param("head_direction", "kappa",
                                               user_tuning_params, tuning_defaults)
                    response = von_mises_tuning_curve(feat_data, pref_dir, kappa)
                    neuron_tuning = {"pref_dir": pref_dir, "kappa": kappa}

                elif feat_name == "position_2d":
                    # True 2D Gaussian place field using gaussian_place_field
                    pos_array = available_features["positions_array"]
                    center = _generate_random_tuning_param("position_2d", "center", rng)
                    sigma = _get_tuning_param("position_2d", "sigma",
                                               user_tuning_params, tuning_defaults)
                    response = gaussian_place_field(pos_array, center, sigma)
                    neuron_tuning = {"center": center.tolist(), "sigma": sigma}

                elif feat_name in ["x", "y"]:
                    feat_data = available_features[feat_name]
                    center = _generate_random_tuning_param(feat_name, "center", rng)
                    sigma = _get_tuning_param(feat_name, "sigma",
                                               user_tuning_params, tuning_defaults)
                    # Use 1D Gaussian for marginal response
                    response = np.exp(-0.5 * ((feat_data - center) / sigma) ** 2)
                    neuron_tuning = {"center": center, "sigma": sigma}

                elif feat_name == "speed":
                    feat_data = available_features[feat_name]
                    threshold = _generate_random_tuning_param("speed", "threshold", rng)
                    slope = _get_tuning_param("speed", "slope",
                                               user_tuning_params, tuning_defaults)
                    response = sigmoid_tuning_curve(feat_data, threshold, slope)
                    neuron_tuning = {"threshold": threshold, "slope": slope}

                elif feat_name.startswith("event_"):
                    # Binary feature - response is the event itself
                    feat_data = available_features[feat_name]
                    response = feat_data.astype(float)
                    neuron_tuning = {"binary": True}
                    if skip_prob > 0:
                        skip_seed = seed + current_idx * 100 + 42 if seed is not None else None
                        response = delete_one_islands(
                            response.astype(int), skip_prob, seed=skip_seed
                        ).astype(float)

                elif feat_name.startswith("fbm_"):
                    # FBM feature - ROI-based binary selectivity region
                    # Select a random region in feature value space where this
                    # neuron is responsive (~15% of values). This produces a
                    # binary response identical in structure to discrete events,
                    # so skip_prob operates symmetrically on both feature types.
                    feat_data = available_features[feat_name]
                    roi_seed = seed + current_idx * 100 + 43 if seed is not None else None
                    center, lower_border, upper_border = select_signal_roi(
                        feat_data, seed=roi_seed
                    )
                    response = np.where(
                        (feat_data >= lower_border) & (feat_data <= upper_border),
                        1.0, 0.0,
                    )
                    neuron_tuning = {
                        "center": center,
                        "lower_border": lower_border,
                        "upper_border": upper_border,
                    }
                    if skip_prob > 0:
                        skip_seed = seed + current_idx * 100 + 42 if seed is not None else None
                        response = delete_one_islands(
                            response.astype(int), skip_prob, seed=skip_seed
                        ).astype(float)

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
            combined_response = combine_responses(
                responses, weights=combination_weights, mode=combination_mode
            )
            firing_rates[current_idx] = baseline_rate + (peak_rate - baseline_rate) * combined_response

            # Store combination info in ground truth if using weights
            if combination_weights is not None or combination_mode not in ("or", "and"):
                weight_dict = {}
                if combination_weights is not None:
                    for feat_name, weight in zip(group_features, combination_weights):
                        weight_dict[feat_name] = weight
                ground_truth["tuning_parameters"][current_idx]["_combination"] = {
                    "mode": combination_mode,
                    "weights": weight_dict if weight_dict else None,
                }

        neuron_idx += group_count

    # Add noise to all firing rates
    noise = rng.normal(0, 0.03, firing_rates.shape)
    firing_rates = np.maximum(0, firing_rates + noise)

    # -------------------------------------------------------------------------
    # Convert firing rates to calcium signals
    # -------------------------------------------------------------------------
    if verbose:
        print("Converting to calcium signals...")

    calcium_signals = _firing_rates_to_calcium(
        firing_rates=firing_rates,
        fps=fps,
        duration=duration,
        decay_time=decay_time,
        calcium_noise=calcium_noise,
        amplitude_range=calcium_amplitude_range,
        rng=rng,
    )

    # -------------------------------------------------------------------------
    # Create Experiment object
    # -------------------------------------------------------------------------
    if verbose:
        print("Creating Experiment object...")

    # Build dynamic features - only add features that are actually referenced
    dynamic_features = {}

    # Add behavioral features only if needed
    if needs_head_direction:
        dynamic_features["head_direction"] = TimeSeries(data=head_direction, discrete=False, name="head_direction")
    if needs_x:
        dynamic_features["x"] = TimeSeries(data=positions[0, :], discrete=False, name="x")
    if needs_y:
        dynamic_features["y"] = TimeSeries(data=positions[1, :], discrete=False, name="y")
    if needs_speed:
        dynamic_features["speed"] = TimeSeries(data=speed_normalized, discrete=False, name="speed")
    if needs_position_2d:
        position_mts = MultiTimeSeries(
            [
                TimeSeries(data=positions[0, :], discrete=False, name="position_2d_0"),
                TimeSeries(data=positions[1, :], discrete=False, name="position_2d_1"),
            ],
            allow_zero_columns=True,
            name="position_2d"
        )
        dynamic_features["position_2d"] = position_mts

    # Add discrete features
    for feat_name, feat_data in discrete_features.items():
        dynamic_features[feat_name] = TimeSeries(data=feat_data, discrete=True, name=feat_name)

    # Add FBM features
    for feat_name, feat_data in fbm_features.items():
        dynamic_features[feat_name] = TimeSeries(data=feat_data, discrete=False, name=feat_name)

    # Static features
    static_features = {
        "fps": fps,
        "t_rise_sec": DEFAULT_T_RISE,
        "t_off_sec": min(decay_time, duration / 10),
    }

    # Create experiment with ground_truth attached
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
        ground_truth=ground_truth,
        verbose=False,
        optimize_kinetics=False,
        reconstruct_spikes=reconstruct_spikes,
    )

    if verbose:
        print(f"  Created experiment: {n_neurons} neurons, {n_frames} frames")
        print(f"  Features: {list(dynamic_features.keys())}")
        print(f"  Expected significant pairs: {len(exp.ground_truth['expected_pairs'])}")

    return exp


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
    >>> exp = generate_tuned_selectivity_exp(population, duration=30, verbose=False)
    >>> selectivity_info = ground_truth_to_selectivity_matrix(exp.ground_truth)
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


# =============================================================================
# Mixed Selectivity (from mixed_selectivity)
# =============================================================================


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
        Probability of a neuron being selective. Default: 0.3.
    multi_select_prob : float, optional
        Probability of mixed selectivity for selective neurons. Default: 0.4.
    weights_mode : str, optional
        Weight generation mode: 'random', 'dominant', 'equal'. Default: 'random'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_features, n_neurons)
        Selectivity matrix with weights. Non-zero = selective.
    """
    check_positive(n_neurons=n_neurons, n_features=n_features)
    check_unit(selectivity_prob=selectivity_prob, multi_select_prob=multi_select_prob)

    valid_modes = ["random", "dominant", "equal"]
    if weights_mode not in valid_modes:
        raise ValueError(f"weights_mode must be one of {valid_modes}")

    rng = np.random.default_rng(seed)

    selectivity_matrix = np.zeros((n_features, n_neurons))

    for j in range(n_neurons):
        if rng.random() > selectivity_prob:
            continue

        n_select = rng.choice([2, 3], p=[0.7, 0.3]) if rng.random() < multi_select_prob else 1
        n_select = min(n_select, n_features)
        if n_select == 0:
            continue

        selected = rng.choice(n_features, n_select, replace=False)

        if weights_mode == "equal":
            weights = np.ones(n_select) / n_select
        elif weights_mode == "dominant":
            weights = rng.dirichlet([5] + [1] * (n_select - 1))
        else:
            weights = rng.dirichlet(np.ones(n_select))

        selectivity_matrix[selected, j] = weights

    return selectivity_matrix


def generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=4,
    n_continuous_feats=4,
    n_neurons=50,
    selectivity_prob=0.8,
    multi_select_prob=0.5,
    weights_mode="random",
    duration=1200,
    seed=42,
    fps=20,
    verbose=True,
    baseline_rate=0.1,
    peak_rate=2.0,
    skip_prob=0.1,
    calcium_amplitude_range=(0.5, 2),
    decay_time=2,
    calcium_noise=0.02,
    hurst=0.3,
    target_active_fraction=0.05,
    avg_active_duration=0.5,
):
    """Generate synthetic experiment with mixed selectivity.

    Thin wrapper: rolls dice using Dirichlet, then calls canonical generator.

    Parameters
    ----------
    n_discrete_feats : int
        Number of discrete (event) features. Default: 4.
    n_continuous_feats : int
        Number of continuous (FBM) features. Default: 4.
    n_neurons : int
        Number of neurons. Default: 50.
    selectivity_prob : float
        Probability of neuron being selective. Default: 0.8.
    multi_select_prob : float
        Probability of mixed selectivity. Default: 0.5.
    weights_mode : str
        Weight mode: 'random', 'dominant', 'equal'. Default: 'random'.
    duration : float
        Duration in seconds. Default: 1200.
    seed : int
        Random seed. Default: 42.
    fps : float
        Sampling rate in Hz. Default: 20.
    verbose : bool
        Print progress. Default: True.
    baseline_rate : float
        Baseline firing rate (spikes/frame). Default: 0.1.
    peak_rate : float
        Peak firing rate during selectivity (spikes/frame). Default: 2.0.
    skip_prob : float
        Probability of skipping spikes. Default: 0.1.
    calcium_amplitude_range : tuple
        Calcium amplitude range. Default: (0.5, 2).
    decay_time : float
        Calcium decay time. Default: 2.
    calcium_noise : float
        Calcium noise standard deviation. Default: 0.1.
    hurst : float
        Hurst parameter for FBM. Default: 0.3.
    target_active_fraction : float
        Target active fraction for events. Default: 0.05.
    avg_active_duration : float
        Average active duration for events. Default: 0.5.

    Returns
    -------
    exp : Experiment
        Experiment with ground_truth containing:
        - 'expected_pairs': list of (neuron_idx, feature_name)
        - 'selectivity_matrix': ndarray (n_features, n_neurons)
        - 'feature_names': list of canonical feature names
    """
    # Input validation
    check_positive(n_neurons=n_neurons, duration=duration, fps=fps, decay_time=decay_time)
    check_nonnegative(n_discrete_feats=n_discrete_feats, n_continuous_feats=n_continuous_feats)
    check_unit(selectivity_prob=selectivity_prob, multi_select_prob=multi_select_prob, skip_prob=skip_prob)

    # Build CANONICAL feature names
    n_features = n_discrete_feats + n_continuous_feats
    feature_names = [f"event_{i}" for i in range(n_discrete_feats)]
    feature_names += [f"fbm_{i}" for i in range(n_continuous_feats)]

    # Roll dice
    if verbose:
        print(f"Generating selectivity patterns for {n_neurons} neurons...")

    selectivity_matrix = generate_multiselectivity_patterns(
        n_neurons,
        n_features,
        selectivity_prob=selectivity_prob,
        multi_select_prob=multi_select_prob,
        weights_mode=weights_mode,
        seed=seed + 300 if seed is not None else None,
    )

    # Convert to population config
    population_config = _selectivity_matrix_to_config(
        selectivity_matrix,
        feature_names,
        tuning_type="threshold",
        combination_mode="weighted_sum",
        baseline_rate=baseline_rate,
        peak_rate=peak_rate,
    )

    # Call canonical generator - ONE Experiment
    exp = generate_tuned_selectivity_exp(
        population=population_config,
        duration=duration,
        fps=fps,
        baseline_rate=baseline_rate,
        peak_rate=peak_rate,
        decay_time=decay_time,
        calcium_noise=calcium_noise,
        calcium_amplitude_range=calcium_amplitude_range,
        n_discrete_features=n_discrete_feats,
        event_active_fraction=target_active_fraction,
        event_avg_duration=avg_active_duration,
        skip_prob=skip_prob,
        hurst=hurst,
        seed=seed,
        verbose=verbose,
    )

    # Build expected_pairs
    expected_pairs = []
    for feat_idx, neuron_idx in zip(*np.where(selectivity_matrix > 0)):
        expected_pairs.append((neuron_idx, feature_names[feat_idx]))

    # Attach ground truth
    exp.ground_truth = {
        "expected_pairs": expected_pairs,
        "selectivity_matrix": selectivity_matrix,
        "feature_names": feature_names,
    }

    exp.signature = "SyntheticMixedSelectivity"

    return exp


# =============================================================================
# Circular Manifold Generators (from manifold_circular)
# =============================================================================


def generate_circular_manifold_neurons(
    n_neurons,
    head_direction,
    kappa=4.0,
    seed=None,
    **kwargs,
):
    """
    Generate population of head direction cells with Von Mises tuning.

    Creates a population of neurons with uniformly distributed preferred
    directions on the circle, each responding to head direction with
    Von Mises tuning curves. Includes realistic noise and ensures
    non-negative firing rates suitable for calcium imaging.

    Firing rates are designed to avoid calcium saturation. Default peak_rate
    of 1.0 Hz produces realistic calcium signals, while rates above 2 Hz may
    cause saturation in GCaMP indicators.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population. Must be positive.
    head_direction : ndarray
        Head direction trajectory in radians. Shape: (n_timepoints,).
    kappa : float, optional
        Concentration parameter for Von Mises tuning curves.
        Typical values: 2-8 (higher = narrower tuning). Default is 4.0.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict
        Additional parameters from DEFAULT_SYNTHETIC_PARAMS:
        baseline_rate, peak_rate, firing_noise.

    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints) with firing rates in Hz.
        All values are non-negative.
    preferred_directions : ndarray
        Preferred direction for each neuron in radians [0, 2*pi).
        Shape: (n_neurons,).
    """
    # Extract parameters using helper
    p = _extract_synthetic_params(**kwargs)
    baseline_rate = p["baseline_rate"]
    peak_rate = p["peak_rate"]
    firing_noise = p["firing_noise"]

    # Input validation
    check_positive(n_neurons=n_neurons)
    check_nonnegative(firing_noise=firing_noise)

    # Validate firing rate
    validate_peak_rate(peak_rate, context="generate_circular_manifold_neurons")

    rng = np.random.default_rng(seed)

    head_direction = np.asarray(head_direction)
    n_timepoints = len(head_direction)

    # Uniformly distribute preferred directions around the circle
    preferred_directions = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)

    # Add small random jitter to break perfect symmetry
    JITTER_STD = 0.1  # radians, approximately 5.7 degrees
    jitter = rng.normal(0, JITTER_STD, n_neurons)
    preferred_directions = (preferred_directions + jitter) % (2 * np.pi)

    # Generate firing rates for each neuron
    firing_rates = np.zeros((n_neurons, n_timepoints))

    for i in range(n_neurons):
        # Von Mises tuning curve
        tuning_response = von_mises_tuning_curve(head_direction, preferred_directions[i], kappa)

        # Scale to desired firing rate range
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * tuning_response

        # Add noise
        noise = rng.normal(0, firing_noise, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)  # Ensure non-negative

        firing_rates[i, :] = firing_rate

    return firing_rates, preferred_directions


def generate_circular_manifold_data(
    n_neurons,
    kappa=4.0,
    step_std=0.1,
    seed=None,
    verbose=True,
    **kwargs,
):
    """
    Generate synthetic data with neurons on circular manifold (head direction cells).

    Creates a complete dataset with head direction trajectory, neural responses
    with Von Mises tuning, and realistic calcium imaging signals including noise.

    Parameters
    ----------
    n_neurons : int
        Number of neurons. Must be positive.
    kappa : float, optional
        Von Mises concentration parameter (tuning width).
        Default is 4.0. Higher values give narrower tuning.
    step_std : float, optional
        Standard deviation of head direction random walk steps in radians.
        Must be non-negative. Default is 0.1.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default is True.
    **kwargs : dict
        Additional parameters from DEFAULT_SYNTHETIC_PARAMS:
        duration, fps, baseline_rate, peak_rate, firing_noise,
        decay_time, calcium_noise, amplitude_range.

    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    head_direction : ndarray
        Head direction trajectory in radians [0, 2*pi).
    preferred_directions : ndarray
        Preferred direction for each neuron in radians.
    firing_rates : ndarray
        Underlying firing rates in Hz.
    """
    # Extract parameters using helper
    p = _extract_synthetic_params(**kwargs)
    duration = p["duration"]
    sampling_rate = p["fps"]
    baseline_rate = p["baseline_rate"]
    peak_rate = p["peak_rate"]
    firing_noise = p["firing_noise"]
    decay_time = p["decay_time"]
    calcium_noise = p["calcium_noise"]
    amplitude_range = p["amplitude_range"]

    # Input validation
    check_positive(
        n_neurons=n_neurons, duration=duration, sampling_rate=sampling_rate, decay_time=decay_time
    )
    check_nonnegative(
        step_std=step_std,
        baseline_rate=baseline_rate,
        firing_noise=firing_noise,
        calcium_noise=calcium_noise,
    )

    rng = np.random.default_rng(seed)

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
        seed=(seed + 1) if seed is not None else None,
        baseline_rate=baseline_rate,
        peak_rate=peak_rate,
        firing_noise=firing_noise,
    )

    # Convert firing rates to calcium signals
    if verbose:
        print("  Converting to calcium signals...")
    calcium_signals = _firing_rates_to_calcium(
        firing_rates=firing_rates,
        fps=sampling_rate,
        duration=duration,
        decay_time=decay_time,
        calcium_noise=calcium_noise,
        amplitude_range=amplitude_range,
        rng=rng,
    )

    if verbose:
        print("  Done!")

    return calcium_signals, head_direction, preferred_directions, firing_rates


def generate_circular_manifold_exp(
    n_neurons=100,
    kappa=4.0,
    step_std=0.1,
    add_mixed_features=False,
    seed=None,
    verbose=True,
    return_info=False,
    **kwargs,
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
    kappa : float, optional
        Von Mises concentration parameter (tuning width).
        Higher values give narrower tuning. Must be positive.
        Typical values: 2-8. Default is 4.0.
    step_std : float, optional
        Head direction random walk step size in radians.
        Must be non-negative. Default is 0.1 (~5.7 degrees).
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
    **kwargs : dict
        Additional parameters from DEFAULT_SYNTHETIC_PARAMS:
        duration, fps, baseline_rate, peak_rate, firing_noise,
        decay_time, calcium_noise, amplitude_range.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object containing:
        - calcium signals as main data
        - static features: fps, decay times, manifold parameters
        - dynamic features: head_direction (and circular_angle if requested)
    info : dict, optional
        Only returned if return_info=True. Contains:
        - 'manifold_type': "circular"
        - 'n_neurons': number of neurons
        - 'head_direction': trajectory array
        - 'preferred_directions': array of preferred directions
        - 'firing_rates': underlying firing rates
        - 'parameters': dict of all generation parameters
    """
    # Extract parameters using helper
    p = _extract_synthetic_params(**kwargs)
    duration = p["duration"]
    fps = p["fps"]
    baseline_rate = p["baseline_rate"]
    peak_rate = p["peak_rate"]
    firing_noise = p["firing_noise"]
    decay_time = p["decay_time"]
    calcium_noise = p["calcium_noise"]

    # Input validation
    check_positive(
        n_neurons=n_neurons,
        duration=duration,
        fps=fps,
        kappa=kappa,
        peak_rate=peak_rate,
        decay_time=decay_time,
    )
    check_nonnegative(
        step_std=step_std,
        baseline_rate=baseline_rate,
        firing_noise=firing_noise,
        calcium_noise=calcium_noise,
    )

    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")

    if baseline_rate >= peak_rate:
        raise ValueError(
            f"baseline_rate ({baseline_rate}) must be less than peak_rate ({peak_rate})"
        )

    # Calculate effective decay time for shuffle mask
    effective_decay_time = get_effective_decay_time(decay_time, duration, verbose)

    # Generate data - pass kwargs through to share defaults
    calcium, head_direction, preferred_directions, firing_rates = generate_circular_manifold_data(
        n_neurons=n_neurons,
        kappa=kappa,
        step_std=step_std,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )

    # Create static features
    static_features = {
        "fps": fps,
        "t_rise_sec": DEFAULT_T_RISE,
        "t_off_sec": effective_decay_time,  # Use effective decay time for shuffle mask
        "manifold_type": "circular",
        "kappa": kappa,
        "baseline_rate": baseline_rate,
        "peak_rate": peak_rate,
    }

    # Create dynamic features
    head_direction_ts = TimeSeries(data=head_direction, discrete=False, name="head_direction")

    dynamic_features = {"head_direction": head_direction_ts}

    # Add circular_angle MultiTimeSeries if requested
    if add_mixed_features:
        # Create circular_angle as MultiTimeSeries with cos and sin components
        # This is the proper representation for circular variables
        circular_angle_mts = circular_to_cos_sin(
            head_direction, period=2 * np.pi, name="circular_angle"
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
        optimize_kinetics=False,
        reconstruct_spikes=None,  # Don't reconstruct spikes for synthetic data
    )

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
                "firing_noise": firing_noise,
                "decay_time": decay_time,
                "calcium_noise": calcium_noise,
            },
        }
        return exp, info

    return exp


# =============================================================================
# 2D Spatial Manifold Generators (from manifold_spatial_2d)
# =============================================================================


def generate_2d_manifold_neurons(
    n_neurons,
    positions,
    field_sigma=0.1,
    grid_arrangement=True,
    bounds=(0, 1),
    seed=None,
    **kwargs,
):
    """
    Generate population of place cells with 2D Gaussian place fields.

    Creates a population of neurons with place fields either arranged in a
    regular grid or randomly distributed. Each neuron responds maximally
    when the animal is at its place field center. Firing rates are designed
    for realistic calcium imaging.

    Firing rates are designed to avoid calcium saturation. Default peak_rate
    of 1.0 Hz produces realistic calcium signals, while rates above 2 Hz may
    cause saturation in GCaMP indicators.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population. Must be positive.
    positions : ndarray
        Shape (2, n_timepoints) with x, y positions.
    field_sigma : float, optional
        Width of place fields. Must be positive. Default is 0.1.
    grid_arrangement : bool, optional
        If True, arrange place fields in a grid. Otherwise random.
        Default is True.
    bounds : tuple, optional
        (min, max) bounds for place field centers. Default is (0, 1).
        Must have bounds[0] < bounds[1].
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict
        Additional parameters from DEFAULT_SYNTHETIC_PARAMS:
        baseline_rate, peak_rate, firing_noise.

    Returns
    -------
    firing_rates : ndarray
        Shape (n_neurons, n_timepoints) with firing rates in Hz.
        All values are non-negative.
    place_field_centers : ndarray
        Shape (n_neurons, 2) with x, y coordinates of place field centers.
    """
    # Extract parameters using helper
    p = _extract_synthetic_params(**kwargs)
    baseline_rate = p["baseline_rate"]
    peak_rate = p["peak_rate"]
    firing_noise = p["firing_noise"]

    # Input validation
    check_positive(n_neurons=n_neurons, field_sigma=field_sigma)
    check_nonnegative(baseline_rate=baseline_rate, firing_noise=firing_noise)

    # Validate firing rate
    validate_peak_rate(peak_rate, context="generate_2d_manifold_neurons")

    # Check parameter relationships
    if baseline_rate > peak_rate:
        raise ValueError(f"baseline_rate ({baseline_rate}) must be <= peak_rate ({peak_rate})")

    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be (min, max) with min < max")

    rng = np.random.default_rng(seed)

    positions = np.asarray(positions)
    if positions.shape[0] != 2:
        raise ValueError("positions must have shape (2, n_timepoints)")
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
        jitter = rng.normal(0, 0.02, place_field_centers.shape)
        place_field_centers += jitter
        place_field_centers = np.clip(place_field_centers, bounds[0], bounds[1])
    else:
        # Random placement
        place_field_centers = rng.uniform(bounds[0], bounds[1], (n_neurons, 2))

    # Generate firing rates
    firing_rates = np.zeros((n_neurons, n_timepoints))

    for i in range(n_neurons):
        # Gaussian place field
        place_response = gaussian_place_field(positions, place_field_centers[i], field_sigma)

        # Scale to desired firing rate range
        firing_rate = baseline_rate + (peak_rate - baseline_rate) * place_response

        # Add noise
        noise = rng.normal(0, firing_noise, n_timepoints)
        firing_rate = np.maximum(0, firing_rate + noise)

        firing_rates[i, :] = firing_rate

    return firing_rates, place_field_centers


def generate_2d_manifold_data(
    n_neurons,
    field_sigma=0.1,
    step_size=0.02,
    momentum=0.8,
    grid_arrangement=True,
    bounds=(0, 1),
    seed=None,
    verbose=True,
    **kwargs,
):
    """
    Generate synthetic data with neurons on 2D spatial manifold (place cells).

    Creates a complete dataset including spatial trajectory, place cell
    responses, and realistic calcium imaging signals. Useful for testing
    spatial coding analyses.

    Parameters
    ----------
    n_neurons : int
        Number of neurons. Must be positive.
    field_sigma : float, optional
        Width of place fields. Must be positive. Default is 0.1.
    step_size : float, optional
        Step size for random walk. Must be positive. Default is 0.02.
    momentum : float, optional
        Momentum for smoother trajectories. Must be in [0, 1]. Default is 0.8.
    grid_arrangement : bool, optional
        If True, arrange place fields in grid. Default is True.
    bounds : tuple, optional
        Spatial bounds (min, max). Default is (0, 1).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default is True.
    **kwargs : dict
        Additional parameters from DEFAULT_SYNTHETIC_PARAMS:
        duration, fps, baseline_rate, peak_rate, firing_noise,
        decay_time, calcium_noise, amplitude_range.

    Returns
    -------
    calcium_signals : ndarray
        Calcium signals (n_neurons x n_timepoints).
    positions : ndarray
        Position trajectory (2 x n_timepoints) with x, y coordinates.
    place_field_centers : ndarray
        Place field centers (n_neurons x 2) with x, y coordinates.
    firing_rates : ndarray
        Underlying firing rates in Hz (n_neurons x n_timepoints).
    """
    # Extract parameters using helper
    p = _extract_synthetic_params(**kwargs)
    duration = p["duration"]
    sampling_rate = p["fps"]
    baseline_rate = p["baseline_rate"]
    peak_rate = p["peak_rate"]
    firing_noise = p["firing_noise"]
    decay_time = p["decay_time"]
    calcium_noise = p["calcium_noise"]
    amplitude_range = p["amplitude_range"]

    # Input validation
    check_positive(
        n_neurons=n_neurons,
        duration=duration,
        sampling_rate=sampling_rate,
        field_sigma=field_sigma,
        step_size=step_size,
        decay_time=decay_time,
    )
    check_nonnegative(
        baseline_rate=baseline_rate, firing_noise=firing_noise, calcium_noise=calcium_noise
    )

    if not isinstance(momentum, (int, float)):
        raise TypeError("momentum must be numeric")
    if not 0 <= momentum <= 1:
        raise ValueError("momentum must be in range [0, 1]")

    rng = np.random.default_rng(seed)

    n_timepoints = int(duration * sampling_rate)

    if verbose:
        print(f"Generating 2D spatial manifold data: {n_neurons} neurons, {duration}s")

    # Generate spatial trajectory
    if verbose:
        print("  Generating 2D random walk trajectory...")
    positions = generate_2d_random_walk(n_timepoints, bounds, step_size, momentum, seed)

    # Generate neural responses - pass kwargs through to share defaults
    if verbose:
        print("  Generating neural responses with place fields...")
    firing_rates, place_field_centers = generate_2d_manifold_neurons(
        n_neurons,
        positions,
        field_sigma=field_sigma,
        grid_arrangement=grid_arrangement,
        bounds=bounds,
        seed=(seed + 1) if seed is not None else None,
        **kwargs,
    )

    # Convert to calcium signals
    if verbose:
        print("  Converting to calcium signals...")
    calcium_signals = _firing_rates_to_calcium(
        firing_rates=firing_rates,
        fps=sampling_rate,
        duration=duration,
        decay_time=decay_time,
        calcium_noise=calcium_noise,
        amplitude_range=amplitude_range,
        rng=rng,
    )

    if verbose:
        print("  Done!")

    return calcium_signals, positions, place_field_centers, firing_rates


def generate_2d_manifold_exp(
    n_neurons=100,
    field_sigma=0.1,
    step_size=0.02,
    momentum=0.8,
    grid_arrangement=True,
    bounds=(0, 1),
    seed=None,
    verbose=True,
    return_info=False,
    **kwargs,
):
    """
    Generate complete experiment with 2D spatial manifold (place cells).

    Creates a DRIADA Experiment object with synthetic hippocampal place cell
    data, including calcium imaging signals and behavioral trajectory.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons. Must be positive. Default is 100.
    field_sigma : float, optional
        Place field width. Must be positive. Default is 0.1.
    step_size : float, optional
        Random walk step size. Must be positive. Default is 0.02.
    momentum : float, optional
        Trajectory smoothness factor. Must be in [0, 1]. Default is 0.8.
    grid_arrangement : bool, optional
        If True, arrange place fields in grid. Default is True.
    bounds : tuple, optional
        Spatial bounds (min, max). Default is (0, 1).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress messages. Default is True.
    return_info : bool, optional
        If True, return (exp, info) tuple with additional information.
        Default is False.
    **kwargs : dict
        Additional parameters from DEFAULT_SYNTHETIC_PARAMS:
        duration, fps, baseline_rate, peak_rate, firing_noise,
        decay_time, calcium_noise, amplitude_range.

    Returns
    -------
    exp : Experiment
        DRIADA Experiment object with 2D spatial manifold data.
    info : dict, optional
        If return_info=True, dictionary with manifold info including:
        manifold_type, n_neurons, positions, place_field_centers,
        firing_rates, and all parameters.
    """
    # Extract parameters using helper
    p = _extract_synthetic_params(**kwargs)
    duration = p["duration"]
    fps = p["fps"]
    baseline_rate = p["baseline_rate"]
    peak_rate = p["peak_rate"]
    firing_noise = p["firing_noise"]
    decay_time = p["decay_time"]
    calcium_noise = p["calcium_noise"]

    # Calculate effective decay time for shuffle mask
    effective_decay_time = get_effective_decay_time(decay_time, duration, verbose)

    # Generate data - pass kwargs through to share defaults
    calcium, positions, place_field_centers, firing_rates = generate_2d_manifold_data(
        n_neurons=n_neurons,
        field_sigma=field_sigma,
        step_size=step_size,
        momentum=momentum,
        grid_arrangement=grid_arrangement,
        bounds=bounds,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )

    # Create static features
    static_features = {
        "fps": fps,
        "t_rise_sec": DEFAULT_T_RISE,
        "t_off_sec": effective_decay_time,  # Use effective decay time for shuffle mask
        "manifold_type": "2d_spatial",
        "field_sigma": field_sigma,
        "baseline_rate": baseline_rate,
        "peak_rate": peak_rate,
        "grid_arrangement": grid_arrangement,
    }

    # Create dynamic features
    # Note: allow_zero_columns=True because random walk can hit boundaries (e.g., (0,0))
    position_ts = MultiTimeSeries(
        [
            TimeSeries(positions[0, :], discrete=False, name="position_2d_0"),
            TimeSeries(positions[1, :], discrete=False, name="position_2d_1"),
        ],
        allow_zero_columns=True,
        name="position_2d"
    )

    # Also create separate x, y features
    x_ts = TimeSeries(data=positions[0, :], discrete=False, name="x")

    y_ts = TimeSeries(data=positions[1, :], discrete=False, name="y")

    dynamic_features = {"position_2d": position_ts, "x": x_ts, "y": y_ts}

    # Store additional information
    static_features["place_field_centers"] = place_field_centers

    # Create experiment
    exp = Experiment(
        signature="2d_spatial_manifold_exp",
        calcium=calcium,
        spikes=None,
        static_features=static_features,
        dynamic_features=dynamic_features,
        exp_identificators={
            "manifold": "2d_spatial",
            "n_neurons": n_neurons,
            "duration": duration,
        },
        optimize_kinetics=False,
        reconstruct_spikes=None,  # Don't reconstruct spikes for synthetic data
    )

    # Create info dictionary if requested
    if return_info:
        info = {
            "manifold_type": "2d_spatial",
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
                "firing_noise": firing_noise,
                "grid_arrangement": grid_arrangement,
                "decay_time": decay_time,
                "calcium_noise": calcium_noise,
                "bounds": bounds,
            },
        }
        return exp, info

    return exp


# =============================================================================
# Legacy/Convenience Wrappers (from experiment_generators)
# =============================================================================


def generate_synthetic_data(
    nfeats,
    nneurons,
    ftype="c",
    duration=600,
    seed=42,
    sampling_rate=20.0,
    baseline_rate=0.1,
    peak_rate=2.0,
    skip_prob=0.0,
    hurst=0.5,
    calcium_amplitude_range=(0.5, 2),
    decay_time=2,
    avg_islands=10,
    avg_duration=5,
    calcium_noise=0.02,
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
    baseline_rate : float
        Baseline firing rate in Hz. Must be non-negative.
    peak_rate : float
        Active firing rate in Hz. Must be non-negative.
    skip_prob : float
        Probability of skipping islands. Must be in [0, 1].
    hurst : float
        Hurst parameter for FBM (0-1). 0.5 = random walk.
    calcium_amplitude_range : tuple
        (min, max) amplitude range for calcium events.
    decay_time : float
        Calcium decay time constant in seconds. Must be positive.
    avg_islands : int
        Average number of islands for discrete features. Must be positive.
    avg_duration : int
        Average duration of islands in seconds. Must be positive.
    calcium_noise : float
        Noise standard deviation for calcium signal. Must be non-negative.
    verbose : bool
        Print progress messages.
    pregenerated_features : list, optional
        Pre-generated feature arrays to use instead of generating new ones.
    apply_random_neuron_shifts : bool
        Apply random circular shifts to break correlations between neurons.

    Returns
    -------
    features : ndarray
        Feature time series of shape (nfeats, n_timepoints).
    signals : ndarray
        Neural calcium signals of shape (nneurons, n_timepoints).
    ground_truth : ndarray
        Ground truth connectivity matrix of shape (nfeats, nneurons).
    """
    # Input validation
    check_nonnegative(
        nfeats=nfeats,
        nneurons=nneurons,
        duration=duration,
        sampling_rate=sampling_rate,
        baseline_rate=baseline_rate,
        peak_rate=peak_rate,
        skip_prob=skip_prob,
        hurst=hurst,
        decay_time=decay_time,
        calcium_noise=calcium_noise,
        avg_islands=avg_islands,
        avg_duration=avg_duration,
    )

    # Additional validation for ranges
    if not 0 <= skip_prob <= 1:
        raise ValueError(f"skip_prob must be in [0, 1], got {skip_prob}")
    if not 0 <= hurst <= 1:
        raise ValueError(f"hurst must be in [0, 1], got {hurst}")
    if len(calcium_amplitude_range) != 2 or calcium_amplitude_range[0] > calcium_amplitude_range[1]:
        raise ValueError(
            f"calcium_amplitude_range must be (min, max) with min <= max, "
            f"got {calcium_amplitude_range}"
        )
    check_nonnegative(ampl_min=calcium_amplitude_range[0], ampl_max=calcium_amplitude_range[1])
    if ftype not in ["c", "d"]:
        raise ValueError(f"ftype must be 'c' or 'd', got '{ftype}'")

    rng = np.random.default_rng(seed)

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
                        # Use seed for reproducibility via generate_binary_time_series
                        feature_seed = seed + i if seed is not None else None
                        binary_series = generate_binary_time_series(
                            length, avg_islands, avg_duration * sampling_rate, seed=feature_seed
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
        fois = rng.choice(np.arange(nfeats), size=nneurons)
        gt[fois, np.arange(nneurons)] = 1  # add info about ground truth feature-signal connections
    else:
        # If no features, neurons won't be selective to any feature
        fois = np.full(nneurons, -1)  # Use -1 to indicate no feature selection
    all_signals = []

    for j in tqdm.tqdm(np.arange(nneurons), disable=not verbose):
        foi = fois[j]

        # Generate unique seeds for this neuron's random operations
        neuron_base_seed = int(rng.integers(0, 2**31))

        # Handle case where there are no features
        if foi == -1 or nfeats == 0:
            # Generate random baseline activity
            binary_series = generate_binary_time_series(
                length, avg_islands // 2, avg_duration * sampling_rate // 2,
                seed=neuron_base_seed
            )
        elif ftype == "c":
            csignal = all_feats[foi].copy()  # Make a copy to avoid modifying the original

            # Apply random per-neuron shift to break correlations
            if apply_random_neuron_shifts:
                # Apply a unique random shift for this neuron
                neuron_shift = rng.integers(0, length)
                csignal = np.roll(csignal, neuron_shift)
                if verbose and j < 3:  # Print for first 3 neurons only
                    print(
                        f"      Neuron {j}: Applied shift={neuron_shift} to continuous feature {foi}"
                    )

            loc, lower_border, upper_border = select_signal_roi(csignal, seed=neuron_base_seed)
            # Generate binary series from a continuous one
            binary_series = np.zeros(length)
            binary_series[np.where((csignal >= lower_border) & (csignal <= upper_border))] = 1

        elif ftype == "d":
            binary_series = all_feats[foi].copy()  # Make a copy

            # Apply random per-neuron shift to break correlations
            if apply_random_neuron_shifts:
                # Apply a unique random shift for this neuron
                neuron_shift = rng.integers(0, length)
                binary_series = np.roll(binary_series, neuron_shift)
                if verbose and j < 3:  # Print for first 3 neurons only
                    print(
                        f"      Neuron {j}: Applied shift={neuron_shift} to discrete feature {foi}"
                    )

        else:
            raise ValueError(f"Unknown feature type: {ftype}")

        # randomly skip some on periods
        mod_binary_series = delete_one_islands(binary_series, skip_prob, seed=neuron_base_seed + 1)

        # Apply Poisson process
        poisson_series = apply_poisson_to_binary_series(
            mod_binary_series, baseline_rate / sampling_rate, peak_rate / sampling_rate,
            seed=neuron_base_seed + 2
        )

        # Generate pseudo-calcium
        pseudo_calcium_signal = generate_pseudo_calcium_signal(
            duration=duration,
            events=poisson_series,
            sampling_rate=sampling_rate,
            amplitude_range=calcium_amplitude_range,
            decay_time=decay_time,
            noise_std=calcium_noise,
            seed=neuron_base_seed + 3,
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
    """
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

    # Handle with_spikes parameter - reconstruct spikes from calcium if requested
    with_spikes = kwargs.get("with_spikes", False)
    if with_spikes:
        tuned_kwargs["reconstruct_spikes"] = "wavelet"

    # Generate experiment using canonical generator
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=duration,
        fps=fps,
        n_discrete_features=n_dfeats,
        seed=seed,
        **tuned_kwargs,
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
    manifold_params=None,
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
    manifold_params : dict, optional
        Override tuning parameters for manifold cells. Supported keys:
        - field_sigma: Size of place field (default: 0.15)
        - baseline_rate: Baseline firing rate (default: 0.05)
        - peak_rate: Peak firing rate (default: 2.0)
        - firing_noise: Firing rate noise (default: 0.05)
        - decay_time: Calcium decay time (default: 2.0)
        - calcium_noise: Calcium signal noise (default: 0.02)

    Returns
    -------
    exp : Experiment
        Experiment object with mixed population. Ground truth accessible via
        exp.ground_truth containing expected_pairs, tuning_parameters, etc.
    info : dict (only if return_info=True)
        Dictionary containing ground_truth for backward compatibility.

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

    # Build tuning_defaults from manifold_params if provided
    tuning_defaults = None
    baseline_rate = 0.05
    peak_rate = 2.0
    decay_time = 2.0
    calcium_noise = 0.02

    if manifold_params is not None:
        tuning_defaults = {}
        if "field_sigma" in manifold_params:
            tuning_defaults["position_2d"] = {"sigma": manifold_params["field_sigma"]}
        if "baseline_rate" in manifold_params:
            baseline_rate = manifold_params["baseline_rate"]
        if "peak_rate" in manifold_params:
            peak_rate = manifold_params["peak_rate"]
        if "decay_time" in manifold_params:
            decay_time = manifold_params["decay_time"]
        # Support new canonical name
        if "calcium_noise" in manifold_params:
            calcium_noise = manifold_params["calcium_noise"]
        # Legacy compatibility: calcium_noise_std and noise_std map to calcium_noise
        if "calcium_noise_std" in manifold_params:
            calcium_noise = manifold_params["calcium_noise_std"]
        if "noise_std" in manifold_params:
            calcium_noise = manifold_params["noise_std"]

    # Generate experiment using canonical generator
    exp = generate_tuned_selectivity_exp(
        population=population,
        tuning_defaults=tuning_defaults,
        duration=duration,
        fps=fps,
        baseline_rate=baseline_rate,
        peak_rate=peak_rate,
        decay_time=decay_time,
        calcium_noise=calcium_noise,
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


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Canonical generator
    "generate_tuned_selectivity_exp",
    "ground_truth_to_selectivity_matrix",
    # Helper (exposed for mixed_selectivity wrapper)
    "_selectivity_matrix_to_config",
    "_get_tuning_param",
    "_generate_random_tuning_param",
    # Mixed selectivity
    "generate_multiselectivity_patterns",
    "generate_synthetic_exp_with_mixed_selectivity",
    # Circular manifold
    "generate_circular_manifold_neurons",
    "generate_circular_manifold_data",
    "generate_circular_manifold_exp",
    # 2D spatial manifold
    "generate_2d_manifold_neurons",
    "generate_2d_manifold_data",
    "generate_2d_manifold_exp",
    # Legacy wrappers
    "generate_synthetic_data",
    "generate_synthetic_exp",
    "generate_mixed_population_exp",
]
