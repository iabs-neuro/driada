"""
Spatial Analysis Utilities for Neural Data
==========================================

This module provides comprehensive spatial analysis tools for neural data,
particularly for analyzing place cells, grid cells, and other spatially-modulated neurons.

Key functionality:
- Place field detection and analysis
- Grid score computation
- Spatial information metrics
- Position decoding
- Speed/direction filtering
- High-level spatial analysis pipelines
"""

import numpy as np
from scipy import ndimage, signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from typing import Optional, Tuple, Dict, List, Union
import logging

try:
    from ..information import TimeSeries, MultiTimeSeries, get_sim
    from .data import check_positive
except ImportError:
    # For standalone module execution (e.g., doctests)
    from driada.information import TimeSeries, MultiTimeSeries, get_sim
    from driada.utils.data import check_positive


def compute_occupancy_map(
    positions: np.ndarray,
    fps: float = 1.0,
    arena_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    bin_size: float = 0.025,
    min_occupancy: float = 0.1,
    smooth_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D occupancy map from position data.

    Parameters
    ----------
    positions : np.ndarray, shape (n_samples, 2)
        X, Y positions over time
    fps : float
        Frames per second (sampling frequency). Default 1.0 assumes 
        one frame per second.
    arena_bounds : tuple of tuples, optional
        ((x_min, x_max), (y_min, y_max)). If None, inferred from data
    bin_size : float
        Size of spatial bins in same units as positions
    min_occupancy : float
        Minimum occupancy time (seconds) for valid bins
    smooth_sigma : float, optional
        Gaussian smoothing sigma in bins. If None, no smoothing

    Returns
    -------
    occupancy_map : np.ndarray
        2D occupancy map in seconds
    x_edges : np.ndarray
        X bin edges
    y_edges : np.ndarray
        Y bin edges
    
    Raises
    ------
    ValueError
        If positions is not 2D or fps is non-positive
    
    Examples
    --------
    >>> positions = np.random.rand(1000, 2)  # 1000 frames
    >>> # With 30 fps recording
    >>> occ_map, x_edges, y_edges = compute_occupancy_map(positions, fps=30.0)
    >>> # occ_map contains time in seconds per bin    """
    if positions.shape[1] != 2:
        raise ValueError(f"Positions must be 2D, got shape {positions.shape}")
    
    check_positive(fps=fps)

    # Determine arena bounds
    if arena_bounds is None:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        # Add small margin
        margin = 0.05 * max(x_max - x_min, y_max - y_min)
        arena_bounds = (
            (x_min - margin, x_max + margin),
            (y_min - margin, y_max + margin),
        )

    # Create bins
    x_edges = np.arange(arena_bounds[0][0], arena_bounds[0][1] + bin_size, bin_size)
    y_edges = np.arange(arena_bounds[1][0], arena_bounds[1][1] + bin_size, bin_size)

    # Compute occupancy
    occupancy, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[x_edges, y_edges]
    )

    # Convert sample counts to time in seconds
    occupancy = occupancy / fps

    # Smooth if requested
    if smooth_sigma is not None and smooth_sigma > 0:
        occupancy = ndimage.gaussian_filter(occupancy, sigma=smooth_sigma)

    # Set unvisited bins to NaN
    occupancy[occupancy < min_occupancy] = np.nan

    return occupancy.T, x_edges, y_edges  # Transpose for correct orientation


def compute_rate_map(
    neural_signal: np.ndarray,
    positions: np.ndarray,
    occupancy_map: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    fps: float = 1.0,
    smooth_sigma: Optional[float] = 1.5,
) -> np.ndarray:
    """
    Compute firing rate map from neural signal and positions.

    Parameters
    ----------
    neural_signal : np.ndarray
        Neural signal (e.g., calcium fluorescence, firing rates, spike counts)
    positions : np.ndarray, shape (n_samples, 2)
        X, Y positions corresponding to neural signal
    occupancy_map : np.ndarray
        2D occupancy map in seconds from compute_occupancy_map
    x_edges : np.ndarray
        X bin edges from compute_occupancy_map
    y_edges : np.ndarray
        Y bin edges from compute_occupancy_map
    fps : float
        Frames per second (sampling frequency). Must match the fps used
        in compute_occupancy_map.
    smooth_sigma : float, optional
        Gaussian smoothing sigma in bins

    Returns
    -------
    rate_map : np.ndarray
        2D activity map (mean signal per spatial bin)
    
    Raises
    ------
    ValueError
        If positions shape doesn't match neural_signal length or fps is non-positive
    
    Examples
    --------
    >>> # Generate sample data
    >>> import numpy as np
    >>> positions = np.random.rand(1000, 2)  # 1000 frames of 2D positions
    >>> neural_signal = np.random.rand(1000)  # Corresponding neural activity
    
    >>> # Compute occupancy first
    >>> occ_map, x_edges, y_edges = compute_occupancy_map(positions, fps=30.0)
    >>> # Then compute rate map
    >>> rate_map = compute_rate_map(neural_signal, positions, occ_map, 
    ...                            x_edges, y_edges, fps=30.0)    """
    # Validate inputs
    if len(neural_signal) != len(positions):
        raise ValueError(f"neural_signal length ({len(neural_signal)}) must match "
                        f"positions length ({len(positions)})")
    check_positive(fps=fps)
    
    # For continuous signals (e.g., calcium), compute mean activity per bin
    # Use weighted 2D histogram to sum activity in each bin
    activity_sum, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=[x_edges, y_edges],
        weights=neural_signal,  # Use signal as weights
    )
    activity_sum = activity_sum.T  # Transpose for correct orientation

    # Convert occupancy from seconds back to sample counts for rate calculation
    # This ensures we use the same bins and smoothing as the occupancy map
    occupancy_counts = occupancy_map * fps
    
    # Compute mean activity per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_map = activity_sum / occupancy_counts
        rate_map[occupancy_counts == 0] = 0  # Set unvisited bins to 0
        rate_map[np.isnan(occupancy_map)] = 0  # Respect occupancy NaN mask

    # Smooth if requested
    if smooth_sigma is not None and smooth_sigma > 0:
        # Only smooth visited bins
        mask = occupancy_counts > 0
        if np.any(mask):
            # Create a smoothed version preserving only visited areas
            smoothed = ndimage.gaussian_filter(rate_map, sigma=smooth_sigma)
            smoothed_mask = ndimage.gaussian_filter(
                mask.astype(float), sigma=smooth_sigma
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_map = smoothed / smoothed_mask
                rate_map[~mask] = 0

    return rate_map


def extract_place_fields(
    rate_map: np.ndarray,
    min_peak_rate: float = 1.0,
    min_field_size: int = 9,
    peak_to_mean_ratio: float = 1.5,
) -> List[Dict[str, Union[float, Tuple[int, int]]]]:
    """
    Extract place fields from a rate map.

    Parameters
    ----------
    rate_map : np.ndarray
        2D firing rate map
    min_peak_rate : float
        Minimum peak firing rate for valid place field
    min_field_size : int
        Minimum number of contiguous bins for valid field
    peak_to_mean_ratio : float
        Minimum ratio of peak to mean rate in field

    Returns
    -------
    place_fields : list of dict
        List of place fields with properties:
        - peak_rate: Peak firing rate
        - center: (x, y) indices of field center
        - size: Number of bins in field
        - mean_rate: Mean rate within field
    
    Raises
    ------
    ValueError
        If input parameters are invalid
    
    Notes
    -----
    Uses 8-connectivity for contiguous region detection.
    
    Examples
    --------
    >>> rate_map = np.random.rand(40, 40)
    >>> fields = extract_place_fields(rate_map, min_peak_rate=0.8)    """
    # Validate inputs
    check_positive(min_peak_rate=min_peak_rate, min_field_size=min_field_size, 
                   peak_to_mean_ratio=peak_to_mean_ratio)
    
    # Threshold rate map
    mean_rate = np.nanmean(rate_map)
    threshold = mean_rate * peak_to_mean_ratio

    # Find connected regions above threshold
    binary_map = rate_map > threshold
    labeled_map, num_fields = ndimage.label(binary_map)

    place_fields = []

    for field_id in range(1, num_fields + 1):
        field_mask = labeled_map == field_id
        field_size = np.sum(field_mask)

        if field_size < min_field_size:
            continue

        field_rates = rate_map[field_mask]
        peak_rate = np.max(field_rates)

        if peak_rate < min_peak_rate:
            continue

        # Find center of mass (weighted by rate)
        y_indices, x_indices = np.where(field_mask)
        field_rates_for_com = rate_map[field_mask]
        center_y = int(np.round(np.average(y_indices, weights=field_rates_for_com)))
        center_x = int(np.round(np.average(x_indices, weights=field_rates_for_com)))

        place_fields.append(
            {
                "peak_rate": peak_rate,
                "center": (center_x, center_y),
                "size": field_size,
                "mean_rate": np.mean(field_rates),
            }
        )

    return place_fields


def compute_spatial_information_rate(
    rate_map: np.ndarray, occupancy_map: np.ndarray
) -> float:
    """
    Compute spatial information rate (bits/spike).

    Implements Skaggs et al. (1993) spatial information metric.

    Parameters
    ----------
    rate_map : np.ndarray
        2D firing rate map
    occupancy_map : np.ndarray
        2D occupancy map in seconds (time spent in each bin)

    Returns
    -------
    spatial_info : float
        Spatial information in bits/spike. Returns 0 if mean rate is 0.
    
    Notes
    -----
    Forces non-negative result. Uses log2 for bits.
    
    Examples
    --------
    >>> import numpy as np
    >>> positions = np.random.rand(1000, 2)
    >>> signal = np.random.rand(1000)  # Neural signal
    >>> occ_map, x_edges, y_edges = compute_occupancy_map(positions, fps=30.0)
    >>> rate_map = compute_rate_map(signal, positions, occ_map, x_edges, y_edges, fps=30.0)
    >>> si = compute_spatial_information_rate(rate_map, occ_map)    """
    # Normalize occupancy to get probability
    valid_mask = ~np.isnan(occupancy_map)
    p_i = occupancy_map[valid_mask] / np.sum(occupancy_map[valid_mask])
    r_i = rate_map[valid_mask]

    # Mean firing rate
    r_mean = np.sum(p_i * r_i)

    if r_mean == 0:
        return 0.0

    # Spatial information
    with np.errstate(divide="ignore", invalid="ignore"):
        info_per_bin = p_i * (r_i / r_mean) * np.log2(r_i / r_mean)
        info_per_bin[np.isnan(info_per_bin)] = 0
        info_per_bin[np.isinf(info_per_bin)] = 0

    spatial_info = np.sum(info_per_bin)

    return max(0.0, spatial_info)  # Ensure non-negative




def compute_spatial_decoding_accuracy(
    neural_activity: np.ndarray,
    positions: np.ndarray,
    test_size: float = 0.5,
    n_estimators: int = 20,
    max_depth: int = 3,
    min_samples_leaf: int = 50,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Compute position decoding accuracy from neural activity.

    Parameters
    ----------
    neural_activity : np.ndarray, shape (n_neurons, n_samples)
        Neural activity matrix
    positions : np.ndarray, shape (n_samples, 2)
        True X, Y positions
    test_size : float
        Fraction of data for testing
    n_estimators : int
        Number of trees in random forest
    max_depth : int
        Maximum tree depth
    min_samples_leaf : int
        Minimum samples per leaf
    random_state : int
        Random seed for reproducibility
    logger : logging.Logger, optional
        Logger for debugging

    Returns
    -------
    metrics : dict
        Decoding accuracy metrics:
        - r2_x: R² score for X position
        - r2_y: R² score for Y position
        - r2_avg: Average R² score
        - mse: Mean squared error
    
    Raises
    ------
    ValueError
        If shape mismatch between neural_activity and positions
    
    Notes
    -----
    Forces non-negative R² scores. Uses all CPU cores.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create sample data: 10 neurons, 1000 time points
    >>> neural_data = np.random.rand(10, 1000)
    >>> positions = np.random.rand(1000, 2)
    >>> metrics = compute_spatial_decoding_accuracy(neural_data, positions)
    >>> print(f"Decoding R²: {metrics['r2_avg']:.3f}")  # doctest: +ELLIPSIS
    Decoding R²: ...
    """
    # Validate inputs
    if neural_activity.shape[1] != positions.shape[0]:
        raise ValueError(f"Shape mismatch: neural_activity has {neural_activity.shape[1]} "
                        f"samples but positions has {positions.shape[0]}")
    if positions.shape[1] != 2:
        raise ValueError(f"Positions must be 2D, got shape {positions.shape}")
    check_positive(test_size=test_size, n_estimators=n_estimators, 
                   min_samples_leaf=min_samples_leaf)
    
    if logger:
        logger.info(
            f"Computing spatial decoding with {neural_activity.shape[0]} neurons"
        )

    # Prepare data (transpose for sklearn format)
    X = neural_activity.T  # (n_samples, n_neurons)
    y = positions

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train decoder
    decoder = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    decoder.fit(X_train, y_train)
    y_pred = decoder.predict(X_test)

    # Compute metrics
    r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
    mse = np.mean((y_test - y_pred) ** 2)

    metrics = {
        "r2_x": max(0.0, r2_x),  # Avoid negative R²
        "r2_y": max(0.0, r2_y),
        "r2_avg": max(0.0, (r2_x + r2_y) / 2),
        "mse": mse,
    }

    if logger:
        logger.info(f"Decoding accuracy: R²_avg = {metrics['r2_avg']:.3f}")

    return metrics


def compute_spatial_information(
    neural_activity: Union[np.ndarray, TimeSeries, MultiTimeSeries],
    positions: Union[np.ndarray, TimeSeries, MultiTimeSeries],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Compute mutual information between neural activity and spatial position.

    Parameters
    ----------
    neural_activity : array-like or TimeSeries
        Neural activity data. If np.ndarray, shape (n_neurons, n_samples) or (n_samples,)
    positions : array-like or TimeSeries
        Spatial position data (X, Y). If np.ndarray, shape (n_samples, 2)
    logger : logging.Logger, optional
        Logger for debugging

    Returns
    -------
    metrics : dict
        Spatial information metrics:
        - mi_x: MI with X position
        - mi_y: MI with Y position
        - mi_total: MI with 2D position
    
    Raises
    ------
    ValueError
        If positions is not 2D or shape mismatch
    ImportError
        If required information theory packages are not available
    
    Notes
    -----
    Uses Gaussian-Copula MI (gcmi) estimator.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create sample data: 5 neurons, 500 time points
    >>> neural_data = np.random.rand(5, 500)
    >>> positions = np.random.rand(500, 2)
    >>> mi_metrics = compute_spatial_information(neural_data, positions)
    >>> print(f"MI with position: {mi_metrics['mi_total']:.3f} bits")  # doctest: +ELLIPSIS
    MI with position: ... bits
    """
    # Convert to TimeSeries if needed
    if isinstance(neural_activity, np.ndarray):
        if neural_activity.ndim == 1:
            neural_ts = TimeSeries(neural_activity, discrete=False)
        else:
            # Create MultiTimeSeries from multiple neurons
            neural_ts_list = [
                TimeSeries(neural_activity[i], discrete=False)
                for i in range(neural_activity.shape[0])
            ]
            neural_ts = MultiTimeSeries(neural_ts_list)
    else:
        neural_ts = neural_activity

    if isinstance(positions, np.ndarray):
        if positions.shape[1] != 2:
            raise ValueError("Positions must be 2D (X, Y)")
        x_ts = TimeSeries(positions[:, 0], discrete=False)
        y_ts = TimeSeries(positions[:, 1], discrete=False)
        pos_2d_ts = MultiTimeSeries([x_ts, y_ts])
    else:
        # Assume positions is already properly formatted
        if isinstance(positions, MultiTimeSeries):
            x_ts = positions.data[0]
            y_ts = positions.data[1]
            pos_2d_ts = positions
        else:
            raise ValueError("Positions must be 2D")

    # Compute mutual information
    mi_x = get_sim(neural_ts, x_ts, metric="mi", estimator="gcmi")
    mi_y = get_sim(neural_ts, y_ts, metric="mi", estimator="gcmi")
    mi_total = get_sim(neural_ts, pos_2d_ts, metric="mi", estimator="gcmi")

    metrics = {"mi_x": mi_x, "mi_y": mi_y, "mi_total": mi_total}

    if logger:
        logger.info(f"Spatial MI: X={mi_x:.3f}, Y={mi_y:.3f}, Total={mi_total:.3f}")

    return metrics


def filter_by_speed(
    data: Dict[str, np.ndarray],
    speed_range: Tuple[float, float] = (0.05, float("inf")),
    position_key: str = "positions",
    smooth_window: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Filter data to include only periods of specific movement speeds.

    Parameters
    ----------
    data : dict
        Dictionary with at least 'positions' key containing (n_samples, 2) array
    speed_range : tuple
        (min_speed, max_speed) to include. Default excludes near-stationary periods.
    position_key : str
        Key for position data in dictionary
    smooth_window : int
        Window size for speed smoothing

    Returns
    -------
    filtered_data : dict
        Data dictionary with speed-filtered arrays. Adds 'speed' key with 
        computed speeds.
    
    Raises
    ------
    ValueError
        If speed_range values are invalid
    KeyError
        If position_key not found in data
    
    Notes
    -----
    First sample assigned zero speed.
    
    Examples
    --------
    >>> import numpy as np
    >>> positions = np.random.rand(1000, 2)
    >>> activity = np.random.rand(1000, 10)  # 10 neurons
    >>> data = {\'positions\': positions, \'neural_activity\': activity}
    >>> # Keep only when animal is moving
    >>> filtered = filter_by_speed(data, speed_range=(0.05, np.inf))    """
    # Validate inputs
    if position_key not in data:
        raise KeyError(f"position_key '{position_key}' not found in data")
    positions = data[position_key]
    if speed_range[0] < 0:
        raise ValueError(f"min_speed must be non-negative, got {speed_range[0]}")
    if speed_range[0] > speed_range[1]:
        raise ValueError(f"min_speed ({speed_range[0]}) > max_speed ({speed_range[1]})")

    # Compute speed
    velocity = np.diff(positions, axis=0)
    speed = np.sqrt(np.sum(velocity**2, axis=1))

    # Smooth speed
    if smooth_window > 1:
        speed = ndimage.uniform_filter1d(speed, size=smooth_window)

    # Add zero speed for first sample
    speed = np.concatenate([[0], speed])

    # Create mask
    mask = (speed >= speed_range[0]) & (speed <= speed_range[1])

    # Filter all arrays in data
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            filtered_data[key] = value[mask]
        else:
            filtered_data[key] = value

    filtered_data["speed"] = speed[mask]

    return filtered_data




def analyze_spatial_coding(
    neural_activity: np.ndarray,
    positions: np.ndarray,
    fps: float = 1.0,
    arena_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    bin_size: float = 0.025,
    min_peak_rate: float = 1.0,
    speed_range: Optional[Tuple[float, float]] = (0.05, float("inf")),
    peak_to_mean_ratio: float = 1.5,
    min_field_size: int = 9,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[np.ndarray, List, Dict, float]]:
    """
    Comprehensive spatial coding analysis pipeline.

    Parameters
    ----------
    neural_activity : np.ndarray, shape (n_neurons, n_samples)
        Neural activity matrix
    positions : np.ndarray, shape (n_samples, 2)
        Position data
    fps : float
        Frames per second (sampling frequency)
    arena_bounds : tuple, optional
        Arena boundaries
    bin_size : float
        Spatial bin size
    min_peak_rate : float
        Minimum peak rate for place fields
    speed_range : tuple, optional
        Speed filter range. None to skip filtering.
    peak_to_mean_ratio : float
        Minimum ratio of peak to mean rate in field
    min_field_size : int
        Minimum number of contiguous bins for valid field
    logger : logging.Logger, optional
        Logger for progress

    Returns
    -------
    results : dict
        Comprehensive spatial analysis results:
        - rate_maps: List of rate maps per neuron
        - place_fields: List of place fields per neuron
        - spatial_info: Spatial information per neuron
        - decoding_accuracy: Position decoding metrics
        - spatial_mi: Mutual information metrics
        - summary: Dict with n_place_cells, mean_spatial_info
    
    Raises
    ------
    ValueError
        If shape mismatch or invalid parameters
    
    Notes
    -----
    - Applies speed filtering before analysis if speed_range provided
    - All analyses use the same spatial binning
    - Rate maps are smoothed with sigma=1.5
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create sample data with spatial structure for more realistic example
    >>> # 20 neurons, 3000 frames at 30fps = 100 seconds
    >>> np.random.seed(42)  # For reproducible example
    >>> t = np.linspace(0, 100, 3000)
    >>> # Create circular trajectory
    >>> positions = np.column_stack([
    ...     0.5 + 0.3 * np.cos(0.1 * t) + 0.05 * np.random.randn(3000),
    ...     0.5 + 0.3 * np.sin(0.1 * t) + 0.05 * np.random.randn(3000)
    ... ])
    >>> # Create neural activity with spatial tuning
    >>> neural_data = np.zeros((20, 3000))
    >>> for i in range(20):
    ...     # Each neuron has a preferred location
    ...     pref_x, pref_y = np.random.rand(2)
    ...     distance = np.sqrt((positions[:, 0] - pref_x)**2 + (positions[:, 1] - pref_y)**2)
    ...     neural_data[i] = np.exp(-distance**2 / 0.1) + 0.1 * np.random.randn(3000)
    >>> results = analyze_spatial_coding(neural_data, positions, fps=30.0)
    >>> print(f"Found {results['summary']['n_place_cells']} place cells")  # doctest: +ELLIPSIS
    Found ... place cells
    """
    # Validate inputs
    if neural_activity.shape[1] != positions.shape[0]:
        raise ValueError(f"Shape mismatch: neural_activity has {neural_activity.shape[1]} "
                        f"samples but positions has {positions.shape[0]}")
    if positions.shape[1] != 2:
        raise ValueError(f"Positions must be 2D, got shape {positions.shape}")
    check_positive(fps=fps, bin_size=bin_size, min_peak_rate=min_peak_rate,
                   peak_to_mean_ratio=peak_to_mean_ratio, min_field_size=min_field_size)
    
    if logger:
        logger.info(f"Analyzing spatial coding for {neural_activity.shape[0]} neurons")

    # Speed filtering if requested
    if speed_range is not None:
        data = {
            "positions": positions,
            "neural_activity": neural_activity.T,  # Transpose for filtering
        }
        filtered = filter_by_speed(data, speed_range)
        positions = filtered["positions"]
        neural_activity = filtered["neural_activity"].T

    # Compute occupancy map
    occupancy_map, x_edges, y_edges = compute_occupancy_map(
        positions, fps, arena_bounds, bin_size
    )

    results = {
        "rate_maps": [],
        "place_fields": [],
        "spatial_info": [],
    }

    # Analyze each neuron
    for i in range(neural_activity.shape[0]):
        # Compute rate map
        rate_map = compute_rate_map(
            neural_activity[i], positions, occupancy_map, x_edges, y_edges, fps
        )
        results["rate_maps"].append(rate_map)

        # Extract place fields
        fields = extract_place_fields(
            rate_map,
            min_peak_rate=min_peak_rate,
            min_field_size=min_field_size,
            peak_to_mean_ratio=peak_to_mean_ratio,
        )
        results["place_fields"].append(fields)

        # Spatial information
        si = compute_spatial_information_rate(rate_map, occupancy_map)
        results["spatial_info"].append(si)

    # Population-level analyses
    results["decoding_accuracy"] = compute_spatial_decoding_accuracy(
        neural_activity, positions, logger=logger
    )

    results["spatial_mi"] = compute_spatial_information(
        neural_activity, positions, logger=logger
    )

    # Summary statistics
    results["summary"] = {
        "n_place_cells": sum(len(pf) > 0 for pf in results["place_fields"]),
        "mean_spatial_info": np.mean(results["spatial_info"]),
    }

    if logger:
        logger.info(
            f"Found {results['summary']['n_place_cells']} place cells"
        )

    return results


def compute_spatial_metrics(
    neural_activity: np.ndarray,
    positions: np.ndarray,
    metrics: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Union[float, Dict]]:
    """
    Compute selected spatial metrics.

    Parameters
    ----------
    neural_activity : np.ndarray, shape (n_neurons, n_samples)
        Neural activity data
    positions : np.ndarray, shape (n_samples, 2)
        Position data
    metrics : list of str, optional
        Metrics to compute. If None, computes all.
        Options: 'decoding', 'information', 'place_fields'
    **kwargs
        Additional arguments passed to analysis functions
        (e.g., fps, logger, test_size, bin_size)

    Returns
    -------
    results : dict
        Computed metrics based on requested analyses
    
    Raises
    ------
    ValueError
        If invalid metric name provided
    
    Examples
    --------
    >>> # Compute only decoding accuracy
    >>> import numpy as np
    >>> neural_data = np.random.rand(15, 2000)  # 15 neurons, 2000 samples
    >>> positions = np.random.rand(2000, 2)
    >>> results = compute_spatial_metrics(neural_data, positions, 
    ...                                  metrics=['decoding'])
    >>> # Compute all metrics
    >>> import numpy as np
    >>> neural_data = np.random.rand(15, 2000)
    >>> positions = np.random.rand(2000, 2)
    >>> results = compute_spatial_metrics(neural_data, positions)    """
    if metrics is None:
        metrics = ["decoding", "information", "place_fields"]
    
    # Validate metrics
    valid_metrics = {"decoding", "information", "place_fields"}
    invalid = set(metrics) - valid_metrics
    if invalid:
        raise ValueError(f"Invalid metrics: {invalid}. Valid options: {valid_metrics}")

    results = {}

    if "decoding" in metrics:
        results["decoding"] = compute_spatial_decoding_accuracy(
            neural_activity, positions, **kwargs
        )

    if "information" in metrics:
        results["information"] = compute_spatial_information(
            neural_activity, positions, **kwargs
        )

    if "place_fields" in metrics:
        analysis = analyze_spatial_coding(neural_activity, positions, **kwargs)
        results["place_fields"] = analysis["place_fields"]
        results["n_place_cells"] = analysis["summary"]["n_place_cells"]

    return results
