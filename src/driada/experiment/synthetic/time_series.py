"""
Time series utilities for synthetic data generation.

This module contains functions for generating various types of time series
used in synthetic neural data, including binary series, fractional Brownian motion,
and signal processing utilities.
"""

import numpy as np
from fbm import FBM
import itertools
from itertools import groupby
from ...utils.data import check_positive, check_nonnegative


def generate_binary_time_series(length, avg_islands, avg_duration, seed=None):
    """
    Generate a binary time series with islands of 1s.

    Parameters
    ----------
    length : int
        Length of the time series. Must be positive.
    avg_islands : int
        Average number of islands (continuous stretches of 1s). Must be non-negative.
    avg_duration : int
        Average duration of each island in time points. Must be positive.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (length,)
        Binary time series with 0s and 1s.
        
    Raises
    ------
    ValueError
        If length or avg_duration are not positive, or avg_islands is negative.
        
    Notes
    -----
    Uses exponential distribution for gaps between islands and normal
    distribution (std = avg_duration/3) for island durations. Automatically
    adjusts number of islands if they don't fit in the requested length.
    Starting state (0 or 1) is randomly chosen.    """
    # Input validation
    check_positive(length=length, avg_duration=avg_duration)
    check_nonnegative(avg_islands=avg_islands)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        
    series = np.zeros(length, dtype=int)

    # Calculate expected total active time and inactive time
    total_active_time = avg_islands * avg_duration
    total_inactive_time = length - total_active_time

    # If we can't fit the requested islands, adjust
    if total_active_time > length:
        # Reduce number of islands to fit
        avg_islands = max(1, length // avg_duration)
        total_active_time = avg_islands * avg_duration
        total_inactive_time = length - total_active_time

    # Calculate average gap between islands
    # We have avg_islands active periods and avg_islands+1 potential gaps
    # But often starts with active, so effectively avg_islands gaps
    avg_gap = total_inactive_time / avg_islands if avg_islands > 0 else length

    position = 0
    island_count = 0

    # Randomly decide if we start with on or off
    current_state = np.random.randint(0, 2)

    while position < length and island_count < avg_islands:
        if current_state == 0:
            # Off state: use exponential distribution around average gap
            duration = max(1, int(np.random.exponential(avg_gap)))
        else:
            # On state: use normal distribution around average duration
            duration = max(1, int(np.random.normal(avg_duration, avg_duration / 3)))
            island_count += 1

        # Ensure we don't go past the series length
        duration = min(duration, length - position)

        # Fill the series with the current state
        series[position : position + duration] = current_state

        # Switch state
        current_state = 1 - current_state
        position += duration

    # Fill any remaining time with zeros
    if position < length:
        series[position:] = 0

    return series


def apply_poisson_to_binary_series(binary_series, rate_0, rate_1, seed=None):
    """
    Apply Poisson sampling to a binary series based on its state.

    Parameters
    ----------
    binary_series : ndarray
        Binary time series (0s and 1s). Must contain only 0s and 1s.
    rate_0 : float
        Poisson rate for 0 state. Must be non-negative.
    rate_1 : float
        Poisson rate for 1 state. Must be non-negative.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (len(binary_series),)
        Poisson-sampled integer series.
        
    Raises
    ------
    ValueError
        If binary_series contains values other than 0 and 1.
        If rate_0 or rate_1 are negative.
        
    Notes
    -----
    Uses itertools.groupby to efficiently process runs of identical values.
    Each run is sampled from Poisson distribution with the appropriate rate.    """
    # Input validation
    binary_series = np.asarray(binary_series)
    if not np.all(np.isin(binary_series, [0, 1])):
        raise ValueError("binary_series must contain only 0s and 1s")
    check_nonnegative(rate_0=rate_0, rate_1=rate_1)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        
    length = len(binary_series)
    poisson_series = np.zeros(length, dtype=int)

    current_pos = 0
    for value, group in itertools.groupby(binary_series):
        run_length = len(list(group))
        if value == 0:
            poisson_series[current_pos : current_pos + run_length] = np.random.poisson(
                rate_0, run_length
            )
        else:
            poisson_series[current_pos : current_pos + run_length] = np.random.poisson(
                rate_1, run_length
            )
        current_pos += run_length

    return poisson_series


def delete_one_islands(binary_ts, probability, seed=None):
    """
    Delete islands of 1s from a binary time series with given probability.

    Parameters
    ----------
    binary_ts : ndarray
        Binary time series. Must contain only 0s and 1s.
    probability : float
        Probability of deleting each island. Must be in [0, 1].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape binary_ts.shape
        Modified binary time series (copy).
        
    Raises
    ------
    ValueError
        If binary_ts contains values other than 0 and 1.
        If probability is not in [0, 1].
        
    Notes
    -----
    Creates a copy of the input array. Each island of 1s has an independent
    probability of being deleted (set to 0s).    """
    # Input validation
    if not np.all(np.isin(binary_ts, [0, 1])):
        raise ValueError("binary_ts must be binary (0s and 1s)")
    if not 0 <= probability <= 1:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
        
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Create a copy of the input array
    result = binary_ts.copy()

    # Identify islands of 1s using groupby
    start = 0
    for key, group in groupby(binary_ts):
        length = sum(1 for _ in group)  # Count elements in the group
        if key == 1 and np.random.random() < probability:
            result[start : start + length] = 0
        start += length

    return result


def generate_fbm_time_series(length, hurst, seed=None, roll_shift=None):
    """
    Generate fractional Brownian motion time series.

    Parameters
    ----------
    length : int
        Length of the series. Must be positive.
    hurst : float
        Hurst parameter. Must be in (0, 1). 0.5 = standard Brownian motion.
    seed : int, optional
        Random seed for reproducibility.
    roll_shift : int, optional
        Circular shift to apply for breaking correlations.

    Returns
    -------
    ndarray of shape (length,)
        FBM time series.
        
    Raises
    ------
    ValueError
        If length is not positive.
        If hurst is not in (0, 1).
        
    Notes
    -----
    Uses Davies-Harte method for efficient FBM generation. The FBM library
    is initialized with n=length-1 to generate exactly 'length' points.    """
    # Input validation
    check_positive(length=length)
    if not 0 < hurst < 1:
        raise ValueError(f"hurst must be in (0, 1), got {hurst}")
        
    if seed is not None:
        np.random.seed(seed)

    f = FBM(n=length - 1, hurst=hurst, length=1.0, method="daviesharte")
    fbm_series = f.fbm()

    # Apply circular shift to break correlation with spatial trajectory
    if roll_shift is not None:
        fbm_series = np.roll(fbm_series, roll_shift)

    return fbm_series


def select_signal_roi(values, seed=None, target_fraction=0.15):
    """
    Select a region of interest (ROI) from signal values.

    Parameters
    ----------
    values : ndarray
        Signal values. Must be non-empty.
    seed : int, optional
        Random seed. If None, uses current random state.
    target_fraction : float, optional
        Fraction of data to include in ROI. Must be in (0, 1]. Default is 0.15.

    Returns
    -------
    tuple of float
        (center, lower_border, upper_border) of the ROI.
        
    Raises
    ------
    ValueError
        If values is empty.
        If target_fraction is not in (0, 1].
        
    Notes
    -----
    Selects a random window containing target_fraction of sorted values.
    Adds epsilon=1e-10 to boundaries to avoid numerical edge cases.    """
    # Input validation
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("values cannot be empty")
    if not 0 < target_fraction <= 1:
        raise ValueError(f"target_fraction must be in (0, 1], got {target_fraction}")
        
    if seed is not None:
        np.random.seed(seed)

    # Sort values to find percentiles
    sorted_values = np.sort(values)
    n = len(sorted_values)

    # Find window size that captures target_fraction of data
    window_size = int(target_fraction * n)

    # Choose a random starting position for the window
    # Ensure we don't go out of bounds
    max_start = n - window_size
    start_idx = np.random.randint(0, max_start + 1)
    end_idx = start_idx + window_size

    # Get the boundaries
    lower_border = sorted_values[start_idx]
    upper_border = sorted_values[end_idx - 1]
    center = (lower_border + upper_border) / 2

    # Add small epsilon to avoid boundary issues
    epsilon = 1e-10
    lower_border -= epsilon
    upper_border += epsilon

    return center, lower_border, upper_border


def discretize_via_roi(continuous_signal, seed=None):
    """
    Discretize a continuous signal based on ROI selection.

    Parameters
    ----------
    continuous_signal : ndarray
        Continuous signal to discretize. Must be non-empty.
    seed : int, optional
        Random seed for ROI selection. If None, uses current random state.

    Returns
    -------
    ndarray of shape continuous_signal.shape
        Binary discretized signal (0s and 1s).
        
    Raises
    ------
    ValueError
        If continuous_signal is empty.
        
    Notes
    -----
    Uses select_signal_roi with default target_fraction=0.15. Returns 1 where
    signal values fall within the selected ROI boundaries (inclusive).    """
    # Input validation
    continuous_signal = np.asarray(continuous_signal)
    if continuous_signal.size == 0:
        raise ValueError("continuous_signal cannot be empty")
        
    # Get ROI boundaries
    _, lower, upper = select_signal_roi(continuous_signal, seed=seed)

    # Create binary signal based on ROI
    binary_signal = (
        (continuous_signal >= lower) & (continuous_signal <= upper)
    ).astype(int)

    return binary_signal
