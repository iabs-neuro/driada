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


def generate_binary_time_series(length, avg_islands, avg_duration):
    """
    Generate a binary time series with islands of 1s.

    Parameters
    ----------
    length : int
        Length of the time series.
    avg_islands : int
        Average number of islands (continuous stretches of 1s).
    avg_duration : int
        Average duration of each island.

    Returns
    -------
    ndarray
        Binary time series.
    """
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


def apply_poisson_to_binary_series(binary_series, rate_0, rate_1):
    """
    Apply Poisson sampling to a binary series based on its state.

    Parameters
    ----------
    binary_series : ndarray
        Binary time series (0s and 1s).
    rate_0 : float
        Poisson rate for 0 state.
    rate_1 : float
        Poisson rate for 1 state.

    Returns
    -------
    ndarray
        Poisson-sampled series.
    """
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


def delete_one_islands(binary_ts, probability):
    """
    Delete islands of 1s from a binary time series with given probability.

    Parameters
    ----------
    binary_ts : ndarray
        Binary time series.
    probability : float
        Probability of deleting each island.

    Returns
    -------
    ndarray
        Modified binary time series.
    """
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
            result[start : start + length] = 0
        start += length

    return result


def generate_fbm_time_series(length, hurst, seed=None, roll_shift=None):
    """
    Generate fractional Brownian motion time series.

    Parameters
    ----------
    length : int
        Length of the series.
    hurst : float
        Hurst parameter (0.5 = standard Brownian motion).
    seed : int, optional
        Random seed.
    roll_shift : int, optional
        Circular shift to apply.

    Returns
    -------
    ndarray
        FBM time series.
    """
    if seed is not None:
        np.random.seed(seed)

    f = FBM(n=length - 1, hurst=hurst, length=1.0, method="daviesharte")
    fbm_series = f.fbm()

    # Apply circular shift to break correlation with spatial trajectory
    if roll_shift is not None:
        fbm_series = np.roll(fbm_series, roll_shift)

    return fbm_series


def select_signal_roi(values, seed=42, target_fraction=0.15):
    """
    Select a region of interest (ROI) from signal values.

    Parameters
    ----------
    values : ndarray
        Signal values.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (center, lower_border, upper_border) of the ROI.
    """
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
        Continuous signal to discretize.
    seed : int, optional
        Random seed for ROI selection.

    Returns
    -------
    ndarray
        Binary discretized signal.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get ROI boundaries
    _, lower, upper = select_signal_roi(continuous_signal, seed=seed)

    # Create binary signal based on ROI
    binary_signal = (
        (continuous_signal >= lower) & (continuous_signal <= upper)
    ).astype(int)

    return binary_signal
