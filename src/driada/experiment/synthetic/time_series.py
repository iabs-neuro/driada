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
    max_iterations = 1000  # Prevent infinite loops
    iterations = 0
    while actual_islands != avg_islands and iterations < max_iterations:
        if actual_islands < avg_islands:
            # If we have too few islands, turn on a random '0' to create a new island
            zero_positions = np.where(series == 0)[0]
            if len(zero_positions) > 0:
                turn_on = np.random.choice(zero_positions)
                series[turn_on] = 1
                # Recalculate actual islands since changing one bit might not create a new island
                new_islands = sum(1 for value, group in itertools.groupby(series) if value == 1)
                if new_islands == actual_islands:
                    # No change, break to avoid infinite loop
                    break
                actual_islands = new_islands
            else:
                # No zeros left, can't add more islands
                break
        else:
            # If we have too many islands, turn off a random '1' to merge islands
            one_positions = np.where(series == 1)[0]
            if len(one_positions) > 1:
                turn_off = np.random.choice(one_positions)
                series[turn_off] = 0
                # Recalculate actual islands since changing one bit might not merge islands
                new_islands = sum(1 for value, group in itertools.groupby(series) if value == 1)
                if new_islands == actual_islands:
                    # No change, break to avoid infinite loop
                    break
                actual_islands = new_islands
            else:
                # Not enough ones, can't reduce islands
                break
        iterations += 1

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
            poisson_series[current_pos:current_pos + run_length] = np.random.poisson(rate_0, run_length)
        else:
            poisson_series[current_pos:current_pos + run_length] = np.random.poisson(rate_1, run_length)
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
            result[start:start + length] = 0
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

    f = FBM(n=length-1, hurst=hurst, length=1.0, method='daviesharte')
    fbm_series = f.fbm()
    
    # Apply circular shift to break correlation with spatial trajectory
    if roll_shift is not None:
        fbm_series = np.roll(fbm_series, roll_shift)

    return fbm_series


def select_signal_roi(values, seed=42):
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
    mean = np.mean(values)
    std = np.std(values)

    np.random.seed(seed)
    # Select random location within mean ± 2*std
    loc = np.random.uniform(mean - 1.5 * std, mean + 1.5 * std)

    # Define borders
    lower_border = loc - 0.5 * std
    upper_border = loc + 0.5 * std

    return loc, lower_border, upper_border


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
    binary_signal = ((continuous_signal >= lower) & (continuous_signal <= upper)).astype(int)
    
    return binary_signal