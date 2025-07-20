"""
Core utilities for synthetic data generation.

This module contains the fundamental utilities used across all synthetic data
generation functions, including firing rate validation and calcium signal generation.
"""

import numpy as np
import warnings
from ..exp_base import *
from ...information.info_base import TimeSeries, MultiTimeSeries, aggregate_multiple_ts


def validate_peak_rate(peak_rate, context=""):
    """
    Validate that peak firing rate is within physiologically realistic range.
    
    Parameters
    ----------
    peak_rate : float
        Peak firing rate in Hz.
    context : str, optional
        Context string for more informative warning message.
        
    Notes
    -----
    Typical firing rates for neurons:
    - Cortical pyramidal cells: 0.1-2 Hz (sparse firing)
    - Hippocampal place cells: 0.5-5 Hz (with brief peaks up to 20 Hz)
    - Fast-spiking interneurons: 5-50 Hz
    
    For calcium imaging with GCaMP indicators:
    - Decay time ~1-2 seconds limits temporal resolution
    - Firing rates >2 Hz can cause signal saturation
    - Realistic modeling should use rates in 0.1-2 Hz range
    """
    if peak_rate > 2.0:
        warning_msg = (
            f"peak_rate={peak_rate:.1f} Hz exceeds recommended maximum of 2.0 Hz "
            f"for calcium imaging. "
            f"High firing rates can cause calcium signal saturation due to slow "
            f"indicator dynamics (decay time ~2s). Consider using peak_rate <= 2.0 Hz "
            f"for more realistic calcium traces."
        )
        if context:
            warning_msg = f"{context}: {warning_msg}"
        warnings.warn(warning_msg, UserWarning, stacklevel=2)


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


def generate_pseudo_calcium_multisignal(n,
                                        events=None,
                                        duration=600,
                                        sampling_rate=20,
                                        event_rate=0.2,
                                        amplitude_range=(0.5,2),
                                        decay_time=2,
                                        noise_std=0.1):
    """
    Generate multiple pseudo calcium signals.
    
    Parameters
    ----------
    n : int
        Number of neurons.
    events : ndarray, optional
        Event array (n_neurons x n_timepoints).
    duration : float
        Duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    event_rate : float
        Average rate of calcium events per second.
    amplitude_range : tuple
        (min, max) for the amplitude of calcium events.
    decay_time : float
        Time constant for the decay of calcium events in seconds.
    noise_std : float
        Standard deviation of the Gaussian noise.
        
    Returns
    -------
    ndarray
        Calcium signals (n_neurons x n_timepoints).
    """
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