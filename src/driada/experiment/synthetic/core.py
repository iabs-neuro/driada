"""
Core utilities for synthetic data generation.

This module contains the fundamental utilities used across all synthetic data
generation functions, including firing rate validation and calcium signal generation.
"""

import numpy as np
import warnings


def validate_peak_rate(peak_rate, context=""):
    """
    Validate that peak firing rate is within physiologically realistic range.

    Parameters
    ----------
    peak_rate : float
        Peak firing rate in Hz. Must be non-negative and finite.
    context : str, optional
        Context string for more informative warning message.
        
    Returns
    -------
    None
        This function only validates and warns; it does not return a value.

    Raises
    ------
    ValueError
        If peak_rate is negative, NaN, or infinite.
    TypeError
        If peak_rate is not numeric.

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
    
    Examples
    --------
    >>> validate_peak_rate(1.5)  # No warning, physiologically realistic
    
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     validate_peak_rate(5.0, context="Place cell simulation")
    ...     len(w) > 0 and "exceeds recommended maximum" in str(w[0].message)
    True
    
    >>> validate_peak_rate(-1.0)
    Traceback (most recent call last):
        ...
    ValueError: peak_rate must be non-negative, got -1.0
    """
    # Validate input
    try:
        peak_rate = float(peak_rate)
    except (TypeError, ValueError):
        raise TypeError(f"peak_rate must be numeric, got {type(peak_rate).__name__}")
    
    if np.isnan(peak_rate):
        raise ValueError("peak_rate cannot be NaN")
    if np.isinf(peak_rate):
        raise ValueError("peak_rate cannot be infinite")
    if peak_rate < 0:
        raise ValueError(f"peak_rate must be non-negative, got {peak_rate}")
    
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


def generate_pseudo_calcium_signal(
    events=None,
    duration=600,
    sampling_rate=20.0,
    event_rate=0.2,
    amplitude_range=(0.5, 2),
    decay_time=2,
    noise_std=0.1,
):
    """Generate a pseudo-calcium imaging signal with noise.
    
    Creates a synthetic calcium fluorescence signal that mimics GCaMP-like
    dynamics with exponential decay, random event amplitudes, and Gaussian noise.
    
    Parameters
    ----------
    events : ndarray or None, optional
        Binary array indicating event occurrences at each time point. 
        If None, events are generated randomly using a Poisson process.
        When provided, indices directly correspond to sample indices (not time in seconds).
    duration : float, default=600
        Total duration of the signal in seconds. Only used if events is None.
        Must be positive.
    sampling_rate : float, default=20.0
        Sampling rate in Hz. Must be positive.
    event_rate : float, default=0.2
        Average rate of calcium events per second. Only used if events is None.
        Must be non-negative.
    amplitude_range : tuple of float, default=(0.5, 2)
        (min, max) range for random calcium event amplitudes.
        Must have min <= max.
    decay_time : float, default=2
        Time constant for exponential decay of calcium events in seconds.
        Typical GCaMP indicators have decay times of 1-2 seconds.
        Must be positive.
    noise_std : float, default=0.1
        Standard deviation of additive Gaussian noise. Must be non-negative.
        
    Returns
    -------
    ndarray
        1D array representing the pseudo-calcium signal with shape (n_samples,).
        
    Raises
    ------
    ValueError
        If duration, sampling_rate, or decay_time <= 0.
        If event_rate or noise_std < 0.
        If amplitude_range[0] > amplitude_range[1].
        
    Notes
    -----
    The calcium signal is modeled as a sum of exponentially decaying transients
    triggered at event times, plus additive Gaussian noise. This approximates
    the dynamics of genetically encoded calcium indicators like GCaMP6.
    
    When events is None: Event times are drawn uniformly in [0, duration) seconds.
    When events is provided: Non-zero indices are treated as event sample indices.
    
    Multiple overlapping events accumulate additively without saturation modeling.
    
    Examples
    --------
    >>> # Generate random calcium signal
    >>> signal = generate_pseudo_calcium_signal(duration=100, event_rate=0.5)
    >>> signal.shape
    (2000,)
    
    >>> # Generate from specific spike times
    >>> spikes = np.zeros(1000)
    >>> spikes[[100, 200, 300]] = 1  # 3 spike events at sample indices
    >>> signal = generate_pseudo_calcium_signal(events=spikes)    """

    # Input validation
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    if decay_time <= 0:
        raise ValueError(f"decay_time must be positive, got {decay_time}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if len(amplitude_range) != 2 or amplitude_range[0] > amplitude_range[1]:
        raise ValueError(f"amplitude_range must be (min, max) with min <= max, got {amplitude_range}")
    
    if events is None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if event_rate < 0:
            raise ValueError(f"event_rate must be non-negative, got {event_rate}")
            
        # Calculate number of samples
        num_samples = int(duration * sampling_rate)

        # Generate calcium events
        num_events = np.random.poisson(event_rate * duration)
        event_times = np.random.uniform(0, duration, num_events)
        event_amplitudes = np.random.uniform(
            amplitude_range[0], amplitude_range[1], num_events
        )

    else:
        num_samples = len(events)
        event_times = np.where(events > 0)[0]
        # Use amplitude_range to modulate event amplitudes instead of using binary values
        if len(event_times) > 0:
            event_amplitudes = np.random.uniform(
                amplitude_range[0], amplitude_range[1], len(event_times)
            )
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

        decay = np.exp(
            -np.arange(num_samples - event_index) / (decay_time * sampling_rate)
        )
        signal[event_index:] += a * decay

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, num_samples)
    signal += noise

    return signal


def generate_pseudo_calcium_multisignal(
    n,
    events=None,
    duration=600,
    sampling_rate=20,
    event_rate=0.2,
    amplitude_range=(0.5, 2),
    decay_time=2,
    noise_std=0.1,
):
    """
    Generate multiple pseudo calcium signals.

    Parameters
    ----------
    n : int
        Number of neurons. Must be non-negative.
    events : ndarray, optional
        Event array of shape (n_neurons, n_timepoints). If provided, must have
        n_neurons == n. Each row corresponds to one neuron's event indices.
    duration : float, default=600
        Duration in seconds. Only used if events is None.
    sampling_rate : float, default=20
        Sampling rate in Hz.
    event_rate : float, default=0.2
        Average rate of calcium events per second. Only used if events is None.
    amplitude_range : tuple, default=(0.5, 2)
        (min, max) for the amplitude of calcium events.
    decay_time : float, default=2
        Time constant for the decay of calcium events in seconds.
    noise_std : float, default=0.1
        Standard deviation of the Gaussian noise.

    Returns
    -------
    ndarray
        Calcium signals of shape (n_neurons, n_timepoints).
        
    Raises
    ------
    ValueError
        If n < 0.
        If events is provided with wrong shape.
        
    Notes
    -----
    This is a convenience wrapper that calls generate_pseudo_calcium_signal
    for each neuron independently. Each neuron gets independent random
    events (if events=None) and independent noise.
    
    Examples
    --------
    >>> # Generate 5 neurons with random events
    >>> signals = generate_pseudo_calcium_multisignal(5, duration=100)
    >>> signals.shape
    (5, 2000)
    
    >>> # Generate with specific events per neuron
    >>> events = np.random.binomial(1, 0.01, size=(3, 1000))
    >>> signals = generate_pseudo_calcium_multisignal(3, events=events)    """
    # Input validation
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    
    if events is not None:
        events = np.asarray(events)
        if events.ndim != 2:
            raise ValueError(f"events must be 2D array, got shape {events.shape}")
        if events.shape[0] != n:
            raise ValueError(f"events first dimension {events.shape[0]} must match n={n}")
    
    sigs = []
    for i in range(n):
        local_events = None
        if events is not None:
            local_events = events[i, :]

        sig = generate_pseudo_calcium_signal(
            events=local_events,
            duration=duration,
            sampling_rate=sampling_rate,
            event_rate=event_rate,
            amplitude_range=amplitude_range,
            decay_time=decay_time,
            noise_std=noise_std,
        )
        sigs.append(sig)

    return np.vstack(sigs)
