import numpy as np


def generate_pseudo_calcium_signal(
    duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std
):
    """Generate a pseudo-calcium imaging signal with noise.
    
    Simulates calcium imaging data by generating random calcium transient
    events with exponential decay and adding Gaussian noise. Events are
    generated using a Poisson process.

    Parameters
    ----------
    duration : float
        Total duration of the signal in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    event_rate : float
        Average rate of calcium events per second.
    amplitude_range : tuple of float
        Tuple of (min, max) for the amplitude of calcium events.
    decay_time : float
        Time constant for the exponential decay of calcium events in seconds.
    noise_std : float
        Standard deviation of the Gaussian noise to be added.

    Returns
    -------
    numpy.ndarray
        1D array representing the pseudo-calcium signal.
        
    Notes
    -----
    The function generates calcium transients as instantaneous rises followed
    by exponential decays. The number of events follows a Poisson distribution
    with mean `event_rate * duration`. Event times are uniformly distributed
    across the signal duration.
    
    Examples
    --------
    >>> signal = generate_pseudo_calcium_signal(
    ...     duration=10.0,      # 10 seconds
    ...     sampling_rate=30.0, # 30 Hz
    ...     event_rate=2.0,     # 2 events per second on average
    ...     amplitude_range=(0.5, 2.0),
    ...     decay_time=1.0,     # 1 second decay
    ...     noise_std=0.1
    ... )
    >>> signal.shape
    (300,)
    
    See Also
    --------
    ~driada.utils.neural.generate_pseudo_calcium_multisignal :
        Generate multiple calcium signals.
    """
    # Calculate number of samples
    num_samples = int(duration * sampling_rate)

    # Initialize the signal with zeros
    signal = np.zeros(num_samples)

    # Generate calcium events
    num_events = np.random.poisson(event_rate * duration)
    event_times = np.random.uniform(0, duration, num_events)
    event_amplitudes = np.random.uniform(
        amplitude_range[0], amplitude_range[1], num_events
    )

    # Add calcium events to the signal
    for t, a in zip(event_times, event_amplitudes):
        event_index = int(t * sampling_rate)
        # Ensure event_index is within bounds
        if event_index < num_samples:
            decay = np.exp(
                -np.arange(num_samples - event_index) / (decay_time * sampling_rate)
            )
            signal[event_index:] += a * decay

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, num_samples)
    signal += noise

    return signal


def generate_pseudo_calcium_multisignal(
    n, duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std
):
    """Generate multiple pseudo-calcium fluorescence signals.
    
    Creates a collection of synthetic calcium signals that simulate the 
    fluorescence traces typically observed in calcium imaging experiments.
    Each signal is generated independently with the same parameters.
    
    Parameters
    ----------
    n : int
        Number of calcium signals to generate.
    duration : float
        Duration of each signal in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    event_rate : float
        Average rate of calcium events (spikes) per second.
    amplitude_range : tuple of float
        Range (min, max) for event amplitudes.
    decay_time : float
        Exponential decay time constant in seconds.
    noise_std : float
        Standard deviation of Gaussian noise to add.
        
    Returns
    -------
    numpy.ndarray
        Array of shape (n, n_timepoints) containing the generated calcium
        signals. Each row is one neuron's calcium trace.
        
    See Also
    --------
    ~driada.utils.neural.generate_pseudo_calcium_signal :
        Generates a single calcium signal.
    
    Examples
    --------
    >>> # Generate 10 neurons with 30 seconds of data at 30Hz
    >>> signals = generate_pseudo_calcium_multisignal(
    ...     n=10, duration=30, sampling_rate=30, 
    ...     event_rate=0.5, amplitude_range=(0.5, 2.0),
    ...     decay_time=1.0, noise_std=0.1
    ... )
    >>> signals.shape
    (10, 900)    """
    sigs = []
    for i in range(n):
        sig = generate_pseudo_calcium_signal(
            duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std
        )
        sigs.append(sig)

    return np.vstack(sigs)
