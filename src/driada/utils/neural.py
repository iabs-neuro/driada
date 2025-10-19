import numpy as np


def _double_exponential_kernel(t, amplitude, rise_time, decay_time):
    """Generate double-exponential calcium transient kernel.

    Models realistic calcium indicator dynamics with separate rise and decay phases.

    Parameters
    ----------
    t : ndarray
        Time array in samples.
    amplitude : float
        Peak amplitude of the transient.
    rise_time : float
        Rise time constant in samples.
    decay_time : float
        Decay time constant in samples.

    Returns
    -------
    ndarray
        Calcium transient waveform.
    """
    # Double exponential: (1 - exp(-t/rise)) * exp(-t/decay)
    kernel = (1 - np.exp(-t / rise_time)) * np.exp(-t / decay_time)
    # Normalize to peak = 1, then scale by amplitude
    kernel = kernel / np.max(kernel) * amplitude
    return kernel


def _exponential_kernel(t, amplitude, decay_time):
    """Generate simple exponential decay kernel.

    Models instantaneous rise followed by exponential decay.

    Parameters
    ----------
    t : ndarray
        Time array in samples.
    amplitude : float
        Peak amplitude of the transient.
    decay_time : float
        Decay time constant in samples.

    Returns
    -------
    ndarray
        Calcium transient waveform.
    """
    return amplitude * np.exp(-t / decay_time)


def _step_kernel(t, amplitude, decay_time, sharpness=5.0):
    """Generate non-physiological step-like kernel for control.

    Models unrealistic calcium dynamics with sharp edges.
    Useful as negative control for testing model assumptions.

    Parameters
    ----------
    t : ndarray
        Time array in samples.
    amplitude : float
        Plateau amplitude.
    decay_time : float
        Duration of plateau in samples.
    sharpness : float, optional
        Controls edge sharpness (higher = sharper). Default 5.0.

    Returns
    -------
    ndarray
        Step-like waveform.
    """
    # Sigmoid rise
    rise = 1 / (1 + np.exp(-sharpness * (t / decay_time - 0.1)))
    # Sigmoid fall
    fall = 1 / (1 + np.exp(sharpness * (t / decay_time - 0.9)))
    return amplitude * rise * fall


def generate_pseudo_calcium_signal(
    duration,
    sampling_rate,
    event_rate,
    amplitude_range,
    decay_time,
    noise_std,
    rise_time=0.25,
    kernel='double_exponential'
):
    """Generate a pseudo-calcium imaging signal with configurable kernels.

    Simulates calcium imaging data by generating random calcium transient
    events using various kernel shapes and adding Gaussian noise. Events are
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
        Time constant for decay in seconds (interpretation depends on kernel).
    noise_std : float
        Standard deviation of the Gaussian noise to be added.
    rise_time : float, optional
        Rise time constant in seconds (only used for 'double_exponential').
        Default is 0.25s (typical for GCaMP indicators).
    kernel : {'double_exponential', 'exponential', 'step'}, optional
        Type of calcium transient kernel:

        - 'double_exponential' (default): Physiologically realistic kernel with
          separate rise and decay phases. Matches Neuron model assumptions.
          Form: (1 - exp(-t/rise_time)) * exp(-t/decay_time)
        - 'exponential': Simple exponential decay with instantaneous rise.
          Legacy behavior, model mismatch with Neuron class.
          Form: amplitude * exp(-t/decay_time)
        - 'step': Non-physiological step-like waveform with sharp edges.
          Useful as negative control for testing model assumptions.

    Returns
    -------
    numpy.ndarray
        1D array representing the pseudo-calcium signal.

    Notes
    -----
    The number of events follows a Poisson distribution with mean
    `event_rate * duration`. Event times are uniformly distributed
    across the signal duration. Events can overlap, producing realistic
    summation of calcium transients.

    The default 'double_exponential' kernel matches the Neuron class
    model and produces physiologically realistic calcium dynamics for
    GCaMP-like indicators.

    Examples
    --------
    >>> # Generate realistic calcium signal (default)
    >>> signal = generate_pseudo_calcium_signal(
    ...     duration=10.0,
    ...     sampling_rate=20.0,
    ...     event_rate=2.0,
    ...     amplitude_range=(0.5, 2.0),
    ...     decay_time=1.0,
    ...     noise_std=0.1,
    ...     rise_time=0.25,
    ...     kernel='double_exponential'
    ... )

    >>> # Generate with simple exponential (legacy)
    >>> signal_exp = generate_pseudo_calcium_signal(
    ...     duration=10.0,
    ...     sampling_rate=20.0,
    ...     event_rate=2.0,
    ...     amplitude_range=(0.5, 2.0),
    ...     decay_time=1.0,
    ...     noise_std=0.1,
    ...     kernel='exponential'
    ... )

    >>> # Generate non-physiological control
    >>> signal_step = generate_pseudo_calcium_signal(
    ...     duration=10.0,
    ...     sampling_rate=20.0,
    ...     event_rate=2.0,
    ...     amplitude_range=(0.5, 2.0),
    ...     decay_time=1.0,
    ...     noise_std=0.1,
    ...     kernel='step'
    ... )

    See Also
    --------
    ~driada.utils.neural.generate_pseudo_calcium_multisignal :
        Generate multiple calcium signals.
    """
    # Validate kernel type
    valid_kernels = ['double_exponential', 'exponential', 'step']
    if kernel not in valid_kernels:
        raise ValueError(f"kernel must be one of {valid_kernels}, got '{kernel}'")

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

    # Convert time constants to samples
    rise_time_samples = rise_time * sampling_rate
    decay_time_samples = decay_time * sampling_rate

    # Add calcium events to the signal
    for t, a in zip(event_times, event_amplitudes):
        event_index = int(t * sampling_rate)
        # Ensure event_index is within bounds
        if event_index < num_samples:
            # Time array for this event
            t_array = np.arange(num_samples - event_index)

            # Generate kernel based on type
            if kernel == 'double_exponential':
                transient = _double_exponential_kernel(
                    t_array, a, rise_time_samples, decay_time_samples
                )
            elif kernel == 'exponential':
                transient = _exponential_kernel(
                    t_array, a, decay_time_samples
                )
            elif kernel == 'step':
                transient = _step_kernel(
                    t_array, a, decay_time_samples
                )

            signal[event_index:] += transient

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, num_samples)
    signal += noise

    return signal


def generate_pseudo_calcium_multisignal(
    n, duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std,
    rise_time=0.25, kernel='double_exponential'
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
            duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std,
            rise_time=rise_time, kernel=kernel
        )
        sigs.append(sig)

    return np.vstack(sigs)
