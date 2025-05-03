import numpy as np


def generate_pseudo_calcium_signal(duration,
                                   sampling_rate,
                                   event_rate,
                                   amplitude_range,
                                   decay_time,
                                   noise_std):
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
    # Calculate number of samples
    num_samples = int(duration * sampling_rate)

    # Initialize the signal with zeros
    signal = np.zeros(num_samples)

    # Generate calcium events
    num_events = np.random.poisson(event_rate * duration)
    event_times = np.random.uniform(0, duration, num_events)
    event_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], num_events)

    # Add calcium events to the signal
    for t, a in zip(event_times, event_amplitudes):
        event_index = int(t * sampling_rate)
        decay = np.exp(-np.arange(num_samples - event_index) / (decay_time * sampling_rate))
        signal[event_index:] += a * decay

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, num_samples)
    signal += noise

    return signal


def generate_pseudo_calcium_multisignal(n, duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std):
    sigs = []
    for i in range(n):
        sig = generate_pseudo_calcium_signal(duration, sampling_rate, event_rate, amplitude_range, decay_time, noise_std)
        sigs.append(sig)

    return np.vstack(sigs)