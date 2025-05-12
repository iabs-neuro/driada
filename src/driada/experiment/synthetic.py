import numpy as np
from fbm import FBM
import itertools
import tqdm
from .exp_base import *
from ..information.info_base import TimeSeries, MultiTimeSeries


def generate_pseudo_calcium_multisignal(n,
                                        events=None,
                                        duration=600,
                                        sampling_rate=20,
                                        event_rate=0.2,
                                        amplitude_range=(0.5,2),
                                        decay_time=2,
                                        noise_std=0.1):
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
        event_amplitudes = events[event_times]

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


def generate_binary_time_series(length, avg_islands, avg_duration):
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
    while actual_islands != avg_islands:
        if actual_islands < avg_islands:
            # If we have too few islands, turn on a random '0' to create a new island
            zero_positions = np.where(series == 0)[0]
            if len(zero_positions) > 0:
                turn_on = np.random.choice(zero_positions)
                series[turn_on] = 1
                actual_islands += 1
        else:
            # If we have too many islands, turn off a random '1' to merge islands
            one_positions = np.where(series == 1)[0]
            if len(one_positions) > 1:
                turn_off = np.random.choice(one_positions)
                series[turn_off] = 0
                actual_islands -= 1

    return series


def apply_poisson_to_binary_series(binary_series, rate_0, rate_1):
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


def generate_binary_time_series(length, avg_islands, avg_duration):
    series = np.zeros(length, dtype=int)
    current_state = 0
    position = 0

    while position < length:
        if current_state == 0:
            off_duration = max(1, int(np.random.exponential(length / (avg_islands * 2))))
            duration = min(off_duration, length - position)
        else:
            duration = max(1, int(np.random.normal(avg_duration, avg_duration / 2)))

        duration = min(duration, length - position)
        series[position:position + duration] = current_state
        current_state = 1 - current_state
        position += duration

    return series


from itertools import groupby
import numpy as np


def delete_one_islands(binary_ts, probability):
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


def generate_fbm_time_series(length, hurst, seed=None):
    if seed is not None:
        np.random.seed(seed)

    f = FBM(n=length-1, hurst=hurst, length=1.0, method='daviesharte')
    fbm_series = f.fbm()

    return fbm_series


def select_signal_roi(values, seed=42):
    mean = np.mean(values)
    std = np.std(values)

    np.random.seed(seed)
    # Select random location within mean ± 2*std
    loc = np.random.uniform(mean - 1.5 * std, mean + 1.5 * std)

    # Define borders
    lower_border = loc - 0.5 * std
    upper_border = loc + 0.5 * std

    return loc, lower_border, upper_border


def generate_synthetic_data(nfeats, nneurons, ftype='c', duration=600, seed=42, sampling_rate=20.0,
                            rate_0=0.1, rate_1=1.0, skip_prob=0.0, hurst=0.5, ampl_range=(0.5, 2), decay_time=2,
                            avg_islands=10, avg_duration=5, noise_std=0.1, verbose=True):
    gt = np.zeros((nfeats, nneurons))
    length = int(duration * sampling_rate)

    print('Generating features...')
    all_feats = []
    for i in tqdm.tqdm(np.arange(nfeats)):
        if ftype == 'c':
            # Generate the series
            fbm_series = generate_fbm_time_series(length, hurst, seed=seed)
            all_feats.append(fbm_series)

        elif ftype == 'd':
            # Generate binary series
            binary_series = generate_binary_time_series(length, avg_islands, avg_duration * sampling_rate)
            all_feats.append(binary_series)

        else:
            raise ValueError('unknown feature flag')

        seed += 1  # save reproducibility, but break degeneracy

    print('Generating signals...')
    fois = np.random.choice(np.arange(nfeats), size=nneurons)
    gt[fois, np.arange(nneurons)] = 1  # add info about ground truth feature-signal connections
    all_signals = []

    for j in tqdm.tqdm(np.arange(nneurons)):
        foi = fois[j]
        if ftype == 'c':
            csignal = all_feats[foi]
            loc, lower_border, upper_border = select_signal_roi(csignal, seed=seed)
            # Generate binary series from a continuous one
            binary_series = np.zeros(length)
            binary_series[np.where((csignal >= lower_border) & (csignal <= upper_border))] = 1

        elif ftype == 'd':
            binary_series = all_feats[foi]

        else:
            raise ValueError('unknown feature flag')

        # randomly skip some on periods
        mod_binary_series = delete_one_islands(binary_series, skip_prob)

        # Apply Poisson process
        poisson_series = apply_poisson_to_binary_series(mod_binary_series,
                                                        rate_0 / sampling_rate,
                                                        rate_1 / sampling_rate)

        # Generate pseudo-calcium
        pseudo_calcium_signal = generate_pseudo_calcium_signal(duration=duration,
                                                               events=poisson_series,
                                                               sampling_rate=sampling_rate,
                                                               amplitude_range=ampl_range,
                                                               decay_time=decay_time,
                                                               noise_std=noise_std)

        all_signals.append(pseudo_calcium_signal)
        seed += 1  # save reproducibility, but break degeneracy

    return np.vstack(all_feats), np.vstack(all_signals), gt


def generate_synthetic_exp(n_dfeats=20, n_cfeats=20, nneurons=500, seed=0, fps=20):
    dfeats, calcium1, gt = generate_synthetic_data(n_dfeats,
                                                   nneurons//2,
                                                   duration=1200,
                                                   hurst=0.3,
                                                   ftype='d',
                                                   seed=seed,
                                                   rate_0=0.1,
                                                   rate_1=1.0,
                                                   skip_prob=0.1,
                                                   noise_std=0.1,
                                                   sampling_rate=fps)

    cfeats, calcium2, gt2 = generate_synthetic_data(n_dfeats,
                                                    nneurons//2,
                                                    duration=1200,
                                                    hurst=0.3,
                                                    ftype='c',
                                                    seed=seed,
                                                    rate_0=0.1,
                                                    rate_1=1.0,
                                                    skip_prob=0.1,
                                                    noise_std=0.1,
                                                    sampling_rate=fps)

    discr_ts = {f'd_feat_{i}': TimeSeries(dfeats[i, :], discrete=True) for i in range(len(dfeats))}
    cont_ts = {f'c_feat_{i}': TimeSeries(cfeats[i, :], discrete=False) for i in range(len(cfeats))}

    exp = Experiment('Synthetic',
                     np.vstack([calcium1, calcium2]),
                     None,
                     {},
                     {'fps': fps},
                     {**discr_ts, **cont_ts},
                     reconstruct_spikes=None)

    return exp
