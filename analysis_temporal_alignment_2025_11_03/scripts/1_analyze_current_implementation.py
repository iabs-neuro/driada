"""Analyze current event placement strategy and measure temporal alignment.

This script:
1. Loads high-SNR neurons from real calcium data
2. Reconstructs spikes using current implementation
3. Measures temporal lag between observed and reconstructed signals
4. Analyzes correlation with kinetics parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from driada.experiment.neuron import Neuron
from driada.information.info_base import TimeSeries

# Load real calcium data
data_path = Path(__file__).parents[2] / "temp/wavelet_analysis/real calcium.npz"
data = np.load(data_path)

ca_data = data['arr_0']  # Real calcium data stored as arr_0
fps = 20.0  # Standard fps for this dataset
n_neurons = ca_data.shape[0]

print(f"Loaded {n_neurons} neurons at {fps} Hz")
print(f"Recording duration: {ca_data.shape[1] / fps:.1f} seconds")

# Test neurons (high SNR)
test_neurons = [192, 33, 146, 22]

def find_peaks_in_signal(signal_data, min_distance_frames=10):
    """Find peaks in calcium signal."""
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(signal_data, distance=min_distance_frames, prominence=0.05)
    return peaks, properties

def cross_correlate_signals(signal1, signal2, max_lag_frames=20):
    """Calculate cross-correlation and find optimal lag."""
    # Normalize signals
    s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
    s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

    # Cross-correlation
    correlation = signal.correlate(s1, s2, mode='same')
    lags = signal.correlation_lags(len(s1), len(s2), mode='same')

    # Find peak within search range
    center = len(correlation) // 2
    search_start = max(0, center - max_lag_frames)
    search_end = min(len(correlation), center + max_lag_frames + 1)

    search_region = correlation[search_start:search_end]
    peak_idx = np.argmax(np.abs(search_region))
    optimal_lag = lags[search_start + peak_idx]

    return optimal_lag, correlation, lags

def measure_event_level_lags(observed, reconstructed, event_times, window_frames=15):
    """Measure temporal lag for each individual event."""
    lags = []

    for event_time in event_times:
        # Extract window around event
        start = max(0, event_time - window_frames)
        end = min(len(observed), event_time + window_frames)

        if end - start < 10:  # Skip if window too small
            continue

        obs_window = observed[start:end]
        rec_window = reconstructed[start:end]

        # Find peaks in both windows
        obs_peaks, _ = find_peaks_in_signal(obs_window, min_distance_frames=3)
        rec_peaks, _ = find_peaks_in_signal(rec_window, min_distance_frames=3)

        if len(obs_peaks) > 0 and len(rec_peaks) > 0:
            # Take highest peak in each window
            obs_peak = obs_peaks[np.argmax(obs_window[obs_peaks])]
            rec_peak = rec_peaks[np.argmax(rec_window[rec_peaks])]

            # Convert to absolute time and calculate lag
            obs_abs = start + obs_peak
            rec_abs = start + rec_peak
            lag = rec_abs - obs_abs
            lags.append(lag)

    return np.array(lags)

# Analyze each test neuron
results = []

for neuron_idx in test_neurons:
    print(f"\n{'='*60}")
    print(f"Analyzing Neuron #{neuron_idx}")
    print(f"{'='*60}")

    # Create neuron object
    ca_array = ca_data[neuron_idx]
    sp_array = np.zeros(ca_data.shape[1], dtype=int)  # Empty spikes initially
    neuron = Neuron(neuron_idx, ca_array, sp_array, fps=fps)

    # Reconstruct spikes using current implementation
    neuron.reconstruct_spikes(
        method='wavelet',
        amplitude_method='deconvolution',
        n_iter=3,
        show_progress=False
    )

    # Calculate SNR using wavelet method (works without ground truth spikes)
    try:
        snr = neuron.get_wavelet_snr()
    except:
        snr = -1  # SNR calculation failed
    print(f"SNR: {snr:.2f}")

    # Get kinetics
    t_rise = neuron.t_rise if neuron.t_rise is not None else neuron.default_t_rise
    t_off = neuron.t_off if neuron.t_off is not None else neuron.default_t_off

    print(f"Kinetics: t_rise={t_rise:.3f} frames ({t_rise/fps*1000:.1f}ms), "
          f"t_off={t_off:.3f} frames ({t_off/fps*1000:.1f}ms)")

    # Get reconstruction
    if neuron.reconstructed is None:
        print("Warning: No reconstruction available")
        continue

    observed = neuron.ca.data
    reconstructed = neuron.reconstructed.data

    # Calculate reconstruction quality (event-level)
    r2 = neuron.get_reconstruction_r2(event_only=True)
    corr = np.corrcoef(observed, reconstructed)[0, 1]
    corr2 = corr ** 2

    print(f"Event R²: {r2:.4f}")
    print(f"Correlation²: {corr2:.4f}")
    print(f"Gap (Corr² - R²): {corr2 - r2:.4f}")

    # Measure temporal lag - global
    lag_global, correlation, lags = cross_correlate_signals(observed, reconstructed)
    print(f"\nGlobal lag: {lag_global} frames ({lag_global/fps*1000:.1f}ms)")

    # Measure event-level lags
    if neuron.sp is not None and len(neuron.sp.data) > 0:
        event_times = np.where(neuron.sp.data > 0)[0]
        print(f"Number of detected events: {len(event_times)}")

        if len(event_times) > 0:
            event_lags = measure_event_level_lags(observed, reconstructed, event_times)

            if len(event_lags) > 0:
                print(f"\nEvent-level lags (n={len(event_lags)}):")
                print(f"  Mean: {np.mean(event_lags):.2f} frames ({np.mean(event_lags)/fps*1000:.1f}ms)")
                print(f"  Std: {np.std(event_lags):.2f} frames ({np.std(event_lags)/fps*1000:.1f}ms)")
                print(f"  Median: {np.median(event_lags):.2f} frames ({np.median(event_lags)/fps*1000:.1f}ms)")
                print(f"  Range: [{np.min(event_lags):.1f}, {np.max(event_lags):.1f}] frames")

                # Check if lag correlates with kinetics
                kernel_peak_offset = t_rise * t_off * np.log(t_off / t_rise) / (t_off - t_rise)
                print(f"\nKernel peak offset: {kernel_peak_offset:.2f} frames ({kernel_peak_offset/fps*1000:.1f}ms)")
                print(f"Ratio (mean_lag / kernel_offset): {np.mean(event_lags) / kernel_peak_offset:.3f}")

                results.append({
                    'neuron_idx': neuron_idx,
                    'snr': snr,
                    'r2': r2,
                    'corr2': corr2,
                    'gap': corr2 - r2,
                    't_rise': t_rise,
                    't_off': t_off,
                    't_rise_ms': t_rise/fps*1000,
                    't_off_ms': t_off/fps*1000,
                    'kernel_peak_offset': kernel_peak_offset,
                    'kernel_peak_offset_ms': kernel_peak_offset/fps*1000,
                    'lag_global': lag_global,
                    'lag_global_ms': lag_global/fps*1000,
                    'event_lags_mean': np.mean(event_lags),
                    'event_lags_std': np.std(event_lags),
                    'event_lags_median': np.median(event_lags),
                    'event_lags_mean_ms': np.mean(event_lags)/fps*1000,
                    'event_lags_std_ms': np.std(event_lags)/fps*1000,
                    'n_events': len(event_times),
                    'n_lag_measurements': len(event_lags),
                    'lag_to_offset_ratio': np.mean(event_lags) / kernel_peak_offset if kernel_peak_offset > 0 else np.nan
                })

# Save results
output_path = Path(__file__).parent.parent / "data"
output_path.mkdir(exist_ok=True)

if len(results) > 0:
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = output_path / "current_implementation_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL NEURONS")
    print(f"{'='*60}")
    print(f"Mean R²: {df['r2'].mean():.4f} ± {df['r2'].std():.4f}")
    print(f"Mean Correlation²: {df['corr2'].mean():.4f} ± {df['corr2'].std():.4f}")
    print(f"Mean Gap (Corr² - R²): {df['gap'].mean():.4f} ± {df['gap'].std():.4f}")
    print(f"\nMean event lag: {df['event_lags_mean_ms'].mean():.1f} ± {df['event_lags_mean_ms'].std():.1f} ms")
    print(f"Mean kernel offset: {df['kernel_peak_offset_ms'].mean():.1f} ± {df['kernel_peak_offset_ms'].std():.1f} ms")
    print(f"Mean lag/offset ratio: {df['lag_to_offset_ratio'].mean():.3f} ± {df['lag_to_offset_ratio'].std():.3f}")

    # Check correlation
    from scipy.stats import pearsonr

    if len(df) > 2:
        corr_gap_lag, p_gap_lag = pearsonr(df['gap'], df['event_lags_mean_ms'])
        print(f"\nCorrelation between gap and lag: r={corr_gap_lag:.3f} (p={p_gap_lag:.4f})")

        corr_offset_lag, p_offset_lag = pearsonr(df['kernel_peak_offset_ms'], df['event_lags_mean_ms'])
        print(f"Correlation between kernel offset and lag: r={corr_offset_lag:.3f} (p={p_offset_lag:.4f})")
else:
    print("\nNo results to save")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
