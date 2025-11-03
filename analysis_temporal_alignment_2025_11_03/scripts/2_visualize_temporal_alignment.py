"""Visualize temporal alignment between observed and reconstructed calcium signals.

This script creates detailed visualizations showing:
1. Observed vs reconstructed calcium overlays
2. Event markers (detected spikes)
3. Temporal misalignment patterns
4. Cross-correlation analysis
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

ca_data = data['arr_0']
fps = 20.0
n_neurons = ca_data.shape[0]

print(f"Loaded {n_neurons} neurons at {fps} Hz")

# Focus on neuron #192 (high SNR)
neuron_idx = 192

print(f"\nAnalyzing Neuron #{neuron_idx}")

# Create neuron
ca_array = ca_data[neuron_idx]
sp_array = np.zeros(ca_data.shape[1], dtype=int)
neuron = Neuron(neuron_idx, ca_array, sp_array, fps=fps)

# Reconstruct
neuron.reconstruct_spikes(
    method='wavelet',
    amplitude_method='deconvolution',
    n_iter=3,
    show_progress=False
)

# Get kinetics
t_rise = neuron.t_rise if neuron.t_rise is not None else neuron.default_t_rise
t_off = neuron.t_off if neuron.t_off is not None else neuron.default_t_off

print(f"Kinetics: t_rise={t_rise:.2f} frames, t_off={t_off:.2f} frames")

# Calculate kernel peak offset
if t_off > t_rise and abs(t_off - t_rise) >= 0.1:
    kernel_peak_offset = t_rise * t_off * np.log(t_off / t_rise) / (t_off - t_rise)
else:
    kernel_peak_offset = 0

print(f"Kernel peak offset: {kernel_peak_offset:.2f} frames ({kernel_peak_offset/fps*1000:.1f}ms)")

# Get signals
observed = neuron.ca.data
reconstructed = neuron.reconstructed.data

# Get event times
event_times = np.where(neuron.asp.data > 0)[0]
event_amplitudes = neuron.asp.data[neuron.asp.data > 0]

print(f"Number of events: {len(event_times)}")

# Calculate quality metrics
r2 = neuron.get_reconstruction_r2(event_only=True)
corr = np.corrcoef(observed, reconstructed)[0, 1]

print(f"Event R²: {r2:.4f}")
print(f"Correlation: {corr:.4f}")

# Cross-correlation analysis
def cross_correlate(sig1, sig2, max_lag=50):
    s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-10)
    s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-10)

    correlation = signal.correlate(s1, s2, mode='same')
    lags = signal.correlation_lags(len(s1), len(s2), mode='same')

    # Restrict to search range
    center = len(correlation) // 2
    start_idx = max(0, center - max_lag)
    end_idx = min(len(correlation), center + max_lag + 1)

    search_correlation = correlation[start_idx:end_idx]
    search_lags = lags[start_idx:end_idx]

    peak_idx = np.argmax(np.abs(search_correlation))
    optimal_lag = search_lags[peak_idx]

    return optimal_lag, search_correlation, search_lags

lag, corr_vals, lag_vals = cross_correlate(observed, reconstructed)
print(f"Cross-correlation lag: {lag} frames ({lag/fps*1000:.1f}ms)")

# Create comprehensive visualization
output_dir = Path(__file__).parent.parent / "plots"
output_dir.mkdir(exist_ok=True)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Full time course
ax1 = plt.subplot(4, 1, 1)
time = np.arange(len(observed)) / fps
ax1.plot(time, observed, 'k-', alpha=0.6, label='Observed', linewidth=1)
ax1.plot(time, reconstructed, 'r-', alpha=0.7, label='Reconstructed', linewidth=1.5)

# Mark events
for i, (t_idx, amp) in enumerate(zip(event_times, event_amplitudes)):
    ax1.axvline(t_idx / fps, color='blue', alpha=0.3, linewidth=0.5, linestyle='--')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('ΔF/F')
ax1.set_title(f'Neuron #{neuron_idx}: Full Time Course | Event R²={r2:.3f}, Corr={corr:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Zoom on first 100s
ax2 = plt.subplot(4, 1, 2)
zoom_end = min(int(100 * fps), len(observed))
ax2.plot(time[:zoom_end], observed[:zoom_end], 'k-', alpha=0.6, label='Observed', linewidth=1.5)
ax2.plot(time[:zoom_end], reconstructed[:zoom_end], 'r-', alpha=0.7, label='Reconstructed', linewidth=1.5)

# Mark events in zoom
zoom_events = event_times[event_times < zoom_end]
for t_idx in zoom_events:
    ax2.axvline(t_idx / fps, color='blue', alpha=0.4, linewidth=1, linestyle='--')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('ΔF/F')
ax2.set_title('Zoom: First 100 seconds')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Individual event examples
ax3 = plt.subplot(4, 2, 5)

# Show first 6 events
n_examples = min(6, len(event_times))
window = 40  # frames around event

for i in range(n_examples):
    event_time = event_times[i]
    start_idx = max(0, event_time - window // 2)
    end_idx = min(len(observed), event_time + window // 2)

    local_time = (np.arange(start_idx, end_idx) - event_time) / fps

    if i < 3:
        ax = plt.subplot(4, 2, 5 + i % 3)
    else:
        ax = plt.subplot(4, 2, 6 + i % 3)

    ax.plot(local_time, observed[start_idx:end_idx], 'k-', label='Obs', linewidth=1.5)
    ax.plot(local_time, reconstructed[start_idx:end_idx], 'r-', label='Rec', linewidth=1.5)
    ax.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Event')

    # Show kernel peak offset
    if kernel_peak_offset > 0:
        ax.axvline(kernel_peak_offset / fps, color='green', linestyle=':', alpha=0.7, label=f'Kernel peak offset')

    ax.set_xlabel('Time rel. to event (s)')
    ax.set_ylabel('ΔF/F')
    ax.set_title(f'Event #{i+1}')
    if i == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Plot 4: Cross-correlation
ax4 = plt.subplot(4, 2, 8)
ax4.plot(lag_vals / fps * 1000, corr_vals, 'b-', linewidth=2)
ax4.axvline(lag / fps * 1000, color='r', linestyle='--', label=f'Peak lag: {lag/fps*1000:.1f}ms')
ax4.set_xlabel('Lag (ms)')
ax4.set_ylabel('Cross-correlation')
ax4.set_title('Cross-Correlation Analysis')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

output_path = output_dir / f"temporal_alignment_neuron{neuron_idx}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {output_path}")

plt.close()

# Create residual analysis plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Residuals
residuals = observed - reconstructed

# Plot 1: Residuals over time
ax = axes[0]
ax.plot(time, residuals, 'k-', alpha=0.5, linewidth=0.5)
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.fill_between(time, -np.std(residuals), np.std(residuals), alpha=0.2, color='gray')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Residual (ΔF/F)')
ax.set_title(f'Reconstruction Residuals | RMSE={np.sqrt(np.mean(residuals**2)):.4f}')
ax.grid(True, alpha=0.3)

# Plot 2: Residuals histogram
ax = axes[1]
ax.hist(residuals, bins=100, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(0, color='r', linestyle='--', linewidth=2)
ax.axvline(np.mean(residuals), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
ax.set_xlabel('Residual (ΔF/F)')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Q-Q plot for normality check
from scipy import stats
ax = axes[2]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normality Check)')
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path = output_dir / f"residual_analysis_neuron{neuron_idx}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Residual plot saved: {output_path}")

plt.close()

print("\nVisualization complete!")
