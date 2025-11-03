"""
Detailed analysis of high-SNR neuron #192 to understand why R² is moderate despite exceptional SNR.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd

# Add project to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from driada.experiment.neuron import Neuron

# Load data
data_path = Path("/Users/nikita/PycharmProjects/driada2/temp/wavelet_analysis/real calcium.npz")
data = np.load(data_path)
ca_signals = data['arr_0']

# Load SNR metrics
metrics_path = Path("/Users/nikita/PycharmProjects/driada2/temp/wavelet_analysis/wavelet_snr_metrics.csv")
df = pd.read_csv(metrics_path)

# Focus on neuron #192 (highest SNR = 34.8)
neuron_id = 192
ca = ca_signals[neuron_id]
time = np.arange(len(ca)) / 20.0

print("="*80)
print(f"DETAILED ANALYSIS: NEURON #{neuron_id} (Highest SNR)")
print("="*80)
print()

# Get SNR metrics
row = df[df['neuron_id'] == neuron_id].iloc[0]
print("Wavelet SNR Metrics:")
print(f"  SNR Wavelet: {row['snr_wavelet']:.1f}")
print(f"  Composite Score: {row['composite_score']:.3f}")
print(f"  Peak Consistency: {row['peak_consistency']:.3f}")
print(f"  Event Count: {int(row['event_count'])}")
print(f"  Median Amplitude: {row['median_event_amplitude']:.4f}")
print(f"  Baseline Noise: {row['baseline_noise']:.4f}")
print()

# Create neuron object and reconstruct with DEFAULT kinetics
print("Step 1: Reconstruction with DEFAULT kinetics (t_rise=5, t_off=40 frames)")
neuron = Neuron(cell_id=neuron_id, ca=ca, sp=None)
neuron.reconstruct_spikes(
    method='wavelet',
    iterative=True,
    n_iter=2,
    create_event_regions=True,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    sigma=8
)

# Get reconstruction (use actual kinetics, fallback to defaults if not set)
t_rise = neuron.t_rise if neuron.t_rise is not None else neuron.default_t_rise
t_off = neuron.t_off if neuron.t_off is not None else neuron.default_t_off
ca_recon_default = Neuron.get_restored_calcium(neuron.asp.data, t_rise, t_off)
print(f"  Kinetics used: t_rise={t_rise:.1f} frames, t_off={t_off:.1f} frames")

# Calculate metrics
r2_full_default = neuron.get_reconstruction_r2(event_only=False)
r2_event_default = neuron.get_reconstruction_r2(event_only=True, use_detected_events=True)
mae_default = neuron.get_mae()
rmse_default = np.sqrt(np.mean((ca - ca_recon_default)**2))

print(f"  Events detected: {neuron.sp_count}")
print(f"  R² (full): {r2_full_default:.4f}")
print(f"  R² (event): {r2_event_default:.4f}")
print(f"  MAE: {mae_default:.6f}")
print(f"  RMSE: {rmse_default:.6f}")
print()

# Store default event positions
default_event_idx = np.where(neuron.sp.data > 0)[0]
default_amplitudes = neuron.asp.data[default_event_idx]

# Step 2: Optimize kinetics
print("Step 2: Optimizing kinetics...")
opt_result = neuron.optimize_kinetics(method='direct', fps=20.0, update_reconstruction=False)

if opt_result['optimized']:
    t_rise_opt = opt_result['t_rise']  # seconds
    t_off_opt = opt_result['t_off']
    print(f"  Optimized: t_rise={t_rise_opt:.3f}s ({t_rise_opt*20:.1f} frames), t_off={t_off_opt:.3f}s ({t_off_opt*20:.1f} frames)")

    # Set optimized kinetics
    neuron.t_rise = t_rise_opt * 20.0
    neuron.t_off = t_off_opt * 20.0

    # Re-detect with optimized kinetics
    print()
    print("Step 3: Re-detecting events with OPTIMIZED kinetics")
    neuron.reconstruct_spikes(
        method='wavelet',
        iterative=True,
        n_iter=2,
        create_event_regions=True,
        scale_length_thr=40,
        max_scale_thr=7,
        max_ampl_thr=0.05,
        sigma=8
    )

    # Get optimized reconstruction
    ca_recon_opt = Neuron.get_restored_calcium(neuron.asp.data, neuron.t_rise, neuron.t_off)

    # Calculate metrics
    r2_full_opt = neuron.get_reconstruction_r2(event_only=False)
    r2_event_opt = neuron.get_reconstruction_r2(event_only=True, use_detected_events=True)
    mae_opt = neuron.get_mae()
    rmse_opt = np.sqrt(np.mean((ca - ca_recon_opt)**2))

    print(f"  Events detected: {neuron.sp_count}")
    print(f"  R² (full): {r2_full_opt:.4f}")
    print(f"  R² (event): {r2_event_opt:.4f}")
    print(f"  MAE: {mae_opt:.6f}")
    print(f"  RMSE: {rmse_opt:.6f}")
    print()

    # Store optimized event positions
    opt_event_idx = np.where(neuron.sp.data > 0)[0]
    opt_amplitudes = neuron.asp.data[opt_event_idx]

    # Analysis: Compare event detection
    print("Step 4: Comparing DEFAULT vs OPTIMIZED reconstruction")
    print(f"  Event count: {len(default_event_idx)} → {len(opt_event_idx)} (change: {len(opt_event_idx) - len(default_event_idx):+d})")
    print(f"  R² improvement: {r2_event_opt - r2_event_default:+.4f} ({(r2_event_opt/r2_event_default - 1)*100:+.1f}%)")
    print()

    # Analysis: Peak amplitude comparison
    print("Step 5: Amplitude Analysis")

    # Calculate peak amplitudes in original signal at detected events
    signal_peaks_at_events = []
    recon_peaks_at_events = []

    for ev_idx in opt_event_idx:
        # Find peak in original signal within event region
        search_start = max(0, ev_idx - 5)
        search_end = min(len(ca), ev_idx + 20)
        signal_peak = np.max(ca[search_start:search_end])
        recon_peak = np.max(ca_recon_opt[search_start:search_end])

        signal_peaks_at_events.append(signal_peak)
        recon_peaks_at_events.append(recon_peak)

    signal_peaks = np.array(signal_peaks_at_events)
    recon_peaks = np.array(recon_peaks_at_events)

    # Calculate systematic bias
    amplitude_ratio = np.median(recon_peaks / signal_peaks)
    amplitude_error = np.median(signal_peaks - recon_peaks)

    print(f"  Median signal peak at events: {np.median(signal_peaks):.4f}")
    print(f"  Median reconstruction peak at events: {np.median(recon_peaks):.4f}")
    print(f"  Reconstruction/Signal ratio: {amplitude_ratio:.3f} ({(amplitude_ratio-1)*100:+.1f}%)")
    print(f"  Median underestimation: {amplitude_error:.6f}")
    print()

    # Analysis: Temporal alignment
    print("Step 6: Temporal Alignment Analysis")

    peak_offsets = []
    for ev_idx in opt_event_idx[:min(20, len(opt_event_idx))]:  # Check first 20 events
        # Find peak in signal
        search_start = max(0, ev_idx - 5)
        search_end = min(len(ca), ev_idx + 30)
        signal_peak_idx = search_start + np.argmax(ca[search_start:search_end])
        recon_peak_idx = search_start + np.argmax(ca_recon_opt[search_start:search_end])

        offset_frames = signal_peak_idx - ev_idx
        offset_time = offset_frames / 20.0
        peak_offsets.append(offset_frames)

    peak_offsets = np.array(peak_offsets)
    print(f"  Median peak offset from event marker: {np.median(peak_offsets):.1f} frames ({np.median(peak_offsets)/20:.3f}s)")
    print(f"  Peak offset std: {np.std(peak_offsets):.1f} frames")
    print(f"  Expected peak offset for t_rise={t_rise_opt:.3f}s: ~{t_rise_opt*20:.1f} frames")
    print()

    # Save analysis results
    results = {
        'neuron_id': neuron_id,
        'snr_wavelet': row['snr_wavelet'],
        'default_r2_event': r2_event_default,
        'optimized_r2_event': r2_event_opt,
        'r2_improvement': r2_event_opt - r2_event_default,
        'default_events': len(default_event_idx),
        'optimized_events': len(opt_event_idx),
        't_rise_opt': t_rise_opt,
        't_off_opt': t_off_opt,
        'amplitude_ratio': amplitude_ratio,
        'amplitude_underestimation': amplitude_error,
        'median_peak_offset_frames': np.median(peak_offsets),
        'peak_offset_std_frames': np.std(peak_offsets)
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv('../data/neuron_192_analysis.csv', index=False)
    print("Results saved to: ../data/neuron_192_analysis.csv")
    print()

    # Create visualization
    print("Creating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 60 seconds
    plot_duration = 60.0
    plot_frames = int(plot_duration * 20.0)
    time_plot = time[:plot_frames]

    # Top: Original signal with event markers
    ax = axes[0]
    ax.plot(time_plot, ca[:plot_frames], 'k-', linewidth=1.5, alpha=0.8, label='Original Ca²⁺')

    # Mark detected events
    det_idx_plot = opt_event_idx[opt_event_idx < plot_frames]
    if len(det_idx_plot) > 0:
        ax.scatter(time[det_idx_plot], ca[det_idx_plot],
                  color='red', s=80, marker='o', label=f'Detected events (n={len(det_idx_plot)})',
                  zorder=10, alpha=0.8, facecolors='none', edgecolors='red', linewidths=2)

    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'Neuron #{neuron_id}: Original Signal + Detected Events (SNR={row["snr_wavelet"]:.1f})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Middle: Original + Default reconstruction
    ax = axes[1]
    ax.plot(time_plot, ca[:plot_frames], 'k-', linewidth=1.5, alpha=0.7, label='Original Ca²⁺')
    ax.plot(time_plot, ca_recon_default[:plot_frames], 'b-', linewidth=1.2, alpha=0.8,
           label=f'Default reconstruction (t_rise=5, t_off=40)')

    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'DEFAULT Kinetics: R²(event)={r2_event_default:.3f}, Events={len(default_event_idx)}',
                fontsize=12, fontweight='bold', color='blue')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom: Original + Optimized reconstruction
    ax = axes[2]
    ax.plot(time_plot, ca[:plot_frames], 'k-', linewidth=1.5, alpha=0.7, label='Original Ca²⁺')
    ax.plot(time_plot, ca_recon_opt[:plot_frames], 'g-', linewidth=1.2, alpha=0.8,
           label=f'Optimized reconstruction (t_rise={t_rise_opt:.2f}s, t_off={t_off_opt:.2f}s)')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'OPTIMIZED Kinetics: R²(event)={r2_event_opt:.3f}, Events={len(opt_event_idx)}',
                fontsize=12, fontweight='bold', color='green')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/neuron_192_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: ../plots/neuron_192_detailed_analysis.png")

    plt.close()

    # Create amplitude comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Scatter plot of signal vs reconstruction peaks
    ax = axes[0]
    ax.scatter(signal_peaks, recon_peaks, alpha=0.6, s=50)

    # Add identity line
    max_val = max(np.max(signal_peaks), np.max(recon_peaks))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect match')

    # Add fitted line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(signal_peaks, recon_peaks)
    ax.plot(signal_peaks, slope * signal_peaks + intercept, 'b-', linewidth=2,
           label=f'Fitted (slope={slope:.3f}, R²={r_value**2:.3f})')

    ax.set_xlabel('Signal Peak Amplitude', fontsize=12)
    ax.set_ylabel('Reconstruction Peak Amplitude', fontsize=12)
    ax.set_title(f'Peak Amplitude Comparison (Neuron #{neuron_id})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = (f'Median signal peak: {np.median(signal_peaks):.4f}\n'
                 f'Median recon peak: {np.median(recon_peaks):.4f}\n'
                 f'Ratio: {amplitude_ratio:.3f}\n'
                 f'Underestimation: {amplitude_error:.6f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Bottom: Histogram of amplitude ratios
    ax = axes[1]
    ratios = recon_peaks / signal_peaks
    ax.hist(ratios, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(ratios), color='red', linestyle='--', linewidth=2,
              label=f'Median={np.median(ratios):.3f}')
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2,
              label='Perfect match=1.0')

    ax.set_xlabel('Reconstruction/Signal Amplitude Ratio', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Amplitude Ratios', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/neuron_192_amplitude_analysis.png', dpi=300, bbox_inches='tight')
    print("Amplitude analysis saved to: ../plots/neuron_192_amplitude_analysis.png")

    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

else:
    print("ERROR: Kinetics optimization failed!")
