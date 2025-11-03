"""
Compare 'peak' vs 'deconvolution' amplitude extraction methods to see which causes underestimation.
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

# Focus on neuron #192 (highest SNR)
neuron_id = 192
ca = ca_signals[neuron_id]
time = np.arange(len(ca)) / 20.0

print("="*80)
print(f"AMPLITUDE EXTRACTION METHOD COMPARISON: NEURON #{neuron_id}")
print("="*80)
print()

# Test 1: Peak-based amplitude extraction
print("Test 1: Peak-based amplitude extraction (amplitude_method='peak')")
neuron_peak = Neuron(cell_id=neuron_id, ca=ca, sp=None)
neuron_peak.reconstruct_spikes(
    method='wavelet',
    iterative=True,
    n_iter=2,
    amplitude_method='peak',
    create_event_regions=True,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    sigma=8
)

# Optimize kinetics
opt_result_peak = neuron_peak.optimize_kinetics(method='direct', fps=20.0, update_reconstruction=False)
if opt_result_peak['optimized']:
    t_rise_opt = opt_result_peak['t_rise'] * 20.0  # frames
    t_off_opt = opt_result_peak['t_off'] * 20.0
    print(f"  Optimized kinetics: t_rise={t_rise_opt:.1f} frames, t_off={t_off_opt:.1f} frames")

    # Re-detect with optimized kinetics
    neuron_peak.t_rise = t_rise_opt
    neuron_peak.t_off = t_off_opt
    neuron_peak.reconstruct_spikes(
        method='wavelet',
        iterative=True,
        n_iter=2,
        amplitude_method='peak',
        create_event_regions=True,
        scale_length_thr=40,
        max_scale_thr=7,
        max_ampl_thr=0.05,
        sigma=8
    )

    # Get metrics
    ca_recon_peak = Neuron.get_restored_calcium(neuron_peak.asp.data, t_rise_opt, t_off_opt)
    r2_event_peak = neuron_peak.get_reconstruction_r2(event_only=True, use_detected_events=True)

    # Analyze amplitudes
    peak_event_idx = np.where(neuron_peak.sp.data > 0)[0]
    peak_amplitudes = neuron_peak.asp.data[peak_event_idx]

    # Calculate peak ratios
    signal_peaks_peak = []
    recon_peaks_peak = []
    for ev_idx in peak_event_idx:
        search_start = max(0, ev_idx - 5)
        search_end = min(len(ca), ev_idx + 20)
        signal_peak = np.max(ca[search_start:search_end])
        recon_peak = np.max(ca_recon_peak[search_start:search_end])
        signal_peaks_peak.append(signal_peak)
        recon_peaks_peak.append(recon_peak)

    signal_peaks_peak = np.array(signal_peaks_peak)
    recon_peaks_peak = np.array(recon_peaks_peak)
    amplitude_ratio_peak = np.median(recon_peaks_peak / signal_peaks_peak)

    print(f"  Events detected: {neuron_peak.sp_count}")
    print(f"  R² (event): {r2_event_peak:.4f}")
    print(f"  Median extracted amplitude: {np.median(peak_amplitudes):.4f}")
    print(f"  Reconstruction/Signal ratio: {amplitude_ratio_peak:.3f} ({(amplitude_ratio_peak-1)*100:+.1f}%)")
    print()

# Test 2: Deconvolution-based amplitude extraction (default)
print("Test 2: Deconvolution-based amplitude extraction (amplitude_method='deconvolution')")
neuron_deconv = Neuron(cell_id=neuron_id, ca=ca, sp=None)
neuron_deconv.reconstruct_spikes(
    method='wavelet',
    iterative=True,
    n_iter=2,
    amplitude_method='deconvolution',
    create_event_regions=True,
    scale_length_thr=40,
    max_scale_thr=7,
    max_ampl_thr=0.05,
    sigma=8
)

# Use same optimized kinetics
opt_result_deconv = neuron_deconv.optimize_kinetics(method='direct', fps=20.0, update_reconstruction=False)
if opt_result_deconv['optimized']:
    neuron_deconv.t_rise = opt_result_deconv['t_rise'] * 20.0
    neuron_deconv.t_off = opt_result_deconv['t_off'] * 20.0
    neuron_deconv.reconstruct_spikes(
        method='wavelet',
        iterative=True,
        n_iter=2,
        amplitude_method='deconvolution',
        create_event_regions=True,
        scale_length_thr=40,
        max_scale_thr=7,
        max_ampl_thr=0.05,
        sigma=8
    )

    # Get metrics
    ca_recon_deconv = Neuron.get_restored_calcium(neuron_deconv.asp.data, neuron_deconv.t_rise, neuron_deconv.t_off)
    r2_event_deconv = neuron_deconv.get_reconstruction_r2(event_only=True, use_detected_events=True)

    # Analyze amplitudes
    deconv_event_idx = np.where(neuron_deconv.sp.data > 0)[0]
    deconv_amplitudes = neuron_deconv.asp.data[deconv_event_idx]

    # Calculate peak ratios
    signal_peaks_deconv = []
    recon_peaks_deconv = []
    for ev_idx in deconv_event_idx:
        search_start = max(0, ev_idx - 5)
        search_end = min(len(ca), ev_idx + 20)
        signal_peak = np.max(ca[search_start:search_end])
        recon_peak = np.max(ca_recon_deconv[search_start:search_end])
        signal_peaks_deconv.append(signal_peak)
        recon_peaks_deconv.append(recon_peak)

    signal_peaks_deconv = np.array(signal_peaks_deconv)
    recon_peaks_deconv = np.array(recon_peaks_deconv)
    amplitude_ratio_deconv = np.median(recon_peaks_deconv / signal_peaks_deconv)

    print(f"  Events detected: {neuron_deconv.sp_count}")
    print(f"  R² (event): {r2_event_deconv:.4f}")
    print(f"  Median extracted amplitude: {np.median(deconv_amplitudes):.4f}")
    print(f"  Reconstruction/Signal ratio: {amplitude_ratio_deconv:.3f} ({(amplitude_ratio_deconv-1)*100:+.1f}%)")
    print()

# Comparison
print("="*80)
print("COMPARISON")
print("="*80)
print(f"Peak method:")
print(f"  R² (event): {r2_event_peak:.4f}")
print(f"  Amplitude ratio: {amplitude_ratio_peak:.3f}")
print(f"  Median amplitude: {np.median(peak_amplitudes):.4f}")
print()
print(f"Deconvolution method:")
print(f"  R² (event): {r2_event_deconv:.4f}")
print(f"  Amplitude ratio: {amplitude_ratio_deconv:.3f}")
print(f"  Median amplitude: {np.median(deconv_amplitudes):.4f}")
print()
print(f"R² difference: {r2_event_deconv - r2_event_peak:+.4f} ({(r2_event_deconv/r2_event_peak - 1)*100:+.1f}%)")
print(f"Amplitude difference: {np.median(deconv_amplitudes) - np.median(peak_amplitudes):+.4f}")
print()

# Save results
results = {
    'neuron_id': neuron_id,
    'peak_r2_event': r2_event_peak,
    'deconv_r2_event': r2_event_deconv,
    'peak_amplitude_ratio': amplitude_ratio_peak,
    'deconv_amplitude_ratio': amplitude_ratio_deconv,
    'peak_median_amplitude': np.median(peak_amplitudes),
    'deconv_median_amplitude': np.median(deconv_amplitudes),
    'peak_events': neuron_peak.sp_count,
    'deconv_events': neuron_deconv.sp_count
}

results_df = pd.DataFrame([results])
results_df.to_csv('../data/amplitude_method_comparison.csv', index=False)
print("Results saved to: ../data/amplitude_method_comparison.csv")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 60 seconds
plot_duration = 60.0
plot_frames = int(plot_duration * 20.0)
time_plot = time[:plot_frames]

# Top: Peak method
ax = axes[0]
ax.plot(time_plot, ca[:plot_frames], 'k-', linewidth=1.5, alpha=0.7, label='Original Ca²⁺')
ax.plot(time_plot, ca_recon_peak[:plot_frames], 'b-', linewidth=1.2, alpha=0.8,
       label=f'Reconstruction (peak method)')

det_idx_plot = peak_event_idx[peak_event_idx < plot_frames]
if len(det_idx_plot) > 0:
    ax.scatter(time[det_idx_plot], ca[det_idx_plot],
              color='blue', s=80, marker='o', label=f'Detected events (n={len(det_idx_plot)})',
              zorder=10, alpha=0.8, facecolors='none', edgecolors='blue', linewidths=2)

ax.set_ylabel('ΔF/F', fontsize=12)
ax.set_title(f'PEAK METHOD: R²(event)={r2_event_peak:.3f}, Ratio={amplitude_ratio_peak:.3f}',
            fontsize=12, fontweight='bold', color='blue')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom: Deconvolution method
ax = axes[1]
ax.plot(time_plot, ca[:plot_frames], 'k-', linewidth=1.5, alpha=0.7, label='Original Ca²⁺')
ax.plot(time_plot, ca_recon_deconv[:plot_frames], 'g-', linewidth=1.2, alpha=0.8,
       label=f'Reconstruction (deconvolution method)')

det_idx_plot = deconv_event_idx[deconv_event_idx < plot_frames]
if len(det_idx_plot) > 0:
    ax.scatter(time[det_idx_plot], ca[det_idx_plot],
              color='green', s=80, marker='o', label=f'Detected events (n={len(det_idx_plot)})',
              zorder=10, alpha=0.8, facecolors='none', edgecolors='green', linewidths=2)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('ΔF/F', fontsize=12)
ax.set_title(f'DECONVOLUTION METHOD: R²(event)={r2_event_deconv:.3f}, Ratio={amplitude_ratio_deconv:.3f}',
            fontsize=12, fontweight='bold', color='green')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/amplitude_method_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: ../plots/amplitude_method_comparison.png")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
