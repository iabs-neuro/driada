#!/usr/bin/env python3
"""
Visualize NNLS Event Mask Fix for High-SNR Neurons

Shows before/after comparison of reconstruction quality with event-masked NNLS.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from driada.experiment.neuron import Neuron

def visualize_neuron_reconstruction(neuron_id, with_mask=True, time_window=(0, 600)):
    """Visualize reconstruction for a single neuron."""

    # Load neuron
    data_path = Path("temp/wavelet_analysis/real calcium.npz")
    data = np.load(data_path, allow_pickle=True)
    ca_traces = data['arr_0']
    ca_trace = ca_traces[neuron_id]
    fps = 20.0

    neuron = Neuron(
        cell_id=f"neuron_{neuron_id}",
        ca=ca_trace,
        sp=None,
        fps=fps
    )

    # Run reconstruction
    neuron.reconstruct_spikes(
        method='wavelet',
        amplitude_method='deconvolution',
        iterative=True,
        n_iter=2,
        create_event_regions=with_mask,  # Controls event masking
        verbose=False
    )

    ca_signal = neuron.ca.data
    reconstructed = neuron.reconstructed.data
    event_mask = neuron.events.data > 0 if neuron.events is not None else np.zeros_like(ca_signal, dtype=bool)

    # Get wavelet SNR
    try:
        wavelet_snr = neuron.get_wavelet_snr()
    except:
        wavelet_snr = None

    # Time axis (seconds)
    time = np.arange(len(ca_signal)) / fps

    # Extract time window
    start_frame = int(time_window[0] * fps)
    end_frame = int(time_window[1] * fps)

    time_win = time[start_frame:end_frame]
    ca_win = ca_signal[start_frame:end_frame]
    recon_win = reconstructed[start_frame:end_frame]
    event_win = event_mask[start_frame:end_frame]

    # Calculate peak metrics
    event_starts = np.where(np.diff(event_mask.astype(int)) == 1)[0] + 1
    event_ends = np.where(np.diff(event_mask.astype(int)) == -1)[0] + 1
    if event_mask[0]:
        event_starts = np.concatenate([[0], event_starts])
    if event_mask[-1]:
        event_ends = np.concatenate([event_ends, [len(event_mask)]])

    observed_peaks = []
    recon_peaks = []
    for st, end in zip(event_starts, event_ends):
        if end > st:
            observed_peaks.append(np.max(ca_signal[st:end]))
            recon_peaks.append(np.max(reconstructed[st:end]))

    obs_median = np.median(observed_peaks)
    recon_median = np.median(recon_peaks)
    ratio = recon_median / obs_median

    # Calculate R²
    r2 = neuron.get_reconstruction_r2()

    return {
        'time': time_win,
        'observed': ca_win,
        'reconstructed': recon_win,
        'events': event_win,
        'ratio': ratio,
        'r2': r2,
        'wavelet_snr': wavelet_snr,
        'n_events': len(observed_peaks),
        'obs_median': obs_median,
        'recon_median': recon_median,
    }

def create_comparison_figure(neuron_ids=[192, 100, 200]):
    """Create figure comparing multiple high-SNR neurons."""

    n_neurons = len(neuron_ids)
    fig, axes = plt.subplots(n_neurons, 1, figsize=(14, 4*n_neurons))
    if n_neurons == 1:
        axes = [axes]

    for idx, neuron_id in enumerate(neuron_ids):
        ax = axes[idx]

        print(f"\nProcessing Neuron #{neuron_id}...")
        result = visualize_neuron_reconstruction(neuron_id, with_mask=True)

        # Plot observed signal
        ax.plot(result['time'], result['observed'],
                color='black', linewidth=1.5, alpha=0.7, label='Observed')

        # Plot reconstructed signal
        ax.plot(result['time'], result['reconstructed'],
                color='red', linewidth=1.5, alpha=0.8, label='Reconstructed (event-masked NNLS)')

        # Highlight event regions
        event_regions = np.where(result['events'])[0]
        if len(event_regions) > 0:
            for i in range(len(event_regions)):
                if i == 0 or event_regions[i] != event_regions[i-1] + 1:
                    start = event_regions[i]
                    end = start
                    while end < len(event_regions) - 1 and event_regions[end + 1] == event_regions[end] + 1:
                        end = event_regions[end + 1]
                    ax.axvspan(result['time'][start], result['time'][min(end, len(result['time'])-1)],
                              alpha=0.1, color='blue', label='Event region' if i == 0 else '')

        # Add metrics text
        snr_text = f"SNR={result['wavelet_snr']:.1f}" if result['wavelet_snr'] else "SNR=N/A"
        metrics_text = (
            f"Neuron #{neuron_id} | {snr_text} | Events={result['n_events']}\n"
            f"Peak ratio: {result['ratio']:.3f} ({(1-result['ratio'])*100:.0f}% underest.) | "
            f"R²: {result['r2']:.3f}"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('ΔF/F', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Reconstruction Quality with Event-Masked NNLS',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path('nnls_fix_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    return fig

def create_peak_comparison_figure(neuron_ids=[192, 100, 200]):
    """Create figure showing peak-level comparison."""

    fig, axes = plt.subplots(1, len(neuron_ids), figsize=(6*len(neuron_ids), 5))
    if len(neuron_ids) == 1:
        axes = [axes]

    for idx, neuron_id in enumerate(neuron_ids):
        ax = axes[idx]

        print(f"\nAnalyzing peaks for Neuron #{neuron_id}...")

        # Load and reconstruct
        data_path = Path("temp/wavelet_analysis/real calcium.npz")
        data = np.load(data_path, allow_pickle=True)
        ca_trace = data['arr_0'][neuron_id]

        neuron = Neuron(f"neuron_{neuron_id}", ca_trace, None, fps=20.0)
        neuron.reconstruct_spikes(
            method='wavelet',
            amplitude_method='deconvolution',
            iterative=True,
            n_iter=2,
            create_event_regions=True,
            verbose=False
        )

        # Extract peaks
        ca_signal = neuron.ca.data
        reconstructed = neuron.reconstructed.data
        event_mask = neuron.events.data > 0

        event_starts = np.where(np.diff(event_mask.astype(int)) == 1)[0] + 1
        event_ends = np.where(np.diff(event_mask.astype(int)) == -1)[0] + 1
        if event_mask[0]:
            event_starts = np.concatenate([[0], event_starts])
        if event_mask[-1]:
            event_ends = np.concatenate([event_ends, [len(event_mask)]])

        observed_peaks = []
        recon_peaks = []
        for st, end in zip(event_starts, event_ends):
            if end > st:
                observed_peaks.append(np.max(ca_signal[st:end]))
                recon_peaks.append(np.max(reconstructed[st:end]))

        observed_peaks = np.array(observed_peaks)
        recon_peaks = np.array(recon_peaks)

        # Scatter plot
        ax.scatter(observed_peaks, recon_peaks, alpha=0.6, s=50, color='blue')

        # Perfect reconstruction line
        max_val = max(np.max(observed_peaks), np.max(recon_peaks))
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect (1:1)')

        # Calculate ratio
        ratio = np.median(recon_peaks) / np.median(observed_peaks)

        # Add metrics
        snr = neuron.get_wavelet_snr()
        ax.text(0.05, 0.95,
               f"Neuron #{neuron_id}\nSNR={snr:.1f}\n"
               f"Ratio={ratio:.3f}\n"
               f"Underest.={100*(1-ratio):.0f}%\n"
               f"Events={len(observed_peaks)}",
               transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Observed Peak (ΔF/F)', fontsize=11)
        ax.set_ylabel('Reconstructed Peak (ΔF/F)', fontsize=11)
        ax.set_title(f'Peak Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()

    output_path = Path('nnls_fix_peak_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPeak comparison saved to: {output_path}")

    return fig

if __name__ == '__main__':
    print("Generating reconstruction visualizations with event-masked NNLS fix...\n")
    print("="*80)

    # High-SNR neurons to visualize
    neuron_ids = [192, 100, 200]

    # Create reconstruction time series figure
    print("\n1. Creating reconstruction time series...")
    fig1 = create_comparison_figure(neuron_ids)

    # Create peak comparison figure
    print("\n2. Creating peak comparison scatter plots...")
    fig2 = create_peak_comparison_figure(neuron_ids)

    print("\n" + "="*80)
    print("Visualizations complete!")
    print("\nGenerated files:")
    print("  - nnls_fix_visualization.png (time series)")
    print("  - nnls_fix_peak_comparison.png (peak scatter)")

    plt.show()
