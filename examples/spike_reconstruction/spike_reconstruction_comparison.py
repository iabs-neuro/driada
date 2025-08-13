"""
Example: Compare different spike reconstruction methods.

This example demonstrates how to use different spike reconstruction methods
on calcium imaging data and compare their results.
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment import generate_synthetic_exp, reconstruct_spikes


def main():
    # Generate synthetic experiment with calcium signals
    print("Generating synthetic calcium imaging data...")
    exp = generate_synthetic_exp(
        n_dfeats=2, n_cfeats=1, nneurons=5, duration=120, fps=20, seed=42  # 2 minutes
    )

    # Get calcium data
    calcium = exp.calcium
    fps = exp.fps
    time = np.arange(calcium.scdata.shape[1]) / fps

    # Reconstruct spikes using different methods
    print("\nReconstructing spikes using wavelet method...")
    spikes_wavelet, meta_wavelet = reconstruct_spikes(
        calcium, method="wavelet", fps=fps
    )

    print("Reconstructing spikes using threshold method...")
    spikes_threshold, meta_threshold = reconstruct_spikes(
        calcium,
        method="threshold",
        fps=fps,
        params={"threshold_std": 2.5, "smooth_sigma": 2},
    )

    # Visualize results for one neuron
    neuron_idx = 2

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot calcium signal (scaled data)
    ax = axes[0]
    ax.plot(time, calcium.scdata[neuron_idx, :], "k-", linewidth=1)
    ax.set_ylabel("Calcium\n(scaled)")
    ax.set_title(f"Neuron {neuron_idx}: Spike Reconstruction Comparison")
    ax.grid(True, alpha=0.3)

    # Plot wavelet-detected spikes
    ax = axes[1]
    spike_times_wavelet = np.where(spikes_wavelet.data[neuron_idx, :])[0] / fps
    ax.vlines(spike_times_wavelet, 0, 1, colors="blue", alpha=0.7, label="Wavelet")
    ax.set_ylabel("Wavelet\nSpikes")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Plot threshold-detected spikes
    ax = axes[2]
    spike_times_threshold = np.where(spikes_threshold.data[neuron_idx, :])[0] / fps
    ax.vlines(spike_times_threshold, 0, 1, colors="red", alpha=0.7, label="Threshold")
    ax.set_ylabel("Threshold\nSpikes")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("spike_reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved as: spike_reconstruction_comparison.png")
    plt.show()

    # Compare spike statistics
    print("\n" + "=" * 50)
    print("SPIKE DETECTION STATISTICS")
    print("=" * 50)
    print(f"{'Neuron':<10} {'Wavelet':<15} {'Threshold':<15} {'Jaccard':<15}")
    print("-" * 50)

    for i in range(calcium.scdata.shape[0]):
        wavelet_spikes = spikes_wavelet.data[i, :].astype(bool)
        threshold_spikes = spikes_threshold.data[i, :].astype(bool)

        n_wavelet = np.sum(wavelet_spikes)
        n_threshold = np.sum(threshold_spikes)

        # Calculate Jaccard similarity
        intersection = np.sum(wavelet_spikes & threshold_spikes)
        union = np.sum(wavelet_spikes | threshold_spikes)
        jaccard = intersection / union if union > 0 else 0

        print(f"{i:<10} {n_wavelet:<15} {n_threshold:<15} {jaccard:<15.3f}")

    # Show metadata
    print("\n" + "=" * 50)
    print("WAVELET METHOD METADATA")
    print("=" * 50)
    print("Parameters used:")
    for key, value in meta_wavelet["parameters"].items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")
    print(
        f"Total events detected: {sum(len(events) for events in meta_wavelet['start_events'])}"
    )

    print("\n" + "=" * 50)
    print("THRESHOLD METHOD METADATA")
    print("=" * 50)
    print("Parameters used:")
    for key, value in meta_threshold["parameters"].items():
        print(f"  {key}: {value}")
    print(
        f"Total spikes detected: {sum(len(times) for times in meta_threshold['spike_times'])}"
    )


if __name__ == "__main__":
    main()
