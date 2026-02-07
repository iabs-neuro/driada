"""
Example: Compare different spike reconstruction methods.

This example demonstrates how to use different spike reconstruction methods
on calcium imaging data and compare their results.

Both methods detect calcium transient regions (event start to end) but use
different signal processing: wavelet uses CWT ridge detection while threshold
uses MAD-based signal crossing. Both use iterative detection (n_iter=3) to
catch overlapping events. Comparing event overlap at varying tolerance reveals
timing differences between detection mechanisms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from driada.experiment import generate_synthetic_exp


def main():
    # Generate synthetic experiment with calcium signals
    print("Generating synthetic calcium imaging data...")
    exp = generate_synthetic_exp(
        n_dfeats=2, n_cfeats=1, nneurons=5, duration=120, fps=20, seed=42  # 2 minutes
    )

    calcium = exp.calcium
    fps = exp.fps
    n_neurons = calcium.scdata.shape[0]
    time = np.arange(calcium.scdata.shape[1]) / fps

    # Both methods use Neuron-level iterative reconstruction (n_iter=3)
    # to catch overlapping events via residual analysis.
    wavelet_events = []
    threshold_events = []

    for neuron in exp.neurons:
        # Wavelet: CWT ridge detection on residuals
        print(f"  Neuron {neuron.cell_id}: wavelet...", end="")
        neuron.reconstruct_spikes(
            method="wavelet", iterative=True, n_iter=3, fps=fps
        )
        wavelet_events.append(list(neuron.wvt_ridges))

        # Threshold: MAD-based event detection on residuals
        print(" threshold...", end="")
        neuron.reconstruct_spikes(
            method="threshold", iterative=True, n_iter=3, n_mad=4.0,
            adaptive_thresholds=True, fps=fps,
        )
        threshold_events.append(list(neuron.threshold_events))
        print(
            f" done ({len(wavelet_events[-1])} / {len(threshold_events[-1])} events)"
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

    # Plot wavelet-detected event regions
    ax = axes[1]
    for ev in wavelet_events[neuron_idx]:
        ax.axvspan(ev.start / fps, ev.end / fps, alpha=0.5, color="blue")
    ax.set_ylabel("Wavelet\nEvents")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[Patch(facecolor="blue", alpha=0.5, label="Event region")],
        loc="upper right",
    )

    # Plot threshold-detected event regions
    ax = axes[2]
    for ev in threshold_events[neuron_idx]:
        ax.axvspan(ev.start / fps, ev.end / fps, alpha=0.5, color="red")
    ax.set_ylabel("Threshold\nEvents")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[Patch(facecolor="red", alpha=0.5, label="Event region")],
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig("spike_reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved as: spike_reconstruction_comparison.png")
    plt.close()

    # Event counts per neuron
    print("\n" + "=" * 50)
    print("EVENT COUNTS (iterative, n_iter=3)")
    print("=" * 50)
    print(f"{'Neuron':<10} {'Wavelet':<14} {'Threshold':<14}")
    print("-" * 50)
    for i in range(n_neurons):
        n_w = len(wavelet_events[i])
        n_t = len(threshold_events[i])
        print(f"{i:<10} {n_w:<14} {n_t:<14}")
    total_w = sum(len(wavelet_events[i]) for i in range(n_neurons))
    total_t = sum(len(threshold_events[i]) for i in range(n_neurons))
    print("-" * 50)
    print(f"{'Total':<10} {total_w:<14} {total_t:<14}")

    # Agreement vs tolerance curve
    # Both methods detect event regions via iterative residual analysis.
    # Sweeping tolerance reveals how well the detected regions align.
    tolerance_sec = np.arange(0, 1.05, 0.05)
    tolerance_frames_arr = (tolerance_sec * fps).astype(int)

    # Extract start/end arrays
    w_starts = [[int(e.start) for e in wavelet_events[i]] for i in range(n_neurons)]
    w_ends = [[int(e.end) for e in wavelet_events[i]] for i in range(n_neurons)]
    t_starts = [[int(e.start) for e in threshold_events[i]] for i in range(n_neurons)]
    t_ends = [[int(e.end) for e in threshold_events[i]] for i in range(n_neurons)]

    agreements = []
    for tol in tolerance_frames_arr:
        matched = 0
        for i in range(n_neurons):
            for ws, we in zip(w_starts[i], w_ends[i]):
                # Check if any threshold event overlaps this wavelet event
                for ts, te in zip(t_starts[i], t_ends[i]):
                    if ts <= (we + tol) and te >= (ws - tol):
                        matched += 1
                        break
        agreements.append(matched / total_w if total_w > 0 else 0)

    print("\n" + "=" * 50)
    print("AGREEMENT VS TOLERANCE")
    print("=" * 50)
    print(f"{'Tolerance (s)':<16} {'Matched':<12} {'Agreement':<12}")
    print("-" * 50)
    for tol_s, agr in zip(tolerance_sec, agreements):
        if tol_s % 0.25 < 0.01 or abs(tol_s % 0.25 - 0.25) < 0.01:
            print(f"{tol_s:<16.2f} {int(agr * total_w):<12} {agr:<12.1%}")

    # Plot tolerance curve
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        tolerance_sec, [a * 100 for a in agreements], "ko-", linewidth=2, markersize=4
    )
    ax2.set_xlabel("Tolerance (s)")
    ax2.set_ylabel("Agreement (%)")
    ax2.set_title("Event-Level Agreement: Wavelet vs Threshold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("spike_reconstruction_tolerance.png", dpi=150, bbox_inches="tight")
    print("\nTolerance curve saved as: spike_reconstruction_tolerance.png")
    plt.close()


if __name__ == "__main__":
    main()
