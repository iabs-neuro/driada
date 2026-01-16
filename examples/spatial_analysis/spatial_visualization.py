"""
Spatial Data Visualization Example
===================================

This example demonstrates how to visualize neural spatial data:
- Trajectory and occupancy maps (population-level)
- Rate maps (neuron firing vs position)
- Calcium traces (temporal dynamics)

For place cell DETECTION, use INTENSE (MI-based, shuffle-tested).
This example shows VISUALIZATION utilities only.
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic import generate_2d_manifold_data
from driada.utils.spatial import (
    compute_occupancy_map,
    compute_rate_map,
    compute_spatial_information_rate,
)


def create_spatial_context_figure(positions, sampling_rate=20.0):
    """
    Create figure showing trajectory and occupancy.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with 1x2 layout (trajectory, occupancy)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Trajectory
    ax = axes[0]
    ax.plot(positions[:, 0], positions[:, 1], "k-", alpha=0.3, linewidth=0.5)
    ax.scatter(positions[0, 0], positions[0, 1], c="g", s=100, marker="o", label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], c="r", s=100, marker="s", label="End")
    ax.set_title("Animal Trajectory", fontsize=14)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal")
    ax.legend()

    # Right: Occupancy map
    ax = axes[1]
    occupancy_map, _, _ = compute_occupancy_map(
        positions,
        fps=sampling_rate,
        arena_bounds=((0, 1), (0, 1)),
        bin_size=0.05,
        min_occupancy=0.1,
    )
    im = ax.imshow(
        occupancy_map, origin="lower", cmap="Blues", extent=[0, 1, 0, 1], aspect="equal"
    )
    ax.set_title("Occupancy Map", fontsize=14)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    plt.colorbar(im, ax=ax, label="Time (s)")

    plt.tight_layout()
    return fig


def create_neuron_visualization_figure(
    calcium_signals, positions, sampling_rate=20.0, trace_duration=60
):
    """
    Create figure showing rate maps and calcium traces.

    Parameters
    ----------
    trace_duration : float
        Duration of calcium traces to show (seconds)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with 2x4 layout (rate maps in row 1, traces in row 2)
    """
    n_neurons = calcium_signals.shape[0]
    fig, axes = plt.subplots(2, n_neurons, figsize=(16, 8))

    # Compute occupancy once for all neurons
    occupancy_map, x_edges, y_edges = compute_occupancy_map(
        positions,
        fps=sampling_rate,
        arena_bounds=((0, 1), (0, 1)),
        bin_size=0.05,
        min_occupancy=0.1,
    )

    for i in range(n_neurons):
        # Row 1: Rate maps
        ax = axes[0, i]
        rate_map = compute_rate_map(
            calcium_signals[i],
            positions,
            occupancy_map,
            x_edges,
            y_edges,
            fps=sampling_rate,
            smooth_sigma=1.0,
        )

        im = ax.imshow(
            rate_map, origin="lower", cmap="hot", extent=[0, 1, 0, 1], aspect="equal"
        )
        ax.set_title(f"Neuron {i+1} Rate Map", fontsize=12)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        plt.colorbar(im, ax=ax, label="Rate (Hz)")

        # Row 2: Calcium traces (first 60s)
        ax = axes[1, i]
        n_samples = int(trace_duration * sampling_rate)
        time = np.arange(n_samples) / sampling_rate
        ax.plot(time, calcium_signals[i, :n_samples], "k-", linewidth=0.8)
        ax.set_title(f"Neuron {i+1} Calcium", fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dF/F")
        ax.set_xlim(0, trace_duration)

    plt.tight_layout()
    return fig


def main():
    """Run the visualization example."""
    print("Spatial Data Visualization Example")
    print("=" * 60)

    # Generate data
    print("\nGenerating 4 place cells with 200s trajectory...")
    calcium_signals, positions, centers, _ = generate_2d_manifold_data(
        n_neurons=4,
        duration=200,
        fps=20.0,
        field_sigma=0.1,
        baseline_rate=0.05,
        peak_rate=2.0,
        firing_noise=0.05,
        calcium_noise=0.02,
        decay_time=2.0,
        grid_arrangement=True,
        seed=42,
        verbose=False,
    )

    # Convert positions to (n_samples, 2)
    positions = positions.T

    print(f"Data shape: {calcium_signals.shape[0]} neurons, {calcium_signals.shape[1]} samples")
    print(f"Place field centers: {centers}")

    # Compute spatial information metrics
    print("\nComputing spatial information metrics...")
    occupancy_map, x_edges, y_edges = compute_occupancy_map(
        positions,
        fps=20.0,
        arena_bounds=((0, 1), (0, 1)),
        bin_size=0.05,
        min_occupancy=0.1,
    )

    spatial_info = []
    for i in range(calcium_signals.shape[0]):
        rate_map = compute_rate_map(
            calcium_signals[i],
            positions,
            occupancy_map,
            x_edges,
            y_edges,
            fps=20.0,
            smooth_sigma=1.0,
        )
        si = compute_spatial_information_rate(rate_map, occupancy_map)
        spatial_info.append(si)
        print(f"  Neuron {i+1}: {si:.3f} bits/spike")

    print(f"  Mean spatial info: {np.mean(spatial_info):.3f} bits/spike")

    # Create visualizations
    print("\nCreating spatial context figure...")
    fig1 = create_spatial_context_figure(positions, sampling_rate=20.0)
    fig1.savefig("spatial_context.png", dpi=150, bbox_inches="tight")
    print("  Saved: spatial_context.png")

    print("\nCreating neuron visualization figure...")
    fig2 = create_neuron_visualization_figure(
        calcium_signals, positions, sampling_rate=20.0, trace_duration=60
    )
    fig2.savefig("neuron_visualization.png", dpi=150, bbox_inches="tight")
    print("  Saved: neuron_visualization.png")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("\nSpatial information computed using Skaggs et al. (1993) metric.")
    print("For place cell DETECTION, use INTENSE (MI-based analysis).")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
