"""
Visualization Example for Spatial Analysis
==========================================

This example demonstrates how to visualize occupancy maps, rate maps,
and place fields from neural spatial data.
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.utils.spatial import (
    compute_occupancy_map,
    compute_rate_map,
    extract_place_fields,
    analyze_spatial_coding,
)
from driada.experiment.synthetic import (
    generate_2d_random_walk,
    generate_2d_manifold_neurons,
    generate_pseudo_calcium_signal,
)


def visualize_spatial_analysis(n_samples=2000, n_neurons=4, seed=42):
    """
    Generate and visualize spatial data analysis.

    Parameters
    ----------
    n_samples : int
        Number of time samples
    n_neurons : int
        Number of neurons to simulate
    seed : int
        Random seed for reproducibility
    """

    # Generate trajectory
    print("Generating trajectory...")
    positions = generate_2d_random_walk(
        length=n_samples, bounds=(0, 1), step_size=0.02, momentum=0.8, seed=seed
    ).T  # Transpose to (n_samples, 2)

    # Generate place cells
    print(f"Generating {n_neurons} place cells...")
    firing_rates, centers = generate_2d_manifold_neurons(
        n_neurons=n_neurons,
        positions=positions.T,  # Expects (2, n_samples)
        field_sigma=0.1,
        baseline_rate=0.5,  # Hz
        peak_rate=10.0,  # Hz
        noise_std=0.1,
        grid_arrangement=True,  # Arrange in grid
        seed=seed,
    )

    # Convert to calcium signals
    print("Converting to calcium signals...")
    calcium_signals = np.zeros((n_neurons, n_samples))
    for i in range(n_neurons):
        calcium_signals[i] = generate_pseudo_calcium_signal(
            firing_rates[i], sampling_rate=20.0, decay_time=2.0, noise_std=0.5
        )

    # Compute occupancy map
    print("Computing occupancy map...")
    occupancy_map, x_edges, y_edges = compute_occupancy_map(
        positions,
        arena_bounds=((0, 1), (0, 1)),
        bin_size=0.05,  # 20x20 bins
        min_occupancy=0.1,
    )

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Spatial Analysis Visualization", fontsize=16)

    # Trajectory plot
    ax = plt.subplot(3, n_neurons + 1, 1)
    ax.plot(positions[:, 0], positions[:, 1], "k-", alpha=0.3, linewidth=0.5)
    ax.scatter(positions[0, 0], positions[0, 1], c="g", s=50, marker="o", label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], c="r", s=50, marker="s", label="End")
    ax.set_title("Animal Trajectory")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal")
    ax.legend()

    # Occupancy map
    ax = plt.subplot(3, n_neurons + 1, 2)
    im = ax.imshow(
        occupancy_map, origin="lower", cmap="Blues", extent=[0, 1, 0, 1], aspect="equal"
    )
    ax.set_title("Occupancy Map")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    plt.colorbar(im, ax=ax, label="Time (s)")

    # For each neuron
    for i in range(n_neurons):
        # Compute rate map
        rate_map = compute_rate_map(
            calcium_signals[i],
            positions,
            occupancy_map,
            x_edges,
            y_edges,
            smooth_sigma=1.0,
        )

        # Extract place fields
        fields = extract_place_fields(
            rate_map, min_peak_rate=2.0, min_field_size=4, peak_to_mean_ratio=1.3
        )

        # Row 1: Calcium signal colored by position
        ax = plt.subplot(3, n_neurons + 1, 3 + i)
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=calcium_signals[i],
            s=1,
            cmap="hot",
            alpha=0.5,
        )
        ax.plot(
            centers[i, 0],
            centers[i, 1],
            "w*",
            markersize=10,
            markeredgecolor="k",
            markeredgewidth=1,
        )
        ax.set_title(f"Neuron {i+1} Activity")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        # Row 2: Rate maps
        ax = plt.subplot(3, n_neurons + 1, n_neurons + 4 + i)
        im = ax.imshow(
            rate_map, origin="lower", cmap="hot", extent=[0, 1, 0, 1], aspect="equal"
        )
        ax.set_title(f"Rate Map\n{len(fields)} field(s)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Mark detected place field centers
        for field in fields:
            cx, cy = field["center"]
            # Convert from bin indices to spatial coordinates
            x_coord = x_edges[cx]
            y_coord = y_edges[cy]
            ax.plot(x_coord, y_coord, "b+", markersize=8, markeredgewidth=2)

        # Row 3: Calcium traces
        ax = plt.subplot(3, n_neurons + 1, 2 * n_neurons + 5 + i)
        time = np.arange(n_samples) / 20.0  # Convert to seconds
        ax.plot(time[:500], calcium_signals[i, :500], "k-", linewidth=0.5)
        ax.set_title("Calcium Trace (first 25s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F")
        ax.set_xlim(0, 25)

    plt.tight_layout()
    return fig


def main():
    """Run the visualization example."""
    print("Spatial Maps Visualization Example")
    print("=" * 50)

    # Generate and visualize
    fig = visualize_spatial_analysis(n_samples=2000, n_neurons=4, seed=42)

    # Save figure
    fig.savefig("spatial_maps_visualization.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'spatial_maps_visualization.png'")

    # Run full analysis on one neuron
    print("\nRunning full spatial analysis on the neural population...")

    # Generate data again for analysis
    positions = generate_2d_random_walk(2000, seed=42).T
    firing_rates, _ = generate_2d_manifold_neurons(
        4, positions.T, grid_arrangement=True, seed=42
    )

    calcium_signals = np.array(
        [generate_pseudo_calcium_signal(fr, sampling_rate=20.0) for fr in firing_rates]
    )

    # Run analysis
    results = analyze_spatial_coding(
        calcium_signals,
        positions,
        bin_size=0.05,
        min_peak_rate=2.0,
        speed_range=None,  # No speed filtering
        peak_to_mean_ratio=1.3,
    )

    print("\nAnalysis Summary:")
    print(f"  Place cells detected: {results['summary']['n_place_cells']}")
    print(f"  Grid cells detected: {results['summary']['n_grid_cells']}")
    print(
        f"  Mean spatial information: {results['summary']['mean_spatial_info']:.3f} bits/spike"
    )
    print(f"  Spatial decoding R²: {results['decoding_accuracy']['r2_avg']:.3f}")
    print(f"  Spatial MI (total): {results['spatial_mi']['mi_total']:.3f} bits")

    plt.show()


if __name__ == "__main__":
    main()
