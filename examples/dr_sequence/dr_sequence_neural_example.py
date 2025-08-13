"""
Example comparing direct LE vs PCA→LE sequence on synthetic neural data.

This demonstrates how sequential dimensionality reduction can improve
manifold learning on high-dimensional neural activity data.
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.synthetic.manifold_spatial_2d import generate_2d_manifold_data
from driada.dim_reduction import MVData, dr_sequence
from driada.dim_reduction.manifold_metrics import (
    knn_preservation_rate,
    trustworthiness,
    continuity,
    manifold_preservation_score,
)
from driada.utils.visual import plot_embedding_comparison


def compute_manifold_metrics(true_positions, embedding_coords, k=10):
    """Compute various manifold preservation metrics."""
    metrics = {}

    # KNN preservation
    metrics["knn_preservation"] = knn_preservation_rate(
        true_positions.T,  # (n_samples, n_dims)
        embedding_coords.T,  # (n_samples, n_dims)
        k=k,
    )

    # Trustworthiness and continuity
    metrics["trustworthiness"] = trustworthiness(
        true_positions.T, embedding_coords.T, k=k
    )

    metrics["continuity"] = continuity(true_positions.T, embedding_coords.T, k=k)

    # Overall manifold preservation score (returns dict)
    manifold_scores = manifold_preservation_score(
        true_positions.T, embedding_coords.T, k_neighbors=k
    )
    # Extract the overall score
    metrics["manifold_score"] = manifold_scores["overall_score"]

    return metrics


def main():
    print("Generating synthetic neural data from 2D spatial environment...")

    # Generate synthetic data with 2D spatial manifold
    calcium, positions, place_field_centers, firing_rates = generate_2d_manifold_data(
        n_neurons=100,
        duration=200,  # seconds
        sampling_rate=20.0,  # Hz
        field_sigma=0.1,
        step_size=0.02,
        seed=123,  # Changed seed
        verbose=True,
    )

    # Extract neural activity and true positions
    neural_data = calcium  # (n_neurons, n_timepoints)
    true_positions = positions  # (2, n_timepoints)

    print(f"Neural data shape: {neural_data.shape}")
    print(f"True positions shape: {true_positions.shape}")

    # Create MVData object with downsampling
    mvdata = MVData(neural_data, downsampling=5)

    # Approach 1: Direct LE on all neurons
    print("\n=== Approach 1: Direct LE ===")
    embedding_direct = mvdata.get_embedding(method="le", dim=2, n_neighbors=20)

    # Approach 2: PCA → LE sequence
    print("\n=== Approach 2: PCA → LE ===")
    embedding_sequence = dr_sequence(
        mvdata,
        steps=[
            ("pca", {"dim": 10}),  # First reduce to 10 PCs
            ("le", {"dim": 2, "n_neighbors": 20}),  # Then apply LE
        ],
    )

    # Compute manifold preservation metrics
    print("\n=== Manifold Preservation Metrics ===")

    # Downsample true positions to match the embeddings
    true_positions_ds = true_positions[:, ::5]

    metrics_direct = compute_manifold_metrics(
        true_positions_ds, embedding_direct.coords
    )

    metrics_sequence = compute_manifold_metrics(
        true_positions_ds, embedding_sequence.coords
    )

    print("\nDirect LE:")
    for name, value in metrics_direct.items():
        print(f"  {name}: {value:.4f}")

    print("\nPCA → LE:")
    for name, value in metrics_sequence.items():
        print(f"  {name}: {value:.4f}")

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # Plot true positions
    ax1 = plt.subplot(131)
    scatter = ax1.scatter(
        true_positions_ds[0],
        true_positions_ds[1],
        c=np.arange(true_positions_ds.shape[1]),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax1.set_title("True 2D Positions")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.set_aspect("equal")

    # Plot direct LE embedding
    ax2 = plt.subplot(132)
    ax2.scatter(
        embedding_direct.coords[0],
        embedding_direct.coords[1],
        c=np.arange(embedding_direct.coords.shape[1]),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax2.set_title(
        f'Direct LE\n(Manifold score: {metrics_direct["manifold_score"]:.3f})'
    )
    ax2.set_xlabel("LE 1")
    ax2.set_ylabel("LE 2")
    ax2.set_aspect("equal")

    # Plot PCA→LE embedding
    ax3 = plt.subplot(133)
    ax3.scatter(
        embedding_sequence.coords[0],
        embedding_sequence.coords[1],
        c=np.arange(embedding_sequence.coords.shape[1]),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax3.set_title(
        f'PCA → LE\n(Manifold score: {metrics_sequence["manifold_score"]:.3f})'
    )
    ax3.set_xlabel("LE 1")
    ax3.set_ylabel("LE 2")
    ax3.set_aspect("equal")

    plt.colorbar(scatter, ax=[ax1, ax2, ax3], label="Time", fraction=0.02)
    plt.tight_layout()
    plt.savefig("dr_sequence_neural_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print improvement percentages
    print("\n=== Method Comparison ===")
    print("PCA → LE improves over Direct LE:")
    for metric in ["knn_preservation", "trustworthiness", "continuity"]:
        improvement = (
            (metrics_sequence[metric] - metrics_direct[metric])
            / metrics_direct[metric]
            * 100
        )
        print(f"  {metric}: {improvement:+.1f}%")

    # Create a detailed comparison plot using DRIADA's plotting utilities
    # Prepare embeddings for plot_embedding_comparison
    embeddings_dict = {
        "Direct LE": embedding_direct.coords.T,  # (n_samples, 2)
        "PCA → LE": embedding_sequence.coords.T,  # (n_samples, 2)
    }

    # Create features for coloring
    time_points = np.arange(true_positions_ds.shape[1])
    x_pos = true_positions_ds[0]
    y_pos = true_positions_ds[1]

    features_dict = {"time": time_points, "x_position": x_pos, "y_position": y_pos}

    # Use plot_embedding_comparison for comprehensive visualization
    fig_comparison = plot_embedding_comparison(
        embeddings=embeddings_dict,
        features=features_dict,
        feature_names={
            "time": "Time",
            "x_position": "X Position",
            "y_position": "Y Position",
        },
        with_trajectory=False,  # No trajectory plots
        compute_metrics=False,  # We already computed metrics
        figsize=(15, 10),
        save_path="dr_sequence_embedding_comparison.png",
    )

    # Create metrics comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot true positions for reference
    ax = axes[0]
    scatter = ax.scatter(
        true_positions_ds[0],
        true_positions_ds[1],
        c=time_points,
        cmap="viridis",
        s=20,
        alpha=0.6,
    )
    # Remove trajectory plot
    ax.set_title("True 2D Positions")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="Time")

    # Summary metrics comparison
    ax = axes[1]
    metrics_names = list(metrics_direct.keys())
    x = np.arange(len(metrics_names))
    width = 0.35

    values_direct = [metrics_direct[m] for m in metrics_names]
    values_sequence = [metrics_sequence[m] for m in metrics_names]

    ax.bar(x - width / 2, values_direct, width, label="Direct LE", alpha=0.8)
    ax.bar(x + width / 2, values_sequence, width, label="PCA → LE", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Manifold Preservation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metrics_names], rotation=45, ha="right"
    )
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("dr_sequence_detailed_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print improvement summary
    print("\n=== Improvement Summary ===")
    for metric in metrics_names:
        improvement = (
            (metrics_sequence[metric] - metrics_direct[metric])
            / metrics_direct[metric]
            * 100
        )
        print(f"{metric}: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
