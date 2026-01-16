"""
Extract circular manifold from head direction cells using dimensionality reduction.

This example demonstrates how to:
1. Generate a population of head direction cells with circular tuning
2. Estimate the intrinsic dimensionality of the neural population
3. Apply various dimensionality reduction methods to extract the underlying circular structure
4. Validate the reconstruction against ground truth head direction
5. Visualize the extracted manifold

The key insight is that head direction cells form a ring attractor network,
and their population activity should lie on a circular (1D) manifold embedded in
high-dimensional neural space.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import DRIADA modules
from driada.experiment import generate_circular_manifold_exp
from driada.dimensionality import (
    eff_dim,
    correlation_dimension,
    geodesic_dimension,
    pca_dimension,
)
from driada.dim_reduction.manifold_metrics import compute_embedding_alignment_metrics
from driada.dim_reduction import MVData
import os


def estimate_dimensionality(neural_data, methods=None, ds=1):
    """
    Estimate intrinsic dimensionality using multiple methods from DRIADA.

    Parameters
    ----------
    neural_data : ndarray
        Neural activity matrix (n_neurons x n_timepoints)

    methods : list of str, optional
        List of dimensionality estimation methods to use. If None, uses all available.
        Available methods:
        - 'pca_90': PCA dimension for 90% variance explained
        - 'pca_95': PCA dimension for 95% variance explained
        - 'participation_ratio': Participation ratio (quadratic Renyi entropy)
        - 'correlation_dim': Correlation dimension (Grassberger-Procaccia)
        - 'geodesic_dim': Geodesic distance based estimator

    ds : int, default=1
        Downsampling factor. Use ds=5 for faster computation on long recordings.

    Returns
    -------
    dim_estimates : dict
        Dictionary of dimensionality estimates from each method
    """
    # Default to all methods if none specified
    if methods is None:
        methods = [
            "pca_90",
            "pca_95",
            "participation_ratio",
            "correlation_dim",
            "geodesic_dim",
        ]

    dim_estimates = {}

    # Downsample data if requested
    if ds > 1:
        neural_data_ds = neural_data[:, ::ds]
        print(f"  Downsampled: {neural_data.shape} -> {neural_data_ds.shape}")
    else:
        neural_data_ds = neural_data

    # Transpose data for methods that expect (n_samples, n_features)
    data_transposed = neural_data_ds.T

    # Linear methods
    if "pca_90" in methods:
        dim_estimates["pca_90"] = pca_dimension(data_transposed, threshold=0.90)

    if "pca_95" in methods:
        dim_estimates["pca_95"] = pca_dimension(data_transposed, threshold=0.95)

    # Nonlinear intrinsic methods
    if "correlation_dim" in methods:
        try:
            print("  Computing correlation dimension...")
            dim_estimates["correlation_dim"] = correlation_dimension(data_transposed)
        except Exception as e:
            print(f"  Warning: correlation_dimension failed: {e}")
            dim_estimates["correlation_dim"] = np.nan

    if "geodesic_dim" in methods:
        try:
            print("  Computing geodesic dimension (this may take time)...")
            dim_estimates["geodesic_dim"] = geodesic_dimension(
                data_transposed, k=20, mode="fast", factor=4
            )
        except Exception as e:
            print(f"  Warning: geodesic_dimension failed: {e}")
            dim_estimates["geodesic_dim"] = np.nan

    # Effective dimensionality (participation ratio)
    if "participation_ratio" in methods:
        dim_estimates["participation_ratio"] = eff_dim(
            neural_data_ds.T, enable_correction=False, q=2
        )

    return dim_estimates


def plot_eigenspectrum(neural_data):
    """Plot eigenvalue spectrum of correlation matrix."""
    # Compute correlation matrix
    data_centered = neural_data - np.mean(neural_data, axis=1, keepdims=True)
    corr_mat = np.corrcoef(data_centered)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(corr_mat)[::-1]  # Descending order
    eigenvalues = eigenvalues[eigenvalues > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Eigenvalue spectrum
    ax1.plot(eigenvalues, "o-", markersize=4)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Eigenvalue Spectrum")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Cumulative variance explained
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    ax2.plot(cumvar, "o-", markersize=4)
    ax2.axhline(0.9, color="r", linestyle="--", label="90% variance")
    ax2.axhline(0.95, color="orange", linestyle="--", label="95% variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_circular_coordinates(embedding):
    """Convert 2D embedding to circular coordinates (angles)."""
    # Center the embedding
    centered = embedding - np.mean(embedding, axis=0)

    # Compute angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    angles = np.mod(angles, 2 * np.pi)  # Ensure [0, 2pi]

    return angles


def visualize_manifold_extraction(embeddings, true_angles, method_names):
    """Visualize extracted manifolds from different methods."""
    n_methods = len(embeddings)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))

    if n_methods == 1:
        axes = axes.reshape(2, 1)

    for i, (embedding, method) in enumerate(zip(embeddings, method_names)):
        # Plot 2D embedding colored by true angle
        ax = axes[0, i]
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=true_angles, cmap="hsv", s=20, alpha=0.7
        )
        ax.set_title(f"{method} Embedding")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        # Add colorbar
        if i == n_methods - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("True head direction (rad)")

        # Evaluate using manifold metrics API to get optimal alignment
        alignment_metrics = compute_embedding_alignment_metrics(
            embedding, true_angles, "circular"
        )
        error = alignment_metrics["error"]
        correlation = alignment_metrics["correlation"]
        rotation_offset = alignment_metrics["rotation_offset"]
        is_reflected = alignment_metrics["is_reflected"]

        # Extract angles and apply optimal transformation
        recon_angles = compute_circular_coordinates(embedding)

        # Apply the optimal transformation found by the metrics
        if is_reflected:
            recon_angles = -recon_angles
        recon_angles = recon_angles + rotation_offset

        # Plot true vs reconstructed angles with proper wrapping
        ax = axes[1, i]

        # Wrap angles to [0, 2pi] for visualization
        true_wrapped = np.mod(true_angles, 2 * np.pi)
        recon_wrapped = np.mod(recon_angles, 2 * np.pi)

        # Handle wraparound by plotting points near boundaries twice
        # This creates continuous visualization across the circular boundary
        threshold = 0.5  # radians from boundary

        # Find points near 0/2pi boundary
        near_zero_true = true_wrapped < threshold
        near_2pi_true = true_wrapped > (2 * np.pi - threshold)
        near_zero_recon = recon_wrapped < threshold
        near_2pi_recon = recon_wrapped > (2 * np.pi - threshold)

        # Main scatter plot
        ax.scatter(true_wrapped, recon_wrapped, alpha=0.5, s=10, color="blue")

        # Plot wrapped copies for continuity
        # Points with true angle near 0 and recon near 2pi
        mask1 = near_zero_true & near_2pi_recon
        if np.any(mask1):
            ax.scatter(
                true_wrapped[mask1],
                recon_wrapped[mask1] - 2 * np.pi,
                alpha=0.5,
                s=10,
                color="blue",
            )

        # Points with true angle near 2pi and recon near 0
        mask2 = near_2pi_true & near_zero_recon
        if np.any(mask2):
            ax.scatter(
                true_wrapped[mask2],
                recon_wrapped[mask2] + 2 * np.pi,
                alpha=0.5,
                s=10,
                color="blue",
            )

        # Points with true angle near 0, show at 2pi too
        mask3 = near_zero_true & near_zero_recon
        if np.any(mask3):
            ax.scatter(
                true_wrapped[mask3] + 2 * np.pi,
                recon_wrapped[mask3] + 2 * np.pi,
                alpha=0.5,
                s=10,
                color="blue",
            )

        # Points with true angle near 2pi, show at 0 too
        mask4 = near_2pi_true & near_2pi_recon
        if np.any(mask4):
            ax.scatter(
                true_wrapped[mask4] - 2 * np.pi,
                recon_wrapped[mask4] - 2 * np.pi,
                alpha=0.5,
                s=10,
                color="blue",
            )

        # Reference lines
        ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "r--", alpha=0.5, label="y=x")
        # Continuation lines for wraparound
        # When x goes from 2pi to 0, y should also go from 2pi to 0
        ax.plot([2 * np.pi, 2 * np.pi], [2 * np.pi, 2 * np.pi + 0.5], "r--", alpha=0.5)
        ax.plot([0, 0], [-0.5, 0], "r--", alpha=0.5)
        # And vice versa
        ax.plot([2 * np.pi, 2 * np.pi + 0.5], [2 * np.pi, 2 * np.pi], "r--", alpha=0.5)
        ax.plot([-0.5, 0], [0, 0], "r--", alpha=0.5)

        ax.set_xlabel("True angle (rad)")
        ax.set_ylabel("Reconstructed angle (rad)")
        ax.set_title(f"r = {correlation:.3f}, error = {error:.3f} rad")
        ax.set_xlim([-0.5, 2 * np.pi + 0.5])
        ax.set_ylim([-0.5, 2 * np.pi + 0.5])

        # Add grid lines at 0 and 2pi
        ax.axvline(0, color="gray", alpha=0.3, linestyle=":")
        ax.axvline(2 * np.pi, color="gray", alpha=0.3, linestyle=":")
        ax.axhline(0, color="gray", alpha=0.3, linestyle=":")
        ax.axhline(2 * np.pi, color="gray", alpha=0.3, linestyle=":")

    plt.tight_layout()
    return fig


def main():
    """Main demonstration of circular manifold extraction."""
    print("=" * 70)
    print("CIRCULAR MANIFOLD EXTRACTION FROM HEAD DIRECTION CELLS")
    print("=" * 70)

    # Create output directory for results
    os.makedirs("circular_manifold_results", exist_ok=True)

    print("\n1. Generating head direction cell population...")

    # Generate synthetic head direction cells
    exp, info = generate_circular_manifold_exp(
        n_neurons=100,
        duration=600,  # 10 minutes
        kappa=4.0,  # Tuning width
        seed=42,
        verbose=True,
        return_info=True,
    )

    # Extract neural activity and true head directions
    neural_data = exp.calcium.scdata  # Shape: (n_neurons, n_timepoints) - scaled data
    true_angles = info["head_direction"]  # Ground truth angles

    print(
        f"\nGenerated {neural_data.shape[0]} neurons, {neural_data.shape[1]} timepoints"
    )
    print(f"Neural activity shape: {neural_data.shape}")

    # Estimate intrinsic dimensionality
    print("\n2. Estimating intrinsic dimensionality of neural population...")
    print("-" * 50)

    # Use all available dimensionality estimation methods
    dim_methods = [
        "pca_90",
        "pca_95",
        "participation_ratio",
        "correlation_dim",
        "geodesic_dim",
    ]

    # Use ds=5 downsampling for faster computation
    dim_estimates = estimate_dimensionality(neural_data, methods=dim_methods, ds=5)

    print("Dimensionality estimates:")
    for method, estimate in dim_estimates.items():
        print(f"  {method:20s}: {estimate:.2f}")

    print("\nNote: Head direction cells should have intrinsic dimensionality ~ 1")
    print("      (circular manifold), but finite sampling may increase estimates")

    # Compare with temporally shuffled data to demonstrate manifold structure
    print("\n2b. Comparing with temporally shuffled data (destroys manifold)...")
    print("-" * 50)

    # Get shuffled calcium data from experiment
    shuffled_calcium = exp.get_multicell_shuffled_calcium()

    # Estimate dimensionality on shuffled data (same ds=5 downsampling)
    dim_estimates_shuffled = estimate_dimensionality(shuffled_calcium, methods=dim_methods, ds=5)

    print("\nDimensionality estimates (SHUFFLED data):")
    for method, estimate in dim_estimates_shuffled.items():
        print(f"  {method:20s}: {estimate:.2f}")

    print("\nComparison (Real vs Shuffled):")
    print(f"{'Method':<20s} {'Real':>8s} {'Shuffled':>8s} {'Increase':>10s}")
    print("-" * 50)
    for method in dim_methods:
        real = dim_estimates[method]
        shuffled = dim_estimates_shuffled[method]
        increase = ((shuffled - real) / real) * 100
        print(f"{method:<20s} {real:8.2f} {shuffled:8.2f} {increase:+9.1f}%")

    print("\nInterpretation: Temporal shuffling destroys the circular manifold structure,")
    print("                dramatically increasing dimensionality. This confirms that the")
    print("                low dimensionality in real data reflects true manifold structure.")

    # Plot eigenspectrum
    print("\n3. Plotting eigenvalue spectrum...")
    eigen_fig = plot_eigenspectrum(neural_data)
    plt.savefig(
        "circular_manifold_results/eigenspectrum.png", dpi=150, bbox_inches="tight"
    )

    # Apply dimensionality reduction using MVData
    print("\n4. Applying dimensionality reduction methods using MVData...")
    print("-" * 50)

    # Create MVData object from calcium data with downsampling
    # This helps with computational efficiency and smooths the data
    downsampling = 10
    mvdata = MVData(neural_data, downsampling=downsampling)

    # Downsample true angles to match
    true_angles_ds = true_angles[::downsampling]

    # Dictionary to store embeddings
    embeddings_dict = {}

    # PCA
    print("- PCA...")
    pca_embedding = mvdata.get_embedding(method="pca", dim=2)
    embeddings_dict["PCA"] = (
        pca_embedding.coords.T
    )  # Transpose to get (n_samples, n_components)
    print(
        f"  First 2 PCs explain {100*sum(pca_embedding.reducer_.explained_variance_ratio_):.1f}% of variance"
    )

    # Isomap
    print("- Isomap...")
    isomap_embedding = mvdata.get_embedding(method="isomap", dim=2, n_neighbors=50)
    embeddings_dict["Isomap"] = isomap_embedding.coords.T

    # UMAP with increased parameters for better global structure
    print("- UMAP...")
    umap_embedding = mvdata.get_embedding(
        method="umap", n_components=2, n_neighbors=100, min_dist=0.5
    )
    embeddings_dict["UMAP"] = umap_embedding.coords.T

    # Visualize results
    print("\n5. Visualizing extracted manifolds...")
    from driada.utils.visual import DEFAULT_DPI

    # Create embedding comparison visualization
    # Shows embeddings colored by head direction (top row) and angle reconstruction (bottom row)
    embeddings_list = [embeddings_dict[method] for method in ["PCA", "Isomap", "UMAP"]]
    fig_embedding = visualize_manifold_extraction(
        embeddings_list, true_angles_ds, ["PCA", "Isomap", "UMAP"]
    )
    plt.savefig(
        "circular_manifold_results/embedding_comparison.png",
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )

    # Additional analysis: temporal continuity
    print("\n6. Analyzing temporal continuity of extracted manifolds...")
    from driada.utils.visual import plot_trajectories

    # Use only first 1000 timepoints for trajectory visualization
    traj_len = min(1000, embeddings_dict["PCA"].shape[0])
    trajectories_dict = {
        method: emb[:traj_len] for method, emb in embeddings_dict.items()
    }

    fig3 = plot_trajectories(
        embeddings=trajectories_dict,
        trajectory_kwargs={"arrow_spacing": 50, "linewidth": 0.5, "alpha": 0.5},
        figsize=(15, 5),
        save_path="circular_manifold_results/trajectories.png",
        dpi=DEFAULT_DPI,
    )

    # Summary statistics
    print("\n7. Summary of manifold extraction quality:")
    print("-" * 60)
    print(f"{'Method':10s} | {'Correlation':12s} | {'Mean Error':12s} | {'Quality':8s}")
    print("-" * 60)

    for method, embedding in embeddings_dict.items():
        # Use manifold metrics API
        alignment_metrics = compute_embedding_alignment_metrics(
            embedding, true_angles_ds, "circular"
        )
        r = alignment_metrics["correlation"]
        error = alignment_metrics["error"]

        # Quality assessment
        if abs(r) > 0.95:
            quality_str = "Excellent"
        elif abs(r) > 0.85:
            quality_str = "Good"
        elif abs(r) > 0.70:
            quality_str = "Fair"
        else:
            quality_str = "Poor"

        print(f"{method:10s} | {r:12.3f} | {error:9.3f} rad | {quality_str:8s}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print("- Head direction cells have low intrinsic dimensionality (~1-2)")
    print("- Temporal shuffling destroys manifold structure (dimensionality increases)")
    print("- Nonlinear methods (Isomap, UMAP) better preserve circular topology")
    print("- PCA captures variance but may distort circular structure")
    print("- Higher n_neighbors helps preserve global structure")
    print("=" * 70)

    print("\nResults saved to circular_manifold_results/:")
    print("- eigenspectrum.png")
    print("- embedding_comparison.png")
    print("- trajectories.png")

    plt.show()


if __name__ == "__main__":
    main()
