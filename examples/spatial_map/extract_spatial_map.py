"""
Extract 2D spatial map from place cell population using dimensionality reduction.

This example demonstrates how to:
1. Generate a population of place cells with 2D spatial tuning
2. Apply various dimensionality reduction methods to extract the spatial map
3. Compare different methods (PCA, Isomap, UMAP) for spatial representation
4. Evaluate robustness to noise in neural signals
5. Visualize the extracted spatial representations

The key insight is that place cells form a continuous representation of 2D space,
and population activity should reveal the underlying spatial manifold structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap

# Import DRIADA modules
from driada.experiment import generate_2d_manifold_exp
from driada.dimensionality import (
    pca_dimension,
    effective_rank,
    nn_dimension,
    correlation_dimension,
)
from driada.dim_reduction.manifold_metrics import (
    knn_preservation_rate,
    trustworthiness,
    continuity,
    procrustes_analysis,
)


def estimate_dimensionality(neural_data):
    """
    Estimate intrinsic dimensionality of place cell population.

    Parameters
    ----------
    neural_data : ndarray
        Neural activity matrix (n_neurons x n_timepoints)

    Returns
    -------
    dim_estimates : dict
        Dictionary of dimensionality estimates
    """
    dim_estimates = {}

    # Transpose for methods expecting (n_samples, n_features)
    data_t = neural_data.T

    # Linear methods
    try:
        dim_estimates["pca_90"] = pca_dimension(data_t, threshold=0.90)
        dim_estimates["pca_95"] = pca_dimension(data_t, threshold=0.95)
        dim_estimates["effective_rank"] = effective_rank(data_t)
    except Exception as e:
        print(f"Linear dimension estimation failed: {e}")

    # Nonlinear methods - sample subset for speed
    n_samples = min(1000, data_t.shape[0])
    sample_idx = np.random.choice(data_t.shape[0], n_samples, replace=False)
    data_sample = data_t[sample_idx]

    try:
        dim_estimates["nn_dimension"] = nn_dimension(data_sample, k=5)
    except Exception as e:
        print(f"k-NN dimension failed: {e}")
        dim_estimates["nn_dimension"] = np.nan

    try:
        dim_estimates["correlation_dim"] = correlation_dimension(data_sample, n_bins=20)
    except Exception as e:
        print(f"Correlation dimension failed: {e}")
        dim_estimates["correlation_dim"] = np.nan

    return dim_estimates


def visualize_place_fields(centers, field_sigma):
    """Visualize place field centers in 2D environment."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot place field centers
    ax.scatter(centers[:, 0], centers[:, 1], s=100, alpha=0.6, c="blue")

    # Add circles to show field size
    for center in centers[:20]:  # Show first 20 to avoid clutter
        circle = plt.Circle(
            center, field_sigma, fill=False, edgecolor="blue", alpha=0.3
        )
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    ax.set_title("Place Field Centers (first 20 fields shown)", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    return fig


def apply_dimensionality_reduction(neural_data, true_positions, n_neighbors=30):
    """
    Apply different DR methods and evaluate reconstruction quality.

    Parameters
    ----------
    neural_data : ndarray
        Neural activity (n_neurons x n_timepoints)
    true_positions : ndarray
        True 2D positions (n_timepoints x 2)
    n_neighbors : int
        Number of neighbors for manifold learning methods

    Returns
    -------
    embeddings : dict
        DR embeddings from each method
    metrics : dict
        Quality metrics for each method
    """
    # Transpose data for sklearn format
    data_t = neural_data.T

    embeddings = {}
    metrics = {}

    # PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(data_t)
    embeddings["PCA"] = pca_embedding

    # Compute metrics
    metrics["PCA"] = {
        "explained_var": sum(pca.explained_variance_ratio_[:2]),
        "knn_preservation": knn_preservation_rate(true_positions, pca_embedding, k=10),
        "trustworthiness": trustworthiness(true_positions, pca_embedding, k=10),
        "continuity": continuity(true_positions, pca_embedding, k=10),
    }

    # Procrustes alignment for position reconstruction
    aligned_pca, disparity = procrustes_analysis(true_positions, pca_embedding)
    metrics["PCA"]["procrustes_error"] = disparity

    # Isomap
    print("Applying Isomap...")
    isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
    isomap_embedding = isomap.fit_transform(data_t)
    embeddings["Isomap"] = isomap_embedding

    metrics["Isomap"] = {
        "knn_preservation": knn_preservation_rate(
            true_positions, isomap_embedding, k=10
        ),
        "trustworthiness": trustworthiness(true_positions, isomap_embedding, k=10),
        "continuity": continuity(true_positions, isomap_embedding, k=10),
        "reconstruction_error": isomap.reconstruction_error(),
    }

    aligned_iso, disparity = procrustes_analysis(true_positions, isomap_embedding)
    metrics["Isomap"]["procrustes_error"] = disparity

    # UMAP
    print("Applying UMAP...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
    )
    umap_embedding = reducer.fit_transform(data_t)
    embeddings["UMAP"] = umap_embedding

    metrics["UMAP"] = {
        "knn_preservation": knn_preservation_rate(true_positions, umap_embedding, k=10),
        "trustworthiness": trustworthiness(true_positions, umap_embedding, k=10),
        "continuity": continuity(true_positions, umap_embedding, k=10),
    }

    aligned_umap, disparity = procrustes_analysis(true_positions, umap_embedding)
    metrics["UMAP"]["procrustes_error"] = disparity

    return embeddings, metrics


def visualize_embeddings(embeddings, true_positions, metrics):
    """Visualize DR embeddings colored by true position."""
    from driada.utils.visual import plot_embedding_comparison, DEFAULT_DPI

    # Use position magnitude for coloring
    position_color = np.sqrt(true_positions[:, 0] ** 2 + true_positions[:, 1] ** 2)

    # Create features dict for visual utility
    features = {"position_magnitude": position_color}
    feature_names = {"position_magnitude": "Distance from origin"}

    # Create embedding comparison using visual utility
    fig1 = plot_embedding_comparison(
        embeddings=embeddings,
        features=features,
        feature_names=feature_names,
        with_trajectory=True,
        compute_metrics=True,
        figsize=(18, 15),
        dpi=DEFAULT_DPI,
    )

    # Add metrics text to each subplot
    # This is custom to this example, so we'll modify the figure
    method_list = list(embeddings.keys())
    for i, method in enumerate(method_list):
        # Find the first row subplot for this method
        ax = fig1.axes[i]  # First row axes

        # Add metrics text
        metric_text = f"k-NN: {metrics[method]['knn_preservation']:.3f}\n"
        metric_text += f"Trust: {metrics[method]['trustworthiness']:.3f}\n"
        metric_text += f"Cont: {metrics[method]['continuity']:.3f}"
        if "procrustes_error" in metrics[method]:
            metric_text += f"\nProc: {metrics[method]['procrustes_error']:.1f}"

        ax.text(
            0.02,
            0.98,
            metric_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    return fig1


def test_noise_robustness(neural_data, true_positions, noise_levels):
    """
    Test robustness of DR methods to noise.

    Parameters
    ----------
    neural_data : ndarray
        Clean neural activity (n_neurons x n_timepoints)
    true_positions : ndarray
        True positions (n_timepoints x 2)
    noise_levels : list
        List of noise standard deviations to test

    Returns
    -------
    robustness_results : dict
        Metrics for each method at each noise level
    """
    # Exclude UMAP for speed
    methods = ["PCA", "Isomap"]
    robustness_results = {
        method: {
            "noise_levels": noise_levels,
            "knn_preservation": [],
            "procrustes_error": [],
        }
        for method in methods
    }

    for noise_std in noise_levels:
        print(f"\nTesting noise level: {noise_std}")

        # Add Gaussian noise
        noisy_data = neural_data + np.random.normal(0, noise_std, neural_data.shape)

        # Apply DR methods (excluding UMAP)
        data_t = noisy_data.T

        # PCA
        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(data_t)

        pca_metrics = {
            "knn_preservation": knn_preservation_rate(
                true_positions, pca_embedding, k=10
            ),
            "procrustes_error": procrustes_analysis(true_positions, pca_embedding)[1],
        }

        # Isomap
        isomap = Isomap(n_components=2, n_neighbors=30)
        isomap_embedding = isomap.fit_transform(data_t)

        isomap_metrics = {
            "knn_preservation": knn_preservation_rate(
                true_positions, isomap_embedding, k=10
            ),
            "procrustes_error": procrustes_analysis(true_positions, isomap_embedding)[
                1
            ],
        }

        # Store results
        robustness_results["PCA"]["knn_preservation"].append(
            pca_metrics["knn_preservation"]
        )
        robustness_results["PCA"]["procrustes_error"].append(
            pca_metrics["procrustes_error"]
        )
        robustness_results["Isomap"]["knn_preservation"].append(
            isomap_metrics["knn_preservation"]
        )
        robustness_results["Isomap"]["procrustes_error"].append(
            isomap_metrics["procrustes_error"]
        )

    return robustness_results


def plot_noise_robustness(robustness_results):
    """Plot noise robustness curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method, results in robustness_results.items():
        # k-NN preservation
        ax1.plot(
            results["noise_levels"],
            results["knn_preservation"],
            "o-",
            label=method,
            linewidth=2,
            markersize=8,
        )

        # Procrustes error
        if results["procrustes_error"]:
            ax2.plot(
                results["noise_levels"],
                results["procrustes_error"],
                "o-",
                label=method,
                linewidth=2,
                markersize=8,
            )

    ax1.set_xlabel("Noise Level (std)", fontsize=12)
    ax1.set_ylabel("k-NN Preservation Rate", fontsize=12)
    ax1.set_title("Neighborhood Preservation vs Noise", fontsize=14, pad=20)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    ax2.set_xlabel("Noise Level (std)", fontsize=12)
    ax2.set_ylabel("Procrustes Error", fontsize=12)
    ax2.set_title("Reconstruction Error vs Noise", fontsize=14, pad=20)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0, w_pad=3.0)
    return fig


def main():
    """Main demonstration of spatial map extraction from place cells."""
    print("=" * 70)
    print("SPATIAL MAP EXTRACTION FROM PLACE CELL POPULATIONS")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    print("\n1. Generating place cell population...")

    # Generate synthetic place cells
    fps = 20.0
    exp, info = generate_2d_manifold_exp(
        n_neurons=49,  # 7x7 grid for better coverage
        duration=300,  # Longer duration for better quality
        fps=fps,
        field_sigma=0.15,  # Place field width
        peak_rate=1.0,  # Realistic peak firing rate for calcium imaging
        noise_std=0.05,  # Firing noise
        grid_arrangement=True,
        seed=42,
        verbose=True,
    )

    # Extract data
    neural_data = (
        exp.calcium.scdata
    )  # Shape: (n_neurons, n_timepoints) - scaled data for equal neuron contributions
    positions = info["positions"]  # True positions
    place_centers = info["place_field_centers"]

    print(f"\nGenerated {neural_data.shape[0]} place cells")
    print(f"Recording duration: {neural_data.shape[1]/fps:.1f}s")
    print(f"Trajectory samples: {positions.shape[0]}")

    # Visualize place fields
    print("\n2. Visualizing place field arrangement...")
    field_fig = visualize_place_fields(place_centers, info["field_sigma"])
    plt.savefig("spatial_map_place_fields.png", dpi=150, bbox_inches="tight")

    # Estimate dimensionality
    print("\n3. Estimating intrinsic dimensionality...")
    print("-" * 50)

    dim_estimates = estimate_dimensionality(neural_data)

    print("Dimensionality estimates:")
    for method, estimate in dim_estimates.items():
        if not np.isnan(estimate):
            print(f"  {method:20s}: {estimate:.2f}")

    print("\nNote: Place cells tile 2D space, expect dimensionality ≈ 2")

    # Apply DR methods
    print("\n4. Applying dimensionality reduction methods...")
    print("-" * 50)

    embeddings, metrics = apply_dimensionality_reduction(
        neural_data, positions, n_neighbors=30
    )

    # Print summary metrics
    print("\nReconstruction quality metrics:")
    print(
        f"{'Method':10s} | {'k-NN Pres':10s} | {'Trust':10s} | {'Cont':10s} | {'Procrustes':10s}"
    )
    print("-" * 60)

    for method in ["PCA", "Isomap", "UMAP"]:
        m = metrics[method]
        print(
            f"{method:10s} | {m['knn_preservation']:10.3f} | "
            f"{m['trustworthiness']:10.3f} | {m['continuity']:10.3f} | "
            f"{m.get('procrustes_error', 0):10.3f}"
        )

    # Visualize embeddings
    print("\n5. Visualizing extracted spatial representations...")
    embed_fig = visualize_embeddings(embeddings, positions, metrics)
    plt.savefig("spatial_map_embeddings.png", dpi=150, bbox_inches="tight")

    # Test noise robustness
    print("\n6. Testing robustness to noise...")
    print("-" * 50)

    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    robustness_results = test_noise_robustness(neural_data, positions, noise_levels)

    # Plot robustness
    robust_fig = plot_noise_robustness(robustness_results)
    plt.savefig("spatial_map_noise_robustness.png", dpi=150, bbox_inches="tight")

    # Summary and conclusions
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("- Place cells have intrinsic dimensionality ≈ 2 (2D spatial map)")
    print("- All methods can extract spatial structure from population activity")
    print("- Nonlinear methods (Isomap, UMAP) preserve local structure better")
    print("- PCA captures global variance but may distort local relationships")
    print("- Both PCA and Isomap show graceful degradation with noise")
    print("- Isomap better preserves neighborhood structure under noise")
    print("=" * 70)

    print("\nResults saved to:")
    print("- spatial_map_place_fields.png")
    print("- spatial_map_embeddings.png")
    print("- spatial_map_noise_robustness.png")

    plt.show()


if __name__ == "__main__":
    main()
