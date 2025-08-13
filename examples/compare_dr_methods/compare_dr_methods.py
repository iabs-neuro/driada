"""
Systematic comparison of dimensionality reduction methods in DRIADA.

This example provides a comprehensive comparison of all available DR methods on various
test datasets, evaluating both computational performance and embedding quality. The results
include practical recommendations for choosing the right method for different use cases.

Key features:
1. Tests on multiple synthetic and real datasets
2. Evaluates quality metrics (k-NN preservation, trustworthiness, continuity, stress)
3. Measures computational performance (runtime, memory usage)
4. Provides visualizations and recommendations

Usage:
    python compare_dr_methods.py [--quick]

    Use --quick flag for a faster comparison with reduced dataset sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs
from sklearn.preprocessing import StandardScaler
import time
import tracemalloc
import warnings
from typing import Dict, Tuple
import pandas as pd

# Import DRIADA modules
from driada.dim_reduction import (
    MVData,
    knn_preservation_rate,
    trustworthiness,
    continuity,
    stress,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def generate_test_datasets(
    n_samples: int = 1000, noise: float = 0.0, seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate various test datasets for DR method comparison.

    Parameters
    ----------
    n_samples : int
        Number of samples per dataset
    noise : float
        Noise level to add to datasets
    seed : int
        Random seed for reproducibility

    Returns
    -------
    datasets : dict
        Dictionary mapping dataset names to (data, labels) tuples
    """
    np.random.seed(seed)
    datasets = {}

    # 1. Swiss Roll - classic nonlinear manifold
    print("Generating Swiss Roll...")
    X_swiss, color_swiss = make_swiss_roll(
        n_samples=n_samples, noise=noise, random_state=seed
    )
    datasets["swiss_roll"] = (X_swiss, color_swiss)

    # 2. S-Curve - another nonlinear manifold
    print("Generating S-Curve...")
    X_scurve, color_scurve = make_s_curve(
        n_samples=n_samples, noise=noise, random_state=seed
    )
    datasets["s_curve"] = (X_scurve, color_scurve)

    # 3. Circular manifold - tests circular topology
    print("Generating Circular manifold...")
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    circle_3d = np.column_stack(
        [
            np.cos(angles),
            np.sin(angles),
            0.1 * np.random.randn(n_samples),  # Small noise in 3rd dimension
        ]
    )
    datasets["circle_3d"] = (circle_3d, angles)

    # 4. High-dimensional Gaussian - tests linear methods
    print("Generating High-D Gaussian...")
    # Create data with intrinsic dimension ~5 embedded in 50D
    n_features = 50
    intrinsic_dim = 5
    # Generate low-rank data
    U = np.random.randn(n_features, intrinsic_dim)
    V = np.random.randn(intrinsic_dim, n_samples)
    X_gaussian = U @ V + noise * np.random.randn(n_features, n_samples)
    X_gaussian = X_gaussian.T  # Shape: (n_samples, n_features)
    # Color by first principal component
    color_gaussian = X_gaussian @ U[:, 0]
    datasets["gaussian_50d"] = (X_gaussian, color_gaussian)

    # 5. Clustered data - tests local structure preservation
    print("Generating Clustered data...")
    X_clusters, y_clusters = make_blobs(
        n_samples=n_samples,
        n_features=20,
        centers=5,
        cluster_std=0.5,
        random_state=seed,
    )
    datasets["clusters_20d"] = (X_clusters, y_clusters)

    # 6. Noisy sphere - tests manifold learning with noise
    print("Generating Noisy sphere...")
    # Generate points on unit sphere
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    theta = np.random.uniform(0, np.pi, n_samples)
    sphere_3d = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    # Add noise
    sphere_3d += noise * np.random.randn(n_samples, 3)
    # Color by azimuthal angle
    datasets["sphere_3d"] = (sphere_3d, phi)

    return datasets


def get_dr_method_configs() -> Dict[str, Dict]:
    """
    Get configuration parameters for each DR method using simplified API.

    Returns
    -------
    configs : dict
        Dictionary mapping method names to parameter configurations
    """
    configs = {}

    # PCA - Principal Component Analysis
    configs["pca"] = {
        "params": {},  # All defaults
        "description": "Linear projection maximizing variance",
    }

    # MDS - Multi-Dimensional Scaling
    configs["mds"] = {
        "params": {},  # All defaults
        "requires_distmat": True,
        "description": "Preserves pairwise distances",
    }

    # Isomap - Isometric mapping
    configs["isomap"] = {
        "params": {"n_neighbors": 10, "max_deleted_nodes": 0.3},
        "description": "Preserves geodesic distances",
    }

    # t-SNE - t-distributed Stochastic Neighbor Embedding
    configs["tsne"] = {
        "params": {},  # Uses default perplexity=30
        "description": "Emphasizes local structure, good for visualization",
    }

    # UMAP - Uniform Manifold Approximation and Projection
    configs["umap"] = {
        "params": {"n_neighbors": 15, "min_dist": 0.1},
        "description": "Balances local and global structure",
    }

    return configs


def evaluate_dr_method(
    data: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    method_config: Dict,
    k_neighbors: int = 10,
) -> Dict:
    """
    Evaluate a single DR method on a dataset.

    Parameters
    ----------
    data : ndarray
        Input data (n_samples, n_features)
    labels : ndarray
        Ground truth labels for visualization
    method_name : str
        Name of the DR method
    method_config : dict
        Configuration parameters for the method
    k_neighbors : int
        Number of neighbors for k-NN preservation metric

    Returns
    -------
    results : dict
        Evaluation results including embedding, metrics, and performance
    """
    results = {
        "method": method_name,
        "description": method_config.get("description", ""),
        "success": False,
    }

    try:
        # Create MVData object
        mvdata = MVData(data.T)  # MVData expects (n_features, n_samples)

        # Track memory usage
        tracemalloc.start()

        # Time the embedding
        start_time = time.time()

        # Get embedding using simplified API
        params = method_config["params"]

        # For MDS, we need to compute distance matrix first
        if method_config.get("requires_distmat", False):
            mvdata.get_distmat()

        embedding = mvdata.get_embedding(method=method_name, **params)

        runtime = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Extract coordinates
        coords = embedding.coords.T  # Shape: (n_samples, n_dims)

        # Handle lost nodes if graph-based method
        if hasattr(embedding, "graph") and hasattr(embedding.graph, "lost_nodes"):
            lost_nodes = embedding.graph.lost_nodes
            if len(lost_nodes) > 0:
                # Create mask for kept nodes
                all_indices = np.arange(data.shape[0])
                kept_mask = np.ones(len(all_indices), dtype=bool)
                kept_mask[lost_nodes] = False

                # Filter data and labels
                data_filtered = data[kept_mask]
                labels_filtered = labels[kept_mask]
            else:
                data_filtered = data
                labels_filtered = labels
        else:
            data_filtered = data
            labels_filtered = labels

        # Compute quality metrics
        if coords.shape[0] == data_filtered.shape[0]:
            # k-NN preservation
            knn_score = knn_preservation_rate(data_filtered, coords, k=k_neighbors)

            # Trustworthiness and continuity
            trust = trustworthiness(data_filtered, coords, k=k_neighbors)
            cont = continuity(data_filtered, coords, k=k_neighbors)

            # Stress (normalized)
            stress_score = stress(data_filtered, coords, normalized=True)

            results.update(
                {
                    "success": True,
                    "embedding": coords,
                    "labels": labels_filtered,
                    "runtime": runtime,
                    "memory_mb": peak / 1024 / 1024,
                    "knn_preservation": knn_score,
                    "trustworthiness": trust,
                    "continuity": cont,
                    "stress": stress_score,
                    "n_samples": coords.shape[0],
                    "n_lost": data.shape[0] - coords.shape[0],
                }
            )
        else:
            results["error"] = (
                f"Dimension mismatch: {coords.shape[0]} vs {data_filtered.shape[0]}"
            )

    except Exception as e:
        results["error"] = f"{type(e).__name__}: {str(e)}"

    return results


def run_comparison(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    methods: Dict[str, Dict],
    quick: bool = False,
) -> pd.DataFrame:
    """
    Run systematic comparison of DR methods on all datasets.

    Parameters
    ----------
    datasets : dict
        Dictionary of test datasets
    methods : dict
        Dictionary of DR method configurations
    quick : bool
        If True, skip slow methods on large datasets

    Returns
    -------
    results_df : pd.DataFrame
        Results dataframe with all metrics
    """
    all_results = []

    total_comparisons = len(datasets) * len(methods)
    current = 0

    for dataset_name, (data, labels) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (shape: {data.shape})")
        print(f"{'='*60}")

        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        for method_name, method_config in methods.items():
            current += 1
            print(
                f"\n[{current}/{total_comparisons}] Evaluating {method_name}...",
                end="",
                flush=True,
            )

            # Skip slow methods on large datasets in quick mode
            if quick and method_name in ["tsne"] and data.shape[0] > 500:
                print(" [SKIPPED - quick mode]")
                continue

            # Evaluate method
            result = evaluate_dr_method(data_scaled, labels, method_name, method_config)

            # Add dataset info
            result["dataset"] = dataset_name
            result["n_features"] = data.shape[1]

            if result["success"]:
                print(
                    f" Done! (runtime: {result['runtime']:.2f}s, "
                    + f"k-NN: {result['knn_preservation']:.3f}, "
                    + f"trust: {result['trustworthiness']:.3f})"
                )
            else:
                print(f" Failed! ({result.get('error', 'Unknown error')})")

            all_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df


def visualize_comparison_results(
    results_df: pd.DataFrame, save_prefix: str = "dr_comparison"
):
    """
    Create comprehensive visualizations of comparison results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_comparison
    save_prefix : str
        Prefix for saved figure files
    """
    # Filter successful results
    success_df = results_df[results_df["success"]].copy()

    # 1. Quality metrics heatmap
    plt.figure(figsize=(12, 8))

    # Pivot data for heatmap
    metrics = ["knn_preservation", "trustworthiness", "continuity"]

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        pivot = success_df.pivot_table(
            values=metric, index="method", columns="dataset", aggfunc="mean"
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={"label": metric.replace("_", " ").title()},
        )
        plt.title(f'{metric.replace("_", " ").title()} by Method and Dataset')
        plt.xlabel("Dataset")
        plt.ylabel("Method")

    # 4. Runtime comparison
    plt.subplot(2, 2, 4)
    runtime_pivot = success_df.pivot_table(
        values="runtime", index="method", columns="dataset", aggfunc="mean"
    )

    sns.heatmap(
        np.log10(runtime_pivot + 0.001),
        annot=runtime_pivot.round(2),
        fmt="g",
        cmap="YlOrRd",
        cbar_kws={"label": "log10(Runtime in seconds)"},
    )
    plt.title("Runtime by Method and Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Method")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_metrics_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Quality vs Speed trade-off
    plt.figure(figsize=(10, 6))

    # Calculate average quality score
    success_df["avg_quality"] = success_df[
        ["knn_preservation", "trustworthiness", "continuity"]
    ].mean(axis=1)

    # Create scatter plot
    for dataset in success_df["dataset"].unique():
        mask = success_df["dataset"] == dataset
        plt.scatter(
            success_df[mask]["runtime"],
            success_df[mask]["avg_quality"],
            label=dataset,
            s=100,
            alpha=0.7,
        )

        # Add method labels
        for _, row in success_df[mask].iterrows():
            plt.annotate(
                row["method"],
                (row["runtime"], row["avg_quality"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    plt.xscale("log")
    plt.xlabel("Runtime (seconds, log scale)")
    plt.ylabel("Average Quality Score")
    plt.title("Quality vs Speed Trade-off")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_quality_vs_speed.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Example embeddings for Swiss Roll
    swiss_results = success_df[success_df["dataset"] == "swiss_roll"]

    if len(swiss_results) > 0:
        n_methods = len(swiss_results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, (_, result) in enumerate(swiss_results.iterrows()):
            if i >= 6:  # Only show first 6 methods
                break

            ax = axes[i]
            embedding = result["embedding"]
            labels = result["labels"]

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="viridis",
                s=20,
                alpha=0.7,
            )

            ax.set_title(
                f"{result['method']}\n(k-NN: {result['knn_preservation']:.3f}, "
                + f"Trust: {result['trustworthiness']:.3f})"
            )
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")

        # Hide unused subplots
        for i in range(len(swiss_results), 6):
            axes[i].set_visible(False)

        plt.suptitle("Swiss Roll Embeddings", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{save_prefix}_swiss_roll_embeddings.png", dpi=300, bbox_inches="tight"
        )
        plt.show()


def generate_recommendations(results_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate practical recommendations based on comparison results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from comparison

    Returns
    -------
    recommendations : dict
        Recommendations for different use cases
    """
    # Filter successful results
    success_df = results_df[results_df["success"]].copy()

    # Calculate summary statistics
    method_summary = (
        success_df.groupby("method")
        .agg(
            {
                "knn_preservation": "mean",
                "trustworthiness": "mean",
                "continuity": "mean",
                "stress": "mean",
                "runtime": "mean",
            }
        )
        .round(3)
    )

    recommendations = {}

    # Best overall quality
    success_df["avg_quality"] = success_df[
        ["knn_preservation", "trustworthiness", "continuity"]
    ].mean(axis=1)
    best_quality = success_df.groupby("method")["avg_quality"].mean().idxmax()
    recommendations["best_overall_quality"] = (
        f"{best_quality} - Highest average quality scores across all metrics"
    )

    # Fastest method
    fastest = method_summary["runtime"].idxmin()
    recommendations["fastest"] = (
        f"{fastest} - Average runtime: {method_summary.loc[fastest, 'runtime']:.3f}s"
    )

    # Best for visualization (high trustworthiness)
    best_viz = method_summary["trustworthiness"].idxmax()
    recommendations["best_visualization"] = (
        f"{best_viz} - Highest trustworthiness: {method_summary.loc[best_viz, 'trustworthiness']:.3f}"
    )

    # Best for preserving distances (only for methods that have stress scores)
    methods_with_stress = method_summary[method_summary["stress"].notna()]
    if not methods_with_stress.empty:
        best_distance = methods_with_stress["stress"].idxmin()
        recommendations["best_distance_preservation"] = (
            f"{best_distance} - Lowest stress: {methods_with_stress.loc[best_distance, 'stress']:.3f}"
        )

    # Dataset-specific recommendations
    for dataset in success_df["dataset"].unique():
        dataset_df = success_df[success_df["dataset"] == dataset]
        dataset_summary = dataset_df.groupby("method")["avg_quality"].mean()
        if not dataset_summary.empty:
            best_for_dataset = dataset_summary.idxmax()
            recommendations[f"best_for_{dataset}"] = best_for_dataset

    # Use case recommendations
    recommendations["use_cases"] = {
        "Exploratory visualization": "UMAP or t-SNE - Good balance of local/global structure",
        "Distance preservation": "MDS or Isomap - Explicitly preserve distances",
        "Linear relationships": "PCA - Fast, interpretable, preserves global structure",
        "Manifold learning": "Isomap or UMAP - Designed for nonlinear manifolds",
        "Large datasets": "PCA or UMAP - Computationally efficient",
        "Geodesic distances": "Isomap - Preserves manifold geodesic structure",
    }

    return recommendations, method_summary


def main(quick: bool = False):
    """
    Run the complete DR methods comparison.

    Parameters
    ----------
    quick : bool
        If True, use smaller datasets and skip slow methods
    """
    print("=" * 70)
    print("DIMENSIONALITY REDUCTION METHODS COMPARISON")
    print("=" * 70)

    # Set parameters based on mode
    if quick:
        print("\n🚀 Running in QUICK mode (reduced dataset sizes)")
        n_samples = 200
    else:
        print("\n🔬 Running in FULL mode")
        n_samples = 1000

    # Generate test datasets
    print("\n1. GENERATING TEST DATASETS")
    print("-" * 40)
    datasets = generate_test_datasets(n_samples=n_samples, noise=0.05)
    print(f"\nGenerated {len(datasets)} test datasets")

    # Get method configurations
    print("\n2. CONFIGURING DR METHODS")
    print("-" * 40)
    methods = get_dr_method_configs()
    print(f"Configured {len(methods)} DR methods: {', '.join(methods.keys())}")

    # Run comparison
    print("\n3. RUNNING SYSTEMATIC COMPARISON")
    print("-" * 40)
    results_df = run_comparison(datasets, methods, quick=quick)

    # Save results
    results_df.to_csv("dr_comparison_results.csv", index=False)
    print("\nResults saved to: dr_comparison_results.csv")

    # Visualize results
    print("\n4. GENERATING VISUALIZATIONS")
    print("-" * 40)
    visualize_comparison_results(results_df)

    # Generate recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 40)
    recommendations, method_summary = generate_recommendations(results_df)

    print("\n📊 METHOD SUMMARY:")
    print(method_summary)

    print("\n🏆 BEST METHODS:")
    for key, value in recommendations.items():
        if key != "use_cases":
            print(f"  • {key.replace('_', ' ').title()}: {value}")

    print("\n💡 USE CASE RECOMMENDATIONS:")
    for use_case, recommendation in recommendations["use_cases"].items():
        print(f"  • {use_case}: {recommendation}")

    print("\n" + "=" * 70)
    print("Comparison complete! Check generated plots for detailed results.")
    print("=" * 70)

    return results_df, recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare dimensionality reduction methods"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick comparison with smaller datasets",
    )
    args = parser.parse_args()

    results, recommendations = main(quick=args.quick)
