"""
SelectivityManifoldMapper Demonstration with Manifold Quality Metrics
==================================================================================

This example demonstrates the complete workflow of analyzing how individual neuron
selectivity relates to population-level manifold structure using DRIADA's integrated
INTENSE and dimensionality reduction capabilities.

Key concepts demonstrated:
1. Synthetic data generation with mixed selectivity neurons
2. INTENSE analysis to identify feature-selective neurons
3. Creating population embeddings with multiple DR methods
4. Analyzing neuron selectivity to embedding components
5. Visualizing functional organization in manifolds
6. Demonstrating how behavioral features map to embedding dimensions
7. Quantifying manifold preservation quality using comprehensive metrics
8. Using DRIADA's visual utilities for consistent, publication-ready figures


Usage:
    python selectivity_manifold_mapper_demo.py [--quick] [--save-plots] [--methods METHOD1,METHOD2,...]

    Options:
    --quick         Run with smaller dataset for quick testing (200 neurons, 300s)
    --save-plots    Save generated plots to files
    --methods       Comma-separated list of DR methods to use (default: pca,umap,le)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Optional
import time
import os

# Import DRIADA modules
from driada import generate_mixed_population_exp, compute_cell_feat_significance
from driada.dim_reduction import METHODS_DICT
from driada.integration import SelectivityManifoldMapper
from driada.intense import compute_embedding_selectivity

# Import manifold preservation metrics
from driada.dim_reduction.manifold_metrics import (
    knn_preservation_rate,
    trustworthiness,
    continuity,
    geodesic_distance_correlation,
    stress,
)


def get_output_directory():
    """Get the output directory for saving plots, creating it if needed."""
    output_dir = "selectivity_manifold_mapper_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate SelectivityManifoldMapper functionality"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with smaller dataset for quick testing",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save generated plots to files"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation for faster execution",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pca,umap,le",  # Changed from isomap to le
        help="Comma-separated list of DR methods to use",
    )
    return parser.parse_args()


def generate_rich_synthetic_data(quick_mode: bool = False):
    """
    Generate synthetic neural data with rich structure for demonstration.

    Parameters
    ----------
    quick_mode : bool
        If True, use smaller dataset for faster execution

    Returns
    -------
    exp : Experiment
        Generated experiment with mixed selectivity neurons
    info : dict
        Information about the generated data
    """
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 70)

    # Parameters for data generation
    sel_prob = 0.5
    mixed_sel_prob = 0.0
    manifold_fraction = 0.4
    if quick_mode:
        n_neurons = 200
        duration = 300  # 5 minutes
        print("Quick mode: 200 neurons, 5 minutes")
    else:
        n_neurons = 500
        duration = 600  # 10 minutes
        print("Full mode: 500 neurons, 10 minutes")

    # Generate mixed population with circular manifold (head direction cells)
    exp, info = generate_mixed_population_exp(
        n_neurons=n_neurons,
        manifold_type="circular",
        manifold_fraction=manifold_fraction,  # 40% head direction cells
        n_discrete_features=1,
        n_continuous_features=2,
        duration=duration,
        fps=20.0,
        correlation_mode="independent",  # Features NOT correlated with head direction
        seed=42,
        manifold_params={
            "kappa": 2.0,  # Von Mises concentration parameter
            "noise_std": 0.05,
            "baseline_rate": 0.1,
            "peak_rate": 2.0,
            "decay_time": 2.0,
            "calcium_noise_std": 0.05,
        },
        feature_params={
            "selectivity_prob": sel_prob,
            "multi_select_prob": mixed_sel_prob,
            "rate_0": 0.5,
            "rate_1": 2.0,
            "noise_std": 0.05,
            "hurst": 0.3,
            "skip_prob": 0.0,
            "ampl_range": (1.5, 3.5),
            "decay_time": 2.0,
        },
        return_info=True,
    )

    # Rename features for clarity

    print(f"\nGenerated {exp.n_cells} neurons:")
    print(f"  - Pure head direction cells: ~{int(exp.n_cells * manifold_fraction)}")
    print(
        f"  - Feature-selective cells: ~{int(exp.n_cells * (1-manifold_fraction) * sel_prob)}"
    )
    print(
        f"  - Expected mixed selectivity: ~{int(exp.n_cells * (1-manifold_fraction) * mixed_sel_prob)}"
    )
    print(f"  - Recording duration: {duration}s at 20 Hz")
    print("  - Manifold type: Circular (head direction)")
    print("  - Features: Independent of head direction")

    return exp, info


def run_intense_analysis(exp, quick_mode: bool = False):
    """
    Run INTENSE analysis to identify feature-selective neurons.

    Parameters
    ----------
    exp : Experiment
        Experiment object with neural data
    quick_mode : bool
        If True, use fewer shuffles for faster execution

    Returns
    -------
    results : dict
        INTENSE analysis results
    """
    print("\n" + "=" * 70)
    print("RUNNING INTENSE ANALYSIS")
    print("=" * 70)

    # Parameters for INTENSE
    if quick_mode:
        n_shuffles_stage1 = 50
        n_shuffles_stage2 = 500
        print("Quick mode: 50/500 shuffles")
    else:
        n_shuffles_stage1 = 100
        n_shuffles_stage2 = 2000
        print("Full mode: 100/2000 shuffles")

    # Analyze all behavioral features
    features_to_analyze = ["head_direction", "c_feat_0", "c_feat_1", "d_feat_0"]
    available_features = [f for f in features_to_analyze if f in exp.dynamic_features]

    print(f"Analyzing {exp.n_cells} neurons × {len(available_features)} features")

    # Skip delays for head direction (it's circular)
    skip_delays = (
        {"head_direction": True} if "head_direction" in available_features else {}
    )

    # Run INTENSE analysis
    start_time = time.time()
    stats, significance, info, intense_results = compute_cell_feat_significance(
        exp,
        feat_bunch=available_features,
        mode="two_stage",
        n_shuffles_stage1=n_shuffles_stage1,
        n_shuffles_stage2=n_shuffles_stage2,
        metric_distr_type="norm",
        pval_thr=0.01,
        multicomp_correction=None,
        verbose=True,
        find_optimal_delays=True,
        skip_delays=skip_delays,
        shift_window=2,
        ds=5,  # Downsample for efficiency
        allow_mixed_dimensions=True,  # For position_2d MultiTimeSeries
    )

    elapsed_time = time.time() - start_time
    print(f"\nINTENSE analysis completed in {elapsed_time:.1f} seconds")

    # Get significant neurons
    significant_neurons = exp.get_significant_neurons()
    mixed_selectivity_neurons = exp.get_significant_neurons(min_nspec=2)

    print("\nSelectivity Summary:")
    print(f"  - Total selective neurons: {len(significant_neurons)}/{exp.n_cells}")
    print(f"  - Mixed selectivity neurons: {len(mixed_selectivity_neurons)}")

    # Count selectivity by feature
    feature_counts = {}
    for neuron_id, features in significant_neurons.items():
        for feat in features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    print("\nSelectivity by feature:")
    for feat, count in sorted(feature_counts.items()):
        print(f"  - {feat}: {count} neurons")

    return {
        "stats": stats,
        "significance": significance,
        "significant_neurons": significant_neurons,
        "mixed_selectivity_neurons": mixed_selectivity_neurons,
        "feature_counts": feature_counts,
    }


def create_embeddings_and_analyze(exp, methods: List[str], quick_mode: bool = False):
    """
    Create embeddings using SelectivityManifoldMapper and analyze component selectivity.

    Parameters
    ----------
    exp : Experiment
        Experiment with INTENSE results
    methods : list of str
        DR methods to use
    quick_mode : bool
        If True, use fewer components and shuffles

    Returns
    -------
    mapper : SelectivityManifoldMapper
        Initialized mapper object
    embedding_results : dict
        Results from compute_embedding_selectivity
    """
    print("\n" + "=" * 70)
    print("CREATING EMBEDDINGS AND ANALYZING COMPONENT SELECTIVITY")
    print("=" * 70)

    # Initialize SelectivityManifoldMapper
    mapper = SelectivityManifoldMapper(exp)

    # Parameters for embeddings
    if quick_mode:
        n_components = 5
        n_shuffles = 500
        print("Quick mode: 5 components, 500 shuffles")
    else:
        n_components = 5
        n_shuffles = 1000
        print("Full mode: 5 components, 1000 shuffles")

    # Create embeddings for each method
    for method in methods:
        print(f"\nCreating {method.upper()} embedding...")

        # Method-specific parameters
        if method == "pca":
            dr_kwargs = {}
        elif method == "umap":
            # For circular manifolds, UMAP needs specific tuning
            dr_kwargs = {
                "n_neighbors": 50,  # Reduced for tighter local structure
                "min_dist": 0.8,  # Balanced for circular preservation
                "metric": "euclidean",
                "n_epochs": 1000,  # More epochs for better convergence
                "init": "spectral",  # Better initialization for manifolds
            }
        elif method == "le":
            # Laplacian Eigenmaps for circular manifolds
            dr_kwargs = {
                "n_neighbors": 100,  # Smaller neighborhood for circular structure
            }
        elif method == "isomap":
            dr_kwargs = {"n_neighbors": 50}
        else:
            dr_kwargs = {}

        # Create embedding using all neurons with downsampling
        try:
            # Add downsampling parameter to dr_kwargs
            dr_kwargs["ds"] = 5  # Same downsampling as used in INTENSE analysis

            embedding = mapper.create_embedding(
                method=method,
                n_components=n_components,
                data_type="calcium",
                neuron_selection="all",
                **dr_kwargs,
            )
            print(f"  Created {method} embedding: shape {embedding.shape}")
        except Exception as e:
            print(f"  Failed to create {method} embedding: {e}")
            continue

    # Check which methods were successfully created
    successful_methods = []
    for method in methods:
        try:
            exp.get_embedding(method, "calcium")
            successful_methods.append(method)
        except:
            pass

    if not successful_methods:
        print("No embeddings were successfully created!")
        return mapper, {}

    # Analyze component selectivity for all embeddings
    print("\n" + "-" * 50)
    print("Analyzing neuron selectivity to embedding components...")
    print("-" * 50)

    embedding_results = compute_embedding_selectivity(
        exp,
        embedding_methods=successful_methods,
        mode="two_stage",
        n_shuffles_stage1=50,
        n_shuffles_stage2=n_shuffles,
        metric_distr_type="norm",
        pval_thr=0.01,  # More lenient for components
        multicomp_correction=None,  # No correction for exploratory analysis
        find_optimal_delays=False,  # Components are instantaneous
        verbose=True,
        ds=5,
    )

    # Summarize results
    print("\n" + "=" * 50)
    print("COMPONENT SELECTIVITY SUMMARY")
    print("=" * 50)

    for method, results in embedding_results.items():
        n_sig_neurons = len(results["significant_neurons"])
        n_components = results["n_components"]

        print(f"\n{method.upper()}:")
        print(f"  - Neurons selective to components: {n_sig_neurons}")
        print("  - Components with selective neurons:")

        for comp_idx, neuron_list in results["component_selectivity"].items():
            if neuron_list:
                print(f"    Component {comp_idx}: {len(neuron_list)} neurons")

    return mapper, embedding_results


def compute_manifold_quality_metrics(exp, methods: List[str], k_neighbors: int = 20):
    """
    Compute manifold preservation metrics for each embedding method.

    Parameters
    ----------
    exp : Experiment
        Experiment object with embeddings
    methods : list of str
        DR methods to evaluate
    k_neighbors : int
        Number of neighbors for metrics computation

    Returns
    -------
    metrics_dict : dict
        Dictionary mapping method names to metric values
    """
    print("\n" + "=" * 70)
    print("COMPUTING MANIFOLD PRESERVATION METRICS")
    print("=" * 70)

    # Get calcium data as high-dimensional representation
    # exp.calcium is a MultiTimeSeries object with shape (n_neurons, n_timepoints)
    # Use scdata for scaled data that equalizes neuron contributions
    X_high = exp.calcium.scdata.T  # (n_timepoints, n_neurons)

    # Downsample for computational efficiency (same as used in analysis)
    ds = 5
    X_high = X_high[::ds]

    metrics_dict = {}

    for method in methods:
        try:
            # Get embedding
            embedding_dict = exp.get_embedding(method, "calcium")
            if embedding_dict is None:
                continue

            X_low = embedding_dict["data"]

            # Check if embedding was created with downsampling
            # If created by SelectivityManifoldMapper, it includes ds in metadata
            ds_used = embedding_dict.get("metadata", {}).get("ds", 1)
            if ds_used != ds:
                # Re-downsample the embedding if needed
                if ds_used == 1 and X_low.shape[0] == exp.calcium.scdata.shape[1]:
                    X_low = X_low[::ds]

            # Ensure same dimensions
            if X_low.shape[0] != X_high.shape[0]:
                print(
                    f"Warning: Embedding size mismatch for {method} after adjustment, skipping metrics"
                )
                print(f"  X_high shape: {X_high.shape}, X_low shape: {X_low.shape}")
                continue

            print(f"\nComputing metrics for {method.upper()}...")

            # Compute various metrics
            metrics = {}

            # 1. k-NN preservation rate
            try:
                metrics["knn_preservation"] = knn_preservation_rate(
                    X_high, X_low, k=k_neighbors
                )
                print(f"  k-NN preservation rate: {metrics['knn_preservation']:.3f}")
            except Exception as e:
                print(f"  k-NN preservation failed: {e}")
                metrics["knn_preservation"] = None

            # 2. Trustworthiness
            try:
                metrics["trustworthiness"] = trustworthiness(
                    X_high, X_low, k=k_neighbors
                )
                print(f"  Trustworthiness: {metrics['trustworthiness']:.3f}")
            except Exception as e:
                print(f"  Trustworthiness failed: {e}")
                metrics["trustworthiness"] = None

            # 3. Continuity
            try:
                metrics["continuity"] = continuity(X_high, X_low, k=k_neighbors)
                print(f"  Continuity: {metrics['continuity']:.3f}")
            except Exception as e:
                print(f"  Continuity failed: {e}")
                metrics["continuity"] = None

            # 4. Geodesic distance correlation (for manifold structure)
            try:
                metrics["geodesic_corr"] = geodesic_distance_correlation(
                    X_high, X_low, k_neighbors=k_neighbors, method="spearman"
                )
                print(
                    f"  Geodesic distance correlation: {metrics['geodesic_corr']:.3f}"
                )
            except Exception as e:
                print(f"  Geodesic correlation failed: {e}")
                metrics["geodesic_corr"] = None

            # 5. Normalized stress
            try:
                metrics["stress"] = stress(X_high, X_low, normalized=True)
                print(f"  Normalized stress: {metrics['stress']:.3f}")
            except Exception as e:
                print(f"  Stress computation failed: {e}")
                metrics["stress"] = None

            metrics_dict[method] = metrics

        except Exception as e:
            print(f"Failed to compute metrics for {method}: {e}")
            continue

    return metrics_dict


def visualize_results(
    exp,
    mapper,
    embedding_results,
    methods: List[str],
    manifold_metrics: Optional[Dict] = None,
    save_plots: bool = False,
):
    """
    Create comprehensive visualizations of the results.

    Parameters
    ----------
    exp : Experiment
        Experiment object
    mapper : SelectivityManifoldMapper
        Mapper with stored embeddings
    embedding_results : dict
        Component selectivity results
    methods : list of str
        DR methods to visualize
    manifold_metrics : dict, optional
        Manifold preservation metrics for each method
    save_plots : bool
        Whether to save plots to files
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Embedding comparison figure
    create_embedding_comparison_figure(exp, methods, save_plots)

    # 2. Separate trajectory visualization
    create_trajectory_figure(exp, methods, save_plots)

    # 3. Component selectivity heatmap (using visual utilities)
    create_component_selectivity_heatmap_v2(exp, embedding_results, save_plots)

    # 4. Functional organization analysis
    create_functional_organization_figure(exp, mapper, embedding_results, save_plots)

    # 5. Component interpretation figure
    create_component_interpretation_figure(exp, embedding_results, save_plots)

    # 6. Manifold quality metrics visualization (NEW)
    if manifold_metrics is not None:
        create_manifold_quality_figure(manifold_metrics, save_plots)

    plt.show()


def create_embedding_comparison_figure(
    exp, methods: List[str], save_plots: bool = False
):
    """Create figure comparing embeddings colored by behavioral features."""
    from driada.utils.visual import plot_embedding_comparison
    import os

    # Prepare embeddings dict
    embeddings = {}
    for method in methods:
        embedding_dict = exp.get_embedding(method, "calcium")
        if embedding_dict is not None:
            embeddings[method] = embedding_dict["data"]

    # Get behavioral data and downsample to match embeddings
    features = {}
    ds = 5  # Same downsampling as used in INTENSE analysis

    # Use head_direction for angle (convert from [0, 2π] to [-π, π])
    if "head_direction" in exp.dynamic_features:
        angle_data = exp.dynamic_features["head_direction"].data[::ds]
        # Convert from [0, 2π] to [-π, π] as expected by visual utilities
        features["angle"] = angle_data - np.pi

    # Use first continuous feature for secondary coloring
    if "c_feat_0" in exp.dynamic_features:
        features["speed"] = exp.dynamic_features["c_feat_0"].data[::ds]

    # Create output directory if needed
    if save_plots:
        output_dir = get_output_directory()
        save_path = os.path.join(output_dir, "selectivity_mapper_embeddings.png")
    else:
        save_path = None

    # Create figure using visual utility
    fig = plot_embedding_comparison(
        embeddings=embeddings,
        features=features,
        methods=methods,
        with_trajectory=True,
        compute_metrics=True,
        save_path=save_path,
    )

    if save_plots:
        print(f"Saved: {save_path}")

    return fig


def create_trajectory_figure(exp, methods: List[str], save_plots: bool = False):
    """Create separate figure showing trajectories in embedding space."""
    from driada.utils.visual import plot_trajectories
    import os

    # Prepare embeddings dict
    embeddings = {}
    for method in methods:
        embedding_dict = exp.get_embedding(method, "calcium")
        if embedding_dict is not None:
            embeddings[method] = embedding_dict["data"]

    # Create output directory if needed
    if save_plots:
        output_dir = get_output_directory()
        save_path = os.path.join(output_dir, "selectivity_mapper_trajectories.png")
    else:
        save_path = None

    # Create figure using visual utility
    fig = plot_trajectories(embeddings=embeddings, methods=methods, save_path=save_path)

    if save_plots:
        print(f"Saved: {save_path}")

    return fig


def create_component_selectivity_heatmap_v2(
    exp, embedding_results: Dict, save_plots: bool = False
):
    """Create heatmap showing neuron selectivity to embedding components using visual utilities."""
    from driada.utils.visual import plot_component_selectivity_heatmap
    import os

    # Get methods
    methods = list(embedding_results.keys())

    # Calculate total components and components per method
    n_components_per_method = {
        method: results["n_components"] for method, results in embedding_results.items()
    }
    total_components = sum(n_components_per_method.values())

    # Create single concatenated selectivity matrix
    n_neurons = exp.n_cells
    selectivity_matrix = np.zeros((n_neurons, total_components))

    # Fill in MI values for each method
    comp_offset = 0
    for method, results in embedding_results.items():
        n_components = results["n_components"]

        # Fill in MI values for significant pairs
        for neuron_id, neuron_stats in results["stats"].items():
            for feat_name, stats in neuron_stats.items():
                # Check if this is a component feature
                if isinstance(feat_name, str) and feat_name.startswith(
                    f"{method}_comp"
                ):
                    comp_idx = int(feat_name.split("_comp")[-1])

                    if stats.get("me") is not None:
                        # Check if significant
                        if (
                            neuron_id in results["significance"]
                            and feat_name in results["significance"][neuron_id]
                            and results["significance"][neuron_id][feat_name].get(
                                "stage2", False
                            )
                        ):
                            selectivity_matrix[neuron_id, comp_offset + comp_idx] = (
                                stats["me"]
                            )

        comp_offset += n_components

    # Create output directory if needed
    if save_plots:
        output_dir = get_output_directory()
        save_path = os.path.join(output_dir, "selectivity_mapper_component_heatmap.png")
    else:
        save_path = None

    # Use visual utility to create the figure
    fig = plot_component_selectivity_heatmap(
        selectivity_matrix=selectivity_matrix,
        methods=methods,
        n_components_per_method=n_components_per_method,
        save_path=save_path,
    )

    if save_plots:
        print(f"Saved: {save_path}")

    return fig


def create_functional_organization_figure(
    exp, mapper, embedding_results: Dict, save_plots: bool = False
):
    """Analyze and visualize functional organization in the manifold."""
    import os

    fig = plt.figure(figsize=(15, 5))

    # Get original feature selectivity
    significant_neurons = exp.get_significant_neurons()

    for i, method in enumerate(embedding_results.keys()):
        ax = fig.add_subplot(1, len(embedding_results), i + 1)

        results = embedding_results[method]

        # Analyze overlap between feature-selective and component-selective neurons
        feature_selective = set(significant_neurons.keys())
        component_selective = set(results["significant_neurons"].keys())

        # Create Venn diagram data
        only_features = len(feature_selective - component_selective)
        both = len(feature_selective & component_selective)
        only_components = len(component_selective - feature_selective)

        # Simple bar plot instead of Venn diagram
        categories = ["Features\nonly", "Both", "Components\nonly"]
        values = [only_features, both, only_components]
        colors = ["skyblue", "lightgreen", "salmon"]

        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel("Number of neurons")
        ax.set_title(f"{method.upper()} - Functional Organization")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value}",
                ha="center",
                va="bottom",
            )

        # Add summary statistics
        total_selective = only_features + both + only_components
        if total_selective > 0:
            overlap_pct = (both / total_selective) * 100
            ax.text(
                0.5,
                0.95,
                f"Overlap: {overlap_pct:.1f}%",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.suptitle(
        "Functional Organization: Feature vs Component Selectivity", fontsize=16
    )
    plt.tight_layout()

    if save_plots:
        output_dir = get_output_directory()
        save_path = os.path.join(output_dir, "selectivity_mapper_functional_org.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def create_component_interpretation_figure(
    exp, embedding_results: Dict, save_plots: bool = False
):
    """Visualize how components relate to behavioral features using MI values."""
    from driada.utils.visual import plot_component_interpretation
    from driada.information.info_base import get_sim, TimeSeries
    import os

    # Get list of available methods
    available_methods = [
        method
        for method in embedding_results.keys()
        if method in ["pca", "umap", "le"] and embedding_results[method] is not None
    ]

    if not available_methods:
        print("No DR methods found in results, skipping component interpretation")
        return

    # Get behavioral feature names and keys that actually exist in the experiment
    feature_names = []
    feature_keys = []

    if "head_direction" in exp.dynamic_features:
        feature_names.append("Head Direction")
        feature_keys.append("head_direction")

    if "c_feat_0" in exp.dynamic_features:
        feature_names.append("Continuous Feature 0")
        feature_keys.append("c_feat_0")

    if "c_feat_1" in exp.dynamic_features:
        feature_names.append("Continuous Feature 1")
        feature_keys.append("c_feat_1")

    if "d_feat_0" in exp.dynamic_features:
        feature_names.append("Discrete Feature")
        feature_keys.append("d_feat_0")

    # Prepare MI matrices and metadata
    mi_matrices = {}
    metadata = {}

    ds = 5  # Same as used in INTENSE analysis

    for method in available_methods:
        try:
            embedding_dict = exp.get_embedding(method, "calcium")
            embedding = embedding_dict["data"]

            # Compute MI between components and behavioral features
            n_components = min(5, embedding_results[method]["n_components"])
            mi_matrix = np.zeros((len(feature_names), n_components))

            for comp_idx in range(n_components):
                comp_data = embedding[:, comp_idx]

                for feat_idx, feat_key in enumerate(feature_keys):
                    try:
                        feat_data = exp.dynamic_features[feat_key].data
                        is_discrete = exp.dynamic_features[feat_key].discrete

                        # Create TimeSeries objects
                        comp_ts = TimeSeries(comp_data, discrete=False)
                        feat_ts = TimeSeries(feat_data, discrete=is_discrete)

                        # Compute MI
                        mi = get_sim(
                            comp_ts,
                            feat_ts,
                            metric="mi",
                            shift=0,
                            ds=ds,
                            k=5,
                            estimator="gcmi",
                        )

                        mi_matrix[feat_idx, comp_idx] = mi

                    except Exception as e:
                        print(
                            f"Error computing MI for {method} comp{comp_idx} vs {feat_key}: {e}"
                        )
                        mi_matrix[feat_idx, comp_idx] = 0

            mi_matrices[method] = mi_matrix

            # Add metadata if available
            if method == "pca" and "metadata" in embedding_dict:
                metadata[method] = embedding_dict["metadata"]

        except Exception as e:
            print(f"Failed to process {method}: {e}")
            continue

    # Create output directory if needed
    if save_plots:
        output_dir = get_output_directory()
        save_path = os.path.join(
            output_dir, "selectivity_mapper_component_interpretation.png"
        )
    else:
        save_path = None

    # Create figure using visual utility
    fig = plot_component_interpretation(
        mi_matrices=mi_matrices,
        feature_names=feature_names,
        metadata=metadata,
        n_components=5,
        save_path=save_path,
    )

    if save_plots:
        print(f"Saved: {save_path}")

    return fig


def create_manifold_quality_figure(metrics_dict: Dict, save_plots: bool = False):
    """
    Create visualization of manifold preservation metrics.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping method names to metric values
    save_plots : bool
        Whether to save the plot
    """
    # Prepare data for visualization
    methods = list(metrics_dict.keys())
    metric_names = [
        "knn_preservation",
        "trustworthiness",
        "continuity",
        "geodesic_corr",
        "stress",
    ]
    metric_labels = [
        "k-NN\nPreservation",
        "Trustworthiness",
        "Continuity",
        "Geodesic\nCorrelation",
        "Normalized\nStress",
    ]

    # Create matrix of metric values
    n_methods = len(methods)
    n_metrics = len(metric_names)
    metric_matrix = np.zeros((n_methods, n_metrics))

    for i, method in enumerate(methods):
        for j, metric_name in enumerate(metric_names):
            value = metrics_dict[method].get(metric_name)
            if value is not None:
                # For stress, lower is better, so invert it
                if metric_name == "stress":
                    metric_matrix[i, j] = 1 - value
                else:
                    metric_matrix[i, j] = value
            else:
                metric_matrix[i, j] = np.nan

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Bar plot comparison
    x = np.arange(n_metrics)
    width = 0.25

    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset, metric_matrix[i], width, label=method.upper(), alpha=0.8
        )

        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Score (higher is better)")
    ax1.set_title("Manifold Preservation Metrics Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1.1)

    # 2. Radar plot
    # Prepare data for radar plot
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax2 = plt.subplot(122, projection="polar")

    for i, method in enumerate(methods):
        values = metric_matrix[i].tolist()
        values += values[:1]  # Complete the circle

        # Replace NaN with 0 for plotting
        values = [0 if np.isnan(v) else v for v in values]

        ax2.plot(angles, values, "o-", linewidth=2, label=method.upper())
        ax2.fill(angles, values, alpha=0.25)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylim(0, 1)
    ax2.set_title("Manifold Quality Profile", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    ax2.grid(True)

    plt.suptitle("Manifold Preservation Quality Assessment", fontsize=16)
    plt.tight_layout()

    if save_plots:
        output_dir = get_output_directory()
        save_path = os.path.join(output_dir, "selectivity_mapper_manifold_quality.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    methods = [m.strip() for m in args.methods.split(",")]

    # Validate methods
    valid_methods = []
    for method in methods:
        if method in METHODS_DICT:
            valid_methods.append(method)
        else:
            print(f"Warning: Unknown method '{method}', skipping")

    if not valid_methods:
        print("Error: No valid DR methods specified")
        return

    print("\n" + "=" * 70)
    print("SELECTIVITY MANIFOLD MAPPER DEMONSTRATION")
    print("=" * 70)
    print(f"DR methods: {', '.join(valid_methods)}")
    print(f"Quick mode: {args.quick}")
    print(f"Save plots: {args.save_plots}")

    # Step 1: Generate synthetic data
    exp, info = generate_rich_synthetic_data(args.quick)

    # Step 2: Run INTENSE analysis
    intense_results = run_intense_analysis(exp, args.quick)

    # Step 3: Create embeddings and analyze component selectivity
    mapper, embedding_results = create_embeddings_and_analyze(
        exp, valid_methods, args.quick
    )

    # Step 4: Compute manifold preservation metrics
    manifold_metrics = compute_manifold_quality_metrics(exp, valid_methods)

    # Step 5: Create visualizations (unless disabled)
    if not args.no_viz:
        visualize_results(
            exp,
            mapper,
            embedding_results,
            valid_methods,
            manifold_metrics,
            args.save_plots,
        )

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
