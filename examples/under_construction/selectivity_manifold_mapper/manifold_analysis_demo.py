"""
Manifold Analysis Demonstration
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

Usage:
    python manifold_analysis_demo.py [--quick] [--save-plots] [--methods METHOD1,METHOD2,...]

    Options:
    --quick         Run with smaller dataset for quick testing (200 neurons, 300s)
    --save-plots    Save generated plots to files
    --methods       Comma-separated list of DR methods to use (default: pca,umap)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Optional
import time
import os

# Import DRIADA modules
from driada import generate_mixed_population_exp, compute_cell_feat_significance
from driada.integration import get_functional_organization, compare_embeddings
from driada.intense import compute_embedding_selectivity

# For visualization
from driada.utils.plot import create_default_figure


def get_output_directory():
    """Get the output directory for saving plots, creating it if needed."""
    output_dir = "manifold_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate manifold analysis functionality"
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
        default="pca,umap",
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
    print("\n" + "=" * 80)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 80)

    # Parameters
    if quick_mode:
        n_neurons = 200
        duration = 300  # 5 minutes
        print(f"Quick mode: {n_neurons} neurons, {duration}s recording")
    else:
        n_neurons = 500
        duration = 600  # 10 minutes
        print(f"Full mode: {n_neurons} neurons, {duration}s recording")

    # Generate experiment with mixed selectivity
    exp, info = generate_mixed_population_exp(
        n_neurons=n_neurons,
        duration=duration,
        # Feature selectivity parameters
        position_selective_fraction=0.3,
        direction_selective_fraction=0.3,
        speed_selective_fraction=0.2,
        mixed_selective_fraction=0.2,
        # Spatial organization
        use_spatial_embedding=True,
        spatial_scale=10.0,
        # Noise and dynamics
        noise_level=0.1,
        calcium_dynamics="realistic",
        # Additional features
        include_head_direction=True,
        include_speed=True,
        verbose=True,
    )

    print(f"\nGenerated {exp.n_cells} neurons with {exp.n_frames} timepoints")
    print(f"Available features: {list(exp.dynamic_features.keys())}")

    return exp, info


def run_intense_analysis(exp, quick_mode: bool = False):
    """
    Run INTENSE selectivity analysis on the experiment.

    Parameters
    ----------
    exp : Experiment
        The experiment to analyze
    quick_mode : bool
        If True, use fewer shuffles for faster execution

    Returns
    -------
    results : dict
        INTENSE analysis results
    """
    print("\n" + "=" * 80)
    print("RUNNING INTENSE ANALYSIS")
    print("=" * 80)

    # Parameters
    if quick_mode:
        n_shuffles_stage1 = 10
        n_shuffles_stage2 = 100
    else:
        n_shuffles_stage1 = 100
        n_shuffles_stage2 = 1000

    print(f"Stage 1: {n_shuffles_stage1} shuffles")
    print(f"Stage 2: {n_shuffles_stage2} shuffles")

    # Run INTENSE analysis
    start_time = time.time()
    results = compute_cell_feat_significance(
        exp,
        mode="two_stage",
        n_shuffles_stage1=n_shuffles_stage1,
        n_shuffles_stage2=n_shuffles_stage2,
        verbose=True,
        enable_parallelization=True,
    )

    elapsed = time.time() - start_time
    print(f"\nINTENSE analysis completed in {elapsed:.1f} seconds")

    # Summarize results
    sig_neurons = exp.get_significant_neurons()
    print(f"Found {len(sig_neurons)} significantly selective neurons")

    # Count selectivity types
    selectivity_counts = {}
    for neuron_id, neuron_info in sig_neurons.items():
        features = neuron_info["features"]
        feature_str = ",".join(sorted(features))
        selectivity_counts[feature_str] = selectivity_counts.get(feature_str, 0) + 1

    print("\nSelectivity breakdown:")
    for features, count in sorted(
        selectivity_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {features}: {count} neurons")

    return results


def create_embeddings(exp, methods: List[str], quick_mode: bool = False):
    """
    Create population embeddings using different DR methods.

    Parameters
    ----------
    exp : Experiment
        The experiment with neural data
    methods : list
        List of DR method names
    quick_mode : bool
        If True, use fewer components

    Returns
    -------
    embeddings : dict
        Dictionary mapping method names to embedding results
    """
    print("\n" + "=" * 80)
    print("CREATING EMBEDDINGS")
    print("=" * 80)

    # Parameters
    n_components = 10 if quick_mode else 20
    embeddings = {}

    for method in methods:
        print(f"\nCreating {method.upper()} embedding...")
        start_time = time.time()

        try:
            # Create embedding using the Experiment's new method
            if method == "umap":
                embedding = exp.create_embedding(
                    method,
                    n_components=n_components,
                    neuron_selection="significant",  # Use only significant neurons
                    n_neighbors=15,
                    min_dist=0.1,
                )
            else:
                embedding = exp.create_embedding(
                    method,
                    n_components=n_components,
                    neuron_selection="significant",
                )

            elapsed = time.time() - start_time
            print(f"  Created in {elapsed:.1f} seconds")
            print(f"  Shape: {embedding.shape}")

            # Get stored embedding info
            stored = exp.get_embedding(method, "calcium")
            n_neurons_used = stored["metadata"]["n_neurons"]
            print(f"  Used {n_neurons_used} significant neurons")

            embeddings[method] = {
                "data": embedding,
                "metadata": stored["metadata"],
            }

        except Exception as e:
            print(f"  Failed to create {method} embedding: {e}")
            continue

    return embeddings


def analyze_embedding_selectivity(exp, methods: List[str], quick_mode: bool = False):
    """
    Analyze how neurons are selective to embedding components.

    Parameters
    ----------
    exp : Experiment
        The experiment with embeddings
    methods : list
        List of embedding methods to analyze
    quick_mode : bool
        If True, use fewer shuffles

    Returns
    -------
    results : dict
        Selectivity analysis results
    """
    print("\n" + "=" * 80)
    print("ANALYZING EMBEDDING SELECTIVITY")
    print("=" * 80)

    # Parameters
    if quick_mode:
        n_shuffles = 10
    else:
        n_shuffles = 100

    # Run embedding selectivity analysis
    start_time = time.time()
    results = compute_embedding_selectivity(
        exp,
        embedding_methods=methods,
        mode="stage1",  # Faster for demo
        n_shuffles_stage1=n_shuffles,
        verbose=True,
        enable_parallelization=True,
    )

    elapsed = time.time() - start_time
    print(f"\nEmbedding selectivity analysis completed in {elapsed:.1f} seconds")

    # Summarize results
    for method in methods:
        if method not in results:
            continue

        print(f"\n{method.upper()} selectivity summary:")
        method_stats = results[method]["stats"]
        n_selective = 0

        for comp_idx in range(exp.embeddings["calcium"][method]["shape"][1]):
            comp_name = f"{method}_comp{comp_idx}"
            if comp_name in exp.significance_tables["calcium"]:
                comp_selective = len(
                    [
                        n
                        for n, sig in exp.significance_tables["calcium"][
                            comp_name
                        ].items()
                        if sig.get("stage1", False)
                    ]
                )
                n_selective += comp_selective
                print(f"  Component {comp_idx}: {comp_selective} selective neurons")

        print(f"  Total selective connections: {n_selective}")

    return results


def analyze_functional_organization(exp, methods: List[str]):
    """
    Analyze functional organization in each manifold.

    Parameters
    ----------
    exp : Experiment
        The experiment with embeddings and selectivity
    methods : list
        List of embedding methods to analyze

    Returns
    -------
    organizations : dict
        Functional organization for each method
    """
    print("\n" + "=" * 80)
    print("ANALYZING FUNCTIONAL ORGANIZATION")
    print("=" * 80)

    organizations = {}

    for method in methods:
        if method not in exp.embeddings["calcium"]:
            continue

        print(f"\n{method.upper()} functional organization:")
        org = get_functional_organization(exp, method)
        organizations[method] = org

        # Print summary
        print(f"  Components: {org['n_components']}")
        print(f"  Neurons used: {org['n_neurons_used']}")
        print(f"  Component importance (variance): {org['component_importance'][:5]}...")

        if "n_participating_neurons" in org:
            print(f"  Participating neurons: {org['n_participating_neurons']}")
            print(
                f"  Mean components per neuron: {org['mean_components_per_neuron']:.2f}"
            )
            print(f"  Functional clusters: {len(org['functional_clusters'])}")

            # Show top clusters
            if org["functional_clusters"]:
                print("  Top functional clusters:")
                for i, cluster in enumerate(org["functional_clusters"][:3]):
                    print(
                        f"    Cluster {i+1}: {cluster['size']} neurons, "
                        f"components {cluster['components']}"
                    )

    return organizations


def compare_embedding_methods(exp, methods: List[str]):
    """
    Compare functional organization across different embeddings.

    Parameters
    ----------
    exp : Experiment
        The experiment with multiple embeddings
    methods : list
        List of methods to compare

    Returns
    -------
    comparison : dict
        Comparison results
    """
    print("\n" + "=" * 80)
    print("COMPARING EMBEDDING METHODS")
    print("=" * 80)

    comparison = compare_embeddings(exp, methods)

    print("\nMethod comparison:")
    print(f"  Methods: {comparison['methods']}")
    print(f"  Components: {comparison['n_components']}")
    print(f"  Participating neurons: {comparison['n_participating_neurons']}")
    print(f"  Functional clusters: {comparison['n_functional_clusters']}")

    if "participation_overlap" in comparison:
        print("\nNeuron participation overlap:")
        for pair, overlap in comparison["participation_overlap"].items():
            print(f"  {pair}: {overlap:.3f}")

    return comparison


def visualize_results(exp, organizations: Dict, args):
    """
    Create comprehensive visualizations of the results.

    Parameters
    ----------
    exp : Experiment
        The analyzed experiment
    organizations : dict
        Functional organization for each method
    args : argparse.Namespace
        Command line arguments
    """
    if args.no_viz:
        print("\nSkipping visualizations (--no-viz flag)")
        return

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = get_output_directory() if args.save_plots else None

    # 1. Component importance comparison
    fig, axes = create_default_figure(
        figsize=(12, 8), ncols=len(organizations), squeeze=False
    )
    axes = axes[0]  # Get first row

    for i, (method, org) in enumerate(organizations.items()):
        ax = axes[i]
        importance = org["component_importance"]
        ax.bar(range(len(importance)), importance)
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance Explained")
        ax.set_title(f"{method.upper()} Component Importance")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "component_importance.png"), dpi=300)
    plt.show()

    # 2. Neuron participation heatmap
    if any("neuron_participation" in org for org in organizations.values()):
        fig, axes = create_default_figure(
            figsize=(14, 6), ncols=len(organizations), squeeze=False
        )
        axes = axes[0]

        for i, (method, org) in enumerate(organizations.items()):
            if "neuron_participation" not in org:
                continue

            ax = axes[i]

            # Create participation matrix
            n_neurons = exp.n_cells
            n_components = org["n_components"]
            participation_matrix = np.zeros((n_neurons, n_components))

            for neuron_idx, components in org["neuron_participation"].items():
                for comp in components:
                    participation_matrix[neuron_idx, comp] = 1

            # Plot heatmap
            im = ax.imshow(participation_matrix, aspect="auto", cmap="RdBu_r")
            ax.set_xlabel("Component")
            ax.set_ylabel("Neuron")
            ax.set_title(f"{method.upper()} Neuron Participation")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "neuron_participation.png"), dpi=300)
        plt.show()

    print("\nVisualization complete!")
    if output_dir:
        print(f"Plots saved to: {output_dir}")


def main():
    """Main demonstration workflow."""
    args = parse_arguments()

    print("\n" + "=" * 80)
    print("MANIFOLD ANALYSIS DEMONSTRATION")
    print("=" * 80)

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",")]
    print(f"DR methods to analyze: {methods}")

    # Step 1: Generate synthetic data
    exp, info = generate_rich_synthetic_data(args.quick)

    # Step 2: Run INTENSE analysis
    intense_results = run_intense_analysis(exp, args.quick)

    # Step 3: Create embeddings
    embeddings = create_embeddings(exp, methods, args.quick)

    if not embeddings:
        print("\nNo embeddings created successfully. Exiting.")
        return

    # Step 4: Analyze embedding selectivity
    selectivity_results = analyze_embedding_selectivity(
        exp, list(embeddings.keys()), args.quick
    )

    # Step 5: Analyze functional organization
    organizations = analyze_functional_organization(exp, list(embeddings.keys()))

    # Step 6: Compare methods
    if len(embeddings) > 1:
        comparison = compare_embedding_methods(exp, list(embeddings.keys()))

    # Step 7: Create visualizations
    visualize_results(exp, organizations, args)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey findings:")
    print(f"- Generated {exp.n_cells} neurons with mixed selectivity")
    print(f"- Found {len(exp.get_significant_neurons())} selective neurons")
    print(f"- Created {len(embeddings)} embeddings")
    print(f"- Analyzed functional organization in each manifold")

    if len(embeddings) > 1 and "participation_overlap" in comparison:
        print("\nEmbedding similarities:")
        for pair, overlap in comparison["participation_overlap"].items():
            print(f"  {pair}: {overlap:.1%} neuron overlap")


if __name__ == "__main__":
    main()