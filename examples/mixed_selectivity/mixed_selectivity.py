#!/usr/bin/env python3
"""
Mixed Selectivity Analysis Example

This example demonstrates advanced disentanglement analysis using DRIADA's INTENSE module:
1. Generate synthetic data with mixed selectivity patterns and MultiTimeSeries features
2. Run standard INTENSE analysis to find significant relationships
3. Apply disentanglement analysis to separate mixed selectivity
4. Visualize disentanglement results and interpret findings
5. Demonstrate redundancy vs synergy detection

This showcases INTENSE's advanced capabilities for analyzing neurons that respond
to multiple correlated behavioral variables, including multivariate features.
"""

import sys
import os

# Add the src directory to the path to import driada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import driada
import matplotlib.pyplot as plt
import numpy as np


def generate_mixed_selectivity_data():
    """Generate synthetic data with known mixed selectivity patterns and MultiTimeSeries."""
    print("\n=== GENERATING MIXED SELECTIVITY DATA ===")

    # For demonstration, create a smaller experiment with stronger selectivity
    exp, selectivity_info = (
        driada.experiment.generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=2,  # Fewer features for clearer demonstration
            n_continuous_feats=2,  # Fewer continuous features
            n_neurons=30,  # Smaller population for faster analysis
            n_multifeatures=1,  # One multifeature
            selectivity_prob=1.0,  # All neurons are selective
            multi_select_prob=0.8,  # Most have mixed selectivity
            weights_mode="dominant",  # One feature dominates (clearer for disentanglement)
            duration=600,  # 10 minutes
            seed=42,
            fps=20,
            verbose=False,
            create_discrete_pairs=True,  # Create discrete versions of continuous signals for disentanglement demo
            rate_0=0.1,
            rate_1=3.0,
            skip_prob=0.0,
            ampl_range=(1.5, 3.5),
            decay_time=2,
            noise_std=0.05,
        )
    )

    print(
        f"Generated experiment: {exp.n_cells} neurons, {len(exp.dynamic_features)} features, {exp.n_frames/exp.fps:.1f}s recording"
    )

    return exp, selectivity_info


def run_intense_analysis(exp):
    """Run INTENSE analysis to identify significant relationships."""
    print("\n=== RUNNING INTENSE ANALYSIS ===")

    # Run comprehensive analysis including multifeatures
    # Get features that should skip delay optimization
    # Check both tuple names (multifeatures) and MultiTimeSeries instances
    skip_delays = []
    for feat_name, feat_data in exp.dynamic_features.items():
        if isinstance(feat_name, tuple):  # Tuple name indicates multifeature
            skip_delays.append(feat_name)
        elif isinstance(feat_data, driada.MultiTimeSeries):
            skip_delays.append(feat_name)

    # Run comprehensive analysis with disentanglement
    results = driada.compute_cell_feat_significance(
        exp,
        mode="two_stage",
        n_shuffles_stage1=50,  # Increased for better statistics
        n_shuffles_stage2=500,  # Increased for more reliable p-values
        allow_mixed_dimensions=True,  # Enable MultiTimeSeries analysis
        skip_delays=(
            skip_delays if skip_delays else None
        ),  # Skip delay optimization for MultiTimeSeries
        verbose=False,
        with_disentanglement=True,  # Enable disentanglement analysis
        multifeature_map=driada.intense.DEFAULT_MULTIFEATURE_MAP,
        metric_distr_type="norm",  # Use normal (Gaussian) distribution for shuffled MI
        pval_thr=0.05,  # Slightly less conservative threshold
    )

    stats, significance, info, intense_results, disentanglement_results = results

    # Extract significant relationships
    significant_neurons = exp.get_significant_neurons()

    # Also get neurons with mixed selectivity (at least 2 features)
    mixed_selectivity_neurons = exp.get_significant_neurons(min_nspec=2)

    # Count multifeature relationships
    from driada.information.info_base import MultiTimeSeries

    multifeature_count = 0
    for cell_id, features in significant_neurons.items():
        for feat in features:
            if feat in exp.dynamic_features and isinstance(
                exp.dynamic_features[feat], MultiTimeSeries
            ):
                multifeature_count += 1

    # Use the pre-filtered mixed selectivity neurons
    mixed_candidates = mixed_selectivity_neurons

    print(
        f"Found {len(significant_neurons)} significant neurons, {len(mixed_candidates)} with mixed selectivity"
    )

    return (
        stats,
        significance,
        info,
        intense_results,
        significant_neurons,
        mixed_candidates,
        disentanglement_results,
    )


def analyze_disentanglement(disentanglement_results, mixed_candidates):
    """Process disentanglement results from the pipeline."""
    print("\n=== DISENTANGLEMENT ANALYSIS ===")

    if not mixed_candidates:
        print("No mixed selectivity candidates found for disentanglement.")
        return None, None, None

    if disentanglement_results is None:
        print("No disentanglement results available.")
        return None, None, None

    # Extract results from the pipeline
    disent_matrix = disentanglement_results.get("disent_matrix")
    count_matrix = disentanglement_results.get("count_matrix")
    feat_names = disentanglement_results.get("feature_names", [])

    if disent_matrix is None or count_matrix is None:
        print("Disentanglement matrices not found in results.")
        return None, None, None

    print("Disentanglement analysis completed by pipeline")
    print(
        f"Matrix shape: {disent_matrix.shape}, Non-zero entries: {np.count_nonzero(count_matrix)}"
    )
    print(f"Feature names analyzed: {feat_names}")

    # Show summary if available
    if "summary" in disentanglement_results:
        summary = disentanglement_results["summary"]
        if "overall_stats" in summary:
            stats = summary["overall_stats"]
            print("\nOverall statistics:")
            print(f"  Total neuron pairs: {stats.get('total_neuron_pairs', 0)}")
            print(f"  Redundancy rate: {stats.get('redundancy_rate', 0):.1f}%")
            print(
                f"  True mixed selectivity rate: {stats.get('true_mixed_selectivity_rate', 0):.1f}%"
            )

    return disent_matrix, count_matrix, feat_names


def interpret_disentanglement_results(
    exp, disent_matrix, count_matrix, feat_names, mixed_candidates
):
    """Interpret and summarize disentanglement analysis results from matrices."""
    print("\n=== INTERPRETING DISENTANGLEMENT RESULTS ===")

    if disent_matrix is None or count_matrix is None:
        return [], [], []

    redundancy_cases = []
    synergy_cases = []
    independence_cases = []

    # Calculate relative disentanglement matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_disent_matrix = np.divide(disent_matrix, count_matrix) * 100
        rel_disent_matrix[count_matrix == 0] = np.nan

    # Extract disentanglement cases based on matrix values
    for i in range(len(feat_names)):
        for j in range(
            i + 1, len(feat_names)
        ):  # Only upper triangle to avoid duplicates
            if count_matrix[i, j] > 0:  # Only consider pairs with data
                feat1 = feat_names[i]
                feat2 = feat_names[j]

                # Get disentanglement score (percentage)
                disent_score = rel_disent_matrix[i, j]

                if not np.isnan(disent_score):
                    # Classify based on disentanglement score
                    if disent_score < 30:  # Redundancy: feat2 dominates
                        redundancy_cases.append(
                            (f"{feat1}-{feat2}", (feat1, feat2), disent_score / 100)
                        )
                    elif disent_score > 70:  # Synergy: feat1 dominates
                        synergy_cases.append(
                            (f"{feat1}-{feat2}", (feat1, feat2), disent_score / 100)
                        )
                    else:  # Independence: balanced
                        independence_cases.append(
                            (f"{feat1}-{feat2}", (feat1, feat2), disent_score / 100)
                        )

    # Summary statistics
    total_pairs = len(redundancy_cases) + len(synergy_cases) + len(independence_cases)
    print(
        f"Found {len(redundancy_cases)} redundancy, {len(independence_cases)} independence, {len(synergy_cases)} synergy cases"
    )

    # Show a few examples if available
    if redundancy_cases and len(redundancy_cases) > 0:
        feat1, feat2 = redundancy_cases[0][1]
        print(f"Example redundancy: {feat1} ↔ {feat2}")
    elif synergy_cases and len(synergy_cases) > 0:
        feat1, feat2 = synergy_cases[0][1]
        print(f"Example synergy: {feat1} + {feat2}")

    return redundancy_cases, synergy_cases, independence_cases


def create_visualizations(
    exp,
    disent_matrix,
    count_matrix,
    feat_names,
    significant_neurons,
    mixed_candidates,
    output_dir,
):
    """Create disentanglement and selectivity visualizations."""
    print("\n=== CREATING VISUALIZATIONS ===")

    # First, create neuron-feature selectivity matrix
    try:
        print("\nCreating neuron-feature selectivity heatmap...")
        fig_select, ax_select, stats_select = driada.intense.plot_selectivity_heatmap(
            exp, significant_neurons, metric="mi", use_log_scale=False, figsize=(12, 10)
        )

        # Save selectivity heatmap
        select_path = os.path.join(output_dir, "neuron_feature_selectivity.png")
        fig_select.savefig(select_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {select_path}")
        print(
            f"  - {stats_select['n_selective']} selective neurons ({stats_select['selectivity_rate']:.1f}%)"
        )
        print(f"  - {stats_select['n_pairs']} neuron-feature pairs")

    except Exception as e:
        print(f"Error creating selectivity heatmap: {str(e)}")
        fig_select = None

    # Create disentanglement heatmap if we have results
    if disent_matrix is not None and count_matrix is not None:

        try:
            print("\nCreating disentanglement heatmap...")
            fig_disent, ax_disent = driada.intense.plot_disentanglement_heatmap(
                disent_matrix,
                count_matrix,
                feat_names,
                title="Feature Disentanglement Analysis",
                figsize=(10, 8),
            )

            # Save heatmap
            heatmap_path = os.path.join(output_dir, "disentanglement_heatmap.png")
            fig_disent.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {heatmap_path}")

        except Exception as e:
            print(f"Error creating disentanglement heatmap: {str(e)}")
            fig_disent = None
    else:
        print("No disentanglement results to visualize")
        fig_disent = None

    # Show plots
    if fig_select is not None:
        plt.figure(fig_select.number)
        plt.show()
    if fig_disent is not None:
        plt.figure(fig_disent.number)
        plt.show()


def main():
    """Run mixed selectivity analysis example."""
    print("=" * 80)
    print("DRIADA INTENSE - Mixed Selectivity Analysis Example")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("1. Generating synthetic data with neurons that respond to multiple features")
    print(
        "2. Running INTENSE analysis to detect significant neuron-feature relationships"
    )
    print(
        "3. Using get_significant_neurons(min_nspec=2) to filter mixed selectivity neurons"
    )
    print("4. Applying disentanglement analysis to separate overlapping selectivities")
    print("5. Visualizing results with heatmaps and summary plots")

    output_dir = os.path.dirname(__file__)

    # Step 1: Generate mixed selectivity data with MultiTimeSeries
    exp, selectivity_info = generate_mixed_selectivity_data()

    # Step 2: Run INTENSE analysis with disentanglement
    (
        stats,
        significance,
        info,
        intense_results,
        significant_neurons,
        mixed_candidates,
        disentanglement_results,
    ) = run_intense_analysis(exp)

    # Step 3: Process disentanglement results
    disent_matrix, count_matrix, feat_names = analyze_disentanglement(
        disentanglement_results, mixed_candidates
    )

    # Step 4: Interpret results
    redundancy_cases, synergy_cases, independence_cases = (
        interpret_disentanglement_results(
            exp, disent_matrix, count_matrix, feat_names, mixed_candidates
        )
    )

    # Step 5: Create visualizations
    create_visualizations(
        exp,
        disent_matrix,
        count_matrix,
        feat_names,
        significant_neurons,
        mixed_candidates,
        output_dir,
    )

    print("\nVisualization files created:")
    print(
        "  - neuron_feature_selectivity.png: Shows MI values for all neuron-feature pairs"
    )
    print("  - disentanglement_heatmap.png: Shows feature relationship disentanglement")


if __name__ == "__main__":
    main()
