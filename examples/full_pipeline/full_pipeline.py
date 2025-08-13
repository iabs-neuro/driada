#!/usr/bin/env python3
"""
Complete INTENSE Pipeline Example

This example demonstrates a comprehensive analysis workflow using DRIADA's INTENSE module:
1. Generate larger synthetic dataset with multiple feature types including MultiTimeSeries
2. Run comprehensive analysis with parameter exploration
3. Statistical analysis and significance testing
4. Create a selectivity heatmap showing neuron-feature relationships
5. Performance and parameter sensitivity analysis

This showcases INTENSE's capability to analyze complex neuronal selectivity patterns.
"""

import sys
import os
import time

# Add the src directory to the path to import driada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import driada
import matplotlib.pyplot as plt
import numpy as np
from driada.intense.visual import plot_selectivity_heatmap


def analyze_experiment(exp, analysis_name, n_shuffles_stage2=1000):
    """Run INTENSE analysis on experiment with timing."""
    print(f"\n--- {analysis_name} ---")

    start_time = time.time()

    # Get features that should skip delay optimization
    # Check for MultiTimeSeries instances
    from driada.information.info_base import MultiTimeSeries

    skip_delays = []
    for feat_name, feat_data in exp.dynamic_features.items():
        if isinstance(feat_data, MultiTimeSeries):
            skip_delays.append(feat_name)

    stats, significance, info, results = driada.compute_cell_feat_significance(
        exp,
        mode="two_stage",
        n_shuffles_stage1=50,
        n_shuffles_stage2=n_shuffles_stage2,
        allow_mixed_dimensions=True,  # Enable MultiTimeSeries analysis
        skip_delays=(
            skip_delays if skip_delays else None
        ),  # Skip delays for MultiTimeSeries
        verbose=False,
    )

    analysis_time = time.time() - start_time

    # Extract results
    significant_neurons = exp.get_significant_neurons()
    total_pairs = sum(len(features) for features in significant_neurons.values())

    print(f"Analysis completed in {analysis_time:.2f} seconds")
    print(f"Significant neurons: {len(significant_neurons)}")
    print(f"Total significant pairs: {total_pairs}")

    return stats, significance, info, results, significant_neurons, analysis_time


def create_statistical_summary(exp, significant_neurons):
    """Create comprehensive statistical summary."""
    print("\n=== STATISTICAL SUMMARY ===")

    # Feature-wise analysis
    feature_counts = {}
    all_mi_values = []
    all_pvalues = []

    for cell_id, features in significant_neurons.items():
        for feat_name in features:
            if feat_name not in feature_counts:
                feature_counts[feat_name] = 0
            feature_counts[feat_name] += 1

            # Get statistics
            pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_name)
            all_mi_values.append(pair_stats["pre_rval"])
            if "pval" in pair_stats:
                all_pvalues.append(pair_stats["pval"])

    # Feature selectivity summary
    print("\nFeature selectivity counts:")
    for feat_name, count in sorted(feature_counts.items()):
        percentage = (count / exp.n_cells) * 100
        print(f"  {feat_name}: {count}/{exp.n_cells} neurons ({percentage:.1f}%)")

    # MI statistics
    if all_mi_values:
        print("\nMutual Information statistics:")
        print(f"  Mean MI: {np.mean(all_mi_values):.4f}")
        print(f"  Median MI: {np.median(all_mi_values):.4f}")
        print(f"  Min MI: {np.min(all_mi_values):.4f}")
        print(f"  Max MI: {np.max(all_mi_values):.4f}")
        print(f"  Std MI: {np.std(all_mi_values):.4f}")

    # P-value statistics
    if all_pvalues:
        print("\nP-value statistics:")
        print(f"  Mean p-value: {np.mean(all_pvalues):.2e}")
        print(f"  Median p-value: {np.median(all_pvalues):.2e}")
        print(f"  Min p-value: {np.min(all_pvalues):.2e}")
        print(f"  Max p-value: {np.max(all_pvalues):.2e}")

    return feature_counts, all_mi_values, all_pvalues


def create_selectivity_heatmap(exp, significant_neurons, output_dir):
    """Create a heatmap showing MI values for selective neuron-feature pairs."""
    print("\n=== CREATING SELECTIVITY HEATMAP ===")

    # Use the new plot_selectivity_heatmap function from visual.py
    fig, ax, stats = plot_selectivity_heatmap(
        exp,
        significant_neurons,
        metric="mi",  # Show mutual information values
        cmap="viridis",  # Use viridis colormap for continuous values
        use_log_scale=False,  # Linear scale by default
        figsize=(10, 8),
    )

    # Save heatmap
    output_path = os.path.join(output_dir, "neuronal_selectivity_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Selectivity heatmap saved to: {output_path}")

    # Print statistics
    print("\nHeatmap statistics:")
    print(
        f"  • Matrix size: {exp.n_cells} neurons × {len([f for f in exp.dynamic_features.keys() if isinstance(f, str)])} features"
    )
    print(
        f"  • Selective neurons: {stats['n_selective']} ({stats['selectivity_rate']:.1f}%)"
    )
    print(f"  • Total selective pairs: {stats['n_pairs']}")
    print(f"  • Sparsity: {stats['sparsity']:.1f}%")

    if stats["metric_values"]:
        print("\nMI value statistics:")
        print(f"  • Min: {min(stats['metric_values']):.4f}")
        print(f"  • Max: {max(stats['metric_values']):.4f}")
        print(f"  • Mean: {np.mean(stats['metric_values']):.4f}")
        print(f"  • Median: {np.median(stats['metric_values']):.4f}")
        print(f"  • Std: {np.std(stats['metric_values']):.4f}")

    plt.show()


def main():
    """Run complete INTENSE pipeline example."""
    print("=" * 80)
    print("DRIADA INTENSE - Complete Pipeline Example")
    print("=" * 80)

    output_dir = os.path.dirname(__file__)

    # Step 1: Generate comprehensive synthetic experiment with multifeatures
    print("\n1. GENERATING COMPREHENSIVE SYNTHETIC EXPERIMENT")
    print("   - 50 neurons (larger population)")
    print("   - 3 discrete + 3 continuous features")
    print("   - 2 multifeatures (spatial location, movement)")
    print("   - 10 minutes recording (more data)")

    # Use the advanced generator that creates MultiTimeSeries features
    exp, selectivity_info = (
        driada.experiment.generate_synthetic_exp_with_mixed_selectivity(
            n_discrete_feats=3,  # discrete features
            n_continuous_feats=3,  # continuous features
            n_neurons=50,  # larger population
            n_multifeatures=2,  # create 2 multifeatures (e.g., x,y and speed,direction)
            duration=600,  # 10 minutes
            seed=42,  # reproducible
            fps=20,  # sampling rate
            verbose=False,  # suppress detailed output
        )
    )

    print(f"   ✓ Experiment: {exp.n_cells} neurons × {exp.n_frames} timepoints")
    print(f"   ✓ Recording: {exp.n_frames/exp.fps:.1f}s at {exp.fps} Hz")
    print(f"   ✓ Features: {list(exp.dynamic_features.keys())}")

    # Show multifeatures
    from driada.information.info_base import MultiTimeSeries

    multifeatures = [
        feat
        for feat, data in exp.dynamic_features.items()
        if isinstance(data, MultiTimeSeries)
    ]
    if multifeatures:
        print(f"   ✓ MultiTimeSeries features: {multifeatures}")

    # Step 2: Comprehensive analysis
    print("\n2. COMPREHENSIVE ANALYSIS")

    # Standard analysis
    stats, significance, info, results, significant_neurons, analysis_time = (
        analyze_experiment(
            exp, "Standard Analysis (1000 shuffles)", n_shuffles_stage2=1000
        )
    )

    # High-precision analysis for comparison
    print("\n   Comparing with high-precision analysis...")
    _, _, _, _, significant_neurons_hp, analysis_time_hp = analyze_experiment(
        exp, "High-precision Analysis (2000 shuffles)", n_shuffles_stage2=2000
    )

    print("\n   Performance comparison:")
    print(f"   Standard (1000 shuffles): {analysis_time:.2f}s")
    print(f"   High-precision (2000 shuffles): {analysis_time_hp:.2f}s")
    print(f"   Speed difference: {analysis_time_hp/analysis_time:.1f}x slower")

    # Step 3: Statistical analysis
    feature_counts, all_mi_values, all_pvalues = create_statistical_summary(
        exp, significant_neurons
    )

    # Step 4: Parameter sensitivity analysis
    print("\n=== PARAMETER SENSITIVITY ANALYSIS ===")

    shuffle_counts = [500, 1000, 2000]
    results_comparison = {}

    for n_shuffles in shuffle_counts:
        print(f"\nTesting {n_shuffles} shuffles...")
        start_time = time.time()

        _, _, _, _, sig_neurons, _ = analyze_experiment(
            exp, f"Analysis ({n_shuffles} shuffles)", n_shuffles_stage2=n_shuffles
        )

        runtime = time.time() - start_time
        results_comparison[n_shuffles] = {
            "neurons": len(sig_neurons),
            "pairs": sum(len(features) for features in sig_neurons.values()),
            "runtime": runtime,
        }

    print("\nParameter sensitivity results:")
    print(f"{'Shuffles':<10} {'Neurons':<10} {'Pairs':<10} {'Runtime':<10}")
    print("-" * 40)
    for n_shuffles, result in results_comparison.items():
        print(
            f"{n_shuffles:<10} {result['neurons']:<10} {result['pairs']:<10} {result['runtime']:.2f}s"
        )

    # Step 5: Detailed results for top neurons
    print("\n=== DETAILED RESULTS FOR TOP NEURONS ===")

    if significant_neurons:
        # Sort neurons by number of significant features
        sorted_neurons = sorted(
            significant_neurons.items(), key=lambda x: len(x[1]), reverse=True
        )

        for i, (cell_id, features) in enumerate(sorted_neurons[:5]):  # Top 5 neurons
            print(f"\nNeuron {cell_id} - {len(features)} significant feature(s):")

            for feat_name in features:
                pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_name)

                print(f"  Feature '{feat_name}':")
                print(f"    • MI: {pair_stats['pre_rval']:.4f}")
                if "pval" in pair_stats:
                    print(f"    • P-value: {pair_stats['pval']:.2e}")
                print(f"    • Optimal delay: {pair_stats.get('shift_used', 0):.2f}s")

                # Additional statistics if available
                if "me" in pair_stats and pair_stats["me"] is not None:
                    print(f"    • Modulation effect: {pair_stats['me']:.4f}")

    # Step 6: Create selectivity heatmap
    create_selectivity_heatmap(exp, significant_neurons, output_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE ANALYSIS FINISHED")
    print("=" * 80)

    print("\nFinal Summary:")
    print(f"• Dataset: {exp.n_cells} neurons, {len(exp.dynamic_features)} features")
    print(
        f"• Significant neurons: {len(significant_neurons)}/{exp.n_cells} ({len(significant_neurons)/exp.n_cells*100:.1f}%)"
    )
    print(
        f"• Total significant pairs: {sum(len(features) for features in significant_neurons.values())}"
    )
    print(f"• Analysis time: {analysis_time:.2f} seconds")

    if all_mi_values:
        print(f"• MI range: {np.min(all_mi_values):.4f} - {np.max(all_mi_values):.4f}")

    print(f"\nOutput files created in: {output_dir}")
    print("• neuronal_selectivity_heatmap.png - Selectivity matrix visualization")

    print("\nNext steps:")
    print("• Try mixed_selectivity.py for disentanglement analysis")
    print("• Modify parameters to explore different scenarios")
    print("• Use results for publication with higher n_shuffles_stage2 (10000+)")


if __name__ == "__main__":
    main()
