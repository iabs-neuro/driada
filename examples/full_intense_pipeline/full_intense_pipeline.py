#!/usr/bin/env python3
"""
Complete INTENSE Pipeline Example with Principled Continuous Features

This example demonstrates INTENSE analysis across ALL feature types with
ground truth validation:

1. Circular features (head direction) - von Mises tuning (HD cells)
2. Spatial features (x, y coordinates) - Gaussian place fields (place cells)
3. Linear features (running speed) - sigmoid tuning (speed cells)
4. Discrete features (binary events) - event-locked responses
5. Mixed selectivity neurons - respond to multiple feature types

Key outputs:
- Ground truth selectivity matrix
- INTENSE detection results
- Validation metrics (sensitivity, precision, F1)
- Publication-quality visualizations
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import driada
from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.information.info_base import MultiTimeSeries


# =============================================================================
# CONFIGURATION
# =============================================================================
# Population configuration - defines neuron groups and their selectivity
POPULATION = [
    {"name": "hd_cells", "count": 4, "features": ["head_direction"]},
    {"name": "place_cells", "count": 4, "features": ["x", "y"], "combination": "and"},
    {"name": "speed_cells", "count": 4, "features": ["speed"]},
    {"name": "event_cells", "count": 4, "features": ["event_0"]},
    {"name": "mixed_cells", "count": 4, "features": ["head_direction", "event_0"]},
    {"name": "nonselective", "count": 4, "features": []},
]

# Analysis parameters
CONFIG = {
    # Recording parameters
    "duration": 900,        # seconds (15 min - longer for better place cell sampling)
    "fps": 20,              # sampling rate
    "seed": 42,
    # Tuning parameters
    "kappa": 4.0,           # von Mises concentration (HD cells) - increased for sharper tuning
    "place_sigma": 0.15,    # place field width - decreased for sharper fields
    # Calcium dynamics
    "baseline_rate": 0.02,  # baseline firing rate - reduced for better contrast
    "peak_rate": 2.5,       # peak response - increased for stronger signal
    "decay_time": 1.5,      # faster decay for sharper responses
    "calcium_noise": 0.01,  # noise level (reduced for stronger signal)
    # Discrete event parameters
    "n_discrete_features": 2,
    "event_active_fraction": 0.08,  # ~8% active time per event
    "event_avg_duration": 0.8,      # seconds
    # INTENSE analysis parameters
    "n_shuffles_stage1": 100,   # stage 1 screening shuffles
    "n_shuffles_stage2": 5000,  # stage 2 confirmation shuffles
    "pval_thr": 0.01,           # p-value threshold after correction
    "multicomp_correction": "holm",  # multiple comparison correction
}


# =============================================================================
# ANALYSIS
# =============================================================================
def run_intense_analysis(exp, config, verbose=True):
    """Run INTENSE analysis on the experiment."""
    if verbose:
        print("\n" + "=" * 60)
        print("RUNNING INTENSE ANALYSIS")
        print("=" * 60)
        print(f"  Stage 1: {config['n_shuffles_stage1']} shuffles")
        print(f"  Stage 2: {config['n_shuffles_stage2']} shuffles")
        print(f"  P-value threshold: {config['pval_thr']}")

    start_time = time.time()

    # Identify MultiTimeSeries features (skip delay optimization for these)
    skip_delays = [
        feat_name for feat_name, feat_data in exp.dynamic_features.items()
        if isinstance(feat_data, MultiTimeSeries)
    ]

    # Build feature list excluding x and y marginals (use position_2d instead)
    # This avoids spurious detections on place cell marginals
    feat_bunch = [
        feat_name for feat_name in exp.dynamic_features.keys()
        if feat_name not in ["x", "y"]
    ]
    if verbose:
        print(f"  Features to test: {feat_bunch}")

    # Run INTENSE with disentanglement to handle correlated features
    stats, significance, info, results, disent_results = driada.compute_cell_feat_significance(
        exp,
        feat_bunch=feat_bunch,
        mode="two_stage",
        n_shuffles_stage1=config["n_shuffles_stage1"],
        n_shuffles_stage2=config["n_shuffles_stage2"],
        allow_mixed_dimensions=True,
        skip_delays=skip_delays if skip_delays else None,
        ds=5,
        pval_thr=config["pval_thr"],
        multicomp_correction=config["multicomp_correction"],
        use_precomputed_stats=False,  # Force fresh computation
        with_disentanglement=True,  # Enable disentanglement for correlated features
        verbose=True,
    )

    analysis_time = time.time() - start_time
    significant_neurons = exp.get_significant_neurons()

    if verbose:
        total_pairs = sum(len(features) for features in significant_neurons.values())
        print(f"\n  Completed in {analysis_time:.1f} seconds")
        print(f"  Significant neurons: {len(significant_neurons)}/{exp.n_cells}")
        print(f"  Total significant pairs: {total_pairs}")

    return results, significant_neurons, analysis_time, disent_results


# =============================================================================
# DISENTANGLEMENT REPORTING
# =============================================================================
def print_disentanglement_summary(disent_results, ground_truth, significant_neurons):
    """Print disentanglement analysis summary."""
    print("\n" + "=" * 60)
    print("DISENTANGLEMENT ANALYSIS")
    print("=" * 60)

    if disent_results is None:
        print("  Disentanglement not performed.")
        return

    # Extract results
    disent_matrix = disent_results.get("disent_matrix")
    count_matrix = disent_results.get("count_matrix")
    feat_names = disent_results.get("feature_names", [])
    summary = disent_results.get("summary", {})

    if disent_matrix is None or count_matrix is None:
        print("  No disentanglement data available.")
        return

    # Print feature correlations from summary
    if "overall_stats" in summary:
        stats = summary["overall_stats"]
        print(f"\n  Overall Statistics:")
        print(f"    Total neuron-feature pairs analyzed: {stats.get('total_neuron_pairs', 0)}")
        print(f"    Redundancy rate: {stats.get('redundancy_rate', 0):.1f}%")
        print(f"    True mixed selectivity rate: {stats.get('true_mixed_selectivity_rate', 0):.1f}%")

    # Print pairwise results
    if "pairwise_stats" in summary:
        print(f"\n  Pairwise Feature Analysis:")
        for pair_key, pair_stats in summary["pairwise_stats"].items():
            feat1, feat2 = pair_key
            n_pairs = pair_stats.get("n_pairs", 0)
            if n_pairs > 0:
                feat1_pct = pair_stats.get("feat1_primary_pct", 50)
                feat2_pct = pair_stats.get("feat2_primary_pct", 50)
                print(f"    {feat1} vs {feat2} (n={n_pairs}):")
                print(f"      {feat1} primary: {feat1_pct:.0f}%")
                print(f"      {feat2} primary: {feat2_pct:.0f}%")

    # Identify redundant detections based on ground truth
    print(f"\n  Redundancy Analysis for False Positives:")
    expected_pairs = set(ground_truth["expected_pairs"])
    neuron_types = ground_truth["neuron_types"]

    redundant_fps = []
    for neuron_id, features in significant_neurons.items():
        neuron_type = neuron_types.get(neuron_id, "unknown")
        for feat_name in features:
            if (neuron_id, feat_name) not in expected_pairs:
                # This is a false positive - check if it's likely redundant
                redundant_fps.append((neuron_id, neuron_type, feat_name))

    if redundant_fps:
        for neuron_id, neuron_type, feat_name in redundant_fps:
            # Determine likely primary feature based on neuron type
            if neuron_type == "hd_cells" and feat_name == "position_2d":
                print(f"    Neuron {neuron_id} ({neuron_type}) -> {feat_name}: "
                      f"REDUNDANT (head_direction is primary)")
            elif neuron_type == "place_cells" and feat_name == "head_direction":
                print(f"    Neuron {neuron_id} ({neuron_type}) -> {feat_name}: "
                      f"REDUNDANT (position_2d is primary)")
            elif neuron_type == "speed_cells" and feat_name in ["head_direction", "position_2d"]:
                print(f"    Neuron {neuron_id} ({neuron_type}) -> {feat_name}: "
                      f"REDUNDANT (speed is primary)")
            else:
                print(f"    Neuron {neuron_id} ({neuron_type}) -> {feat_name}: "
                      f"Unknown (may be true mixed or noise)")
    else:
        print("    No false positives to analyze.")


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(exp, significant_neurons, ground_truth, metrics, output_dir):
    """Create comprehensive visualization."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # 1. Selectivity heatmap (main plot)
    ax1 = fig.add_subplot(2, 2, (1, 2))

    # Get feature names and create matrix
    feature_names = [f for f in exp.dynamic_features.keys()
                    if not isinstance(exp.dynamic_features[f], MultiTimeSeries)]
    n_neurons = exp.n_cells
    n_features = len(feature_names)

    # Create MI matrix
    mi_matrix = np.zeros((n_neurons, n_features))
    for neuron_id, features in significant_neurons.items():
        for feat_name in features:
            if feat_name in feature_names:
                feat_idx = feature_names.index(feat_name)
                pair_stats = exp.get_neuron_feature_pair_stats(neuron_id, feat_name)
                mi_matrix[neuron_id, feat_idx] = pair_stats.get("pre_rval", 0)

    im = ax1.imshow(mi_matrix, aspect="auto", cmap="viridis")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Neurons")
    ax1.set_title("INTENSE Selectivity Heatmap (MI values)")
    ax1.set_xticks(range(n_features))
    ax1.set_xticklabels(feature_names, rotation=45, ha="right")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Mutual Information")

    # Add neuron type annotations
    type_colors = {
        "hd_cells": "red",
        "place_cells": "blue",
        "speed_cells": "green",
        "event_cells": "orange",
        "mixed_cells": "purple",
        "nonselective": "gray",
    }
    for neuron_id, neuron_type in ground_truth["neuron_types"].items():
        color = type_colors.get(neuron_type, "gray")
        ax1.scatter(-0.7, neuron_id, c=color, s=20, marker="s")

    # 2. Detection rates by type
    ax2 = fig.add_subplot(2, 2, 3)
    types = list(metrics["type_stats"].keys())
    sensitivities = [metrics["type_stats"][t]["sensitivity"] * 100 for t in types]
    colors = [type_colors.get(t, "gray") for t in types]

    bars = ax2.bar(range(len(types)), sensitivities, color=colors)
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax2.set_ylabel("Detection Rate (%)")
    ax2.set_title("Detection Rate by Neuron Type")
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color="k", linestyle="--", alpha=0.3)

    # Add percentage labels
    for bar, pct in zip(bars, sensitivities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    # 3. Summary statistics
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.axis("off")

    summary_text = (
        f"VALIDATION SUMMARY\n"
        f"{'=' * 30}\n\n"
        f"Overall Metrics:\n"
        f"  Sensitivity: {metrics['sensitivity']:.1%}\n"
        f"  Precision:   {metrics['precision']:.1%}\n"
        f"  F1 Score:    {metrics['f1']:.1%}\n\n"
        f"Detection Counts:\n"
        f"  True Positives:  {metrics['true_positives']}\n"
        f"  False Positives: {metrics['false_positives']}\n"
        f"  False Negatives: {metrics['false_negatives']}\n\n"
        f"Population:\n"
        f"  Total neurons: {exp.n_cells}\n"
        f"  Total features: {len(exp.dynamic_features)}\n"
        f"  Expected pairs: {len(ground_truth['expected_pairs'])}\n"
    )
    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
            fontfamily="monospace", fontsize=10, verticalalignment="top")

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, "neuronal_selectivity_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run complete INTENSE pipeline with principled continuous features."""
    print("=" * 70)
    print("DRIADA INTENSE - Principled Continuous Features Demo")
    print("=" * 70)

    output_dir = os.path.dirname(__file__)

    # Step 1: Generate experiment with ground truth
    print("\n[1] GENERATING SYNTHETIC EXPERIMENT")
    print("-" * 40)

    # Custom tuning defaults based on config
    tuning_defaults = {
        "head_direction": {"kappa": CONFIG["kappa"]},
        "x": {"sigma": CONFIG["place_sigma"]},
        "y": {"sigma": CONFIG["place_sigma"]},
    }

    exp, ground_truth = generate_tuned_selectivity_exp(
        population=POPULATION,
        tuning_defaults=tuning_defaults,
        duration=CONFIG["duration"],
        fps=CONFIG["fps"],
        baseline_rate=CONFIG["baseline_rate"],
        peak_rate=CONFIG["peak_rate"],
        decay_time=CONFIG["decay_time"],
        calcium_noise=CONFIG["calcium_noise"],
        n_discrete_features=CONFIG["n_discrete_features"],
        event_active_fraction=CONFIG["event_active_fraction"],
        event_avg_duration=CONFIG["event_avg_duration"],
        seed=CONFIG["seed"],
        verbose=True,
    )

    # Step 2: Run INTENSE analysis with disentanglement
    print("\n[2] RUNNING INTENSE ANALYSIS")
    print("-" * 40)
    results, significant_neurons, analysis_time, disent_results = run_intense_analysis(
        exp, CONFIG, verbose=True
    )

    # Step 3: Validate against ground truth using IntenseResults method
    print("\n[3] VALIDATING AGAINST GROUND TRUTH")
    print("-" * 40)
    metrics = results.validate_against_ground_truth(ground_truth, verbose=True)

    # Step 4: Disentanglement analysis
    print("\n[4] DISENTANGLEMENT ANALYSIS")
    print("-" * 40)
    print_disentanglement_summary(disent_results, ground_truth, significant_neurons)

    # Step 5: Create visualizations
    print("\n[5] CREATING VISUALIZATIONS")
    print("-" * 40)
    create_visualizations(exp, significant_neurons, ground_truth, metrics, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    n_expected = metrics['true_positives'] + metrics['false_negatives']
    print(f"  - Sensitivity: {metrics['sensitivity']:.1%} "
          f"(detected {metrics['true_positives']}/{n_expected} expected pairs)")
    print(f"  - Precision:   {metrics['precision']:.1%} "
          f"({metrics['false_positives']} false positives)")
    print(f"  - F1 Score:    {metrics['f1']:.1%}")
    print(f"  - Analysis time: {analysis_time:.1f}s")

    print(f"\nFeature Type Coverage:")
    for neuron_type, stats in sorted(metrics["type_stats"].items()):
        status = "OK" if stats["sensitivity"] >= 0.5 else "LOW"
        print(f"  [{status}] {neuron_type}: {stats['detected']}/{stats['expected']} detected")

    print(f"\nOutput: {os.path.join(output_dir, 'neuronal_selectivity_heatmap.png')}")


if __name__ == "__main__":
    main()
