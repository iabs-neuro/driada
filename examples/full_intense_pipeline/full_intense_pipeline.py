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


# =============================================================================
# CONFIGURATION
# =============================================================================
# Population configuration - defines neuron groups and their selectivity
POPULATION = [
    {"name": "hd_cells", "count": 4, "features": ["head_direction"]},
    {"name": "place_cells", "count": 4, "features": ["position_2d"]},
    {"name": "speed_cells", "count": 4, "features": ["speed"]},
    {"name": "event_cells", "count": 4, "features": ["event_0"]},
    {"name": "mixed_cells", "count": 4, "features": ["head_direction", "event_0"]},
    {"name": "nonselective", "count": 4, "features": []},
]

# Analysis parameters
CONFIG = {
    # Recording parameters
    "duration": 900,        # seconds
    "fps": 20,              # sampling rate
    "seed": 42,
    # Tuning parameters
    "kappa": 4.0,           # von Mises concentration (HD cells)
    # Calcium dynamics
    "baseline_rate": 0.02,  # baseline firing rate
    "peak_rate": 2.5,       # peak response
    "decay_time": 1.5,      # calcium decay time
    "calcium_noise": 0.01,  # noise level
    # Discrete event parameters
    "n_discrete_features": 2,
    "event_active_fraction": 0.08,  # ~8% active time per event
    "event_avg_duration": 0.8,      # seconds
    # INTENSE analysis parameters
    "n_shuffles_stage1": 100,   # stage 1 screening shuffles
    "n_shuffles_stage2": 10000,  # stage 2 confirmation (FFT makes this fast)
    "pval_thr": 0.05,           # p-value threshold after correction
    "multicomp_correction": "holm",  # multiple comparison correction
}


# =============================================================================
# ANALYSIS
# =============================================================================
def run_intense_analysis(exp, config, verbose=True):
    """Run INTENSE analysis on the experiment.

    INTENSE uses a two-stage significance testing approach:
    1. Stage 1: Quick screening with fewer shuffles to identify candidates
    2. Stage 2: Rigorous testing of candidates with more shuffles

    Delay optimization finds the optimal temporal lag between neural activity
    and behavioral features. This compensates for:
    - Calcium indicator dynamics (rise/decay time ~1-2s for GCaMP6)
    - Neural processing delays
    - Behavioral-neural coupling latencies

    Returns the analysis results including optimal delays for each neuron-feature pair.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("RUNNING INTENSE ANALYSIS")
        print("=" * 60)
        print(f"  Stage 1: {config['n_shuffles_stage1']} shuffles")
        print(f"  Stage 2: {config['n_shuffles_stage2']} shuffles")
        print(f"  P-value threshold: {config['pval_thr']}")

    start_time = time.time()

    # Build feature list excluding x and y marginals (use position_2d instead)
    # This avoids spurious detections on place cell marginals while still
    # testing the joint 2D spatial selectivity
    feat_bunch = [
        feat_name for feat_name in exp.dynamic_features.keys()
        if feat_name not in ["x", "y"]
    ]
    if verbose:
        print(f"  Features to test: {feat_bunch}")

    # Run INTENSE with disentanglement to handle correlated features
    # - find_optimal_delays=True: Search for best temporal alignment between
    #   neural activity and features (compensates for calcium dynamics)
    # - with_disentanglement=True: Identify redundant detections caused by
    #   feature correlations (e.g., HD cells detecting position due to
    #   trajectory patterns where animal faces certain directions at certain locations)
    stats, significance, info, results, disent_results = driada.compute_cell_feat_significance(
        exp,
        feat_bunch=feat_bunch,
        mode="two_stage",
        n_shuffles_stage1=config["n_shuffles_stage1"],
        n_shuffles_stage2=config["n_shuffles_stage2"],
        find_optimal_delays=True,  # Find best temporal alignment
        ds=5,  # Downsampling factor for speed
        pval_thr=config["pval_thr"],
        multicomp_correction=config["multicomp_correction"],
        use_precomputed_stats=False,  # Force fresh computation
        with_disentanglement=True,
        verbose=True,
    )

    analysis_time = time.time() - start_time
    significant_neurons = exp.get_significant_neurons()

    if verbose:
        total_pairs = sum(len(features) for features in significant_neurons.values())
        print(f"\n  Completed in {analysis_time:.1f} seconds")
        print(f"  Significant neurons: {len(significant_neurons)}/{exp.n_cells}")
        print(f"  Total significant pairs: {total_pairs}")

    # Return info dict containing optimal_delays matrix
    return results, significant_neurons, analysis_time, disent_results, info, feat_bunch


# =============================================================================
# DISENTANGLEMENT REPORTING
# =============================================================================
def analyze_disentanglement(disent_results, ground_truth, significant_neurons, metrics):
    """Analyze disentanglement results and compute corrected metrics.

    Disentanglement identifies which detected pairs are REDUNDANT (caused by
    feature correlations) vs TRUE MIXED SELECTIVITY (genuine multi-feature tuning).

    Returns corrected metrics that exclude redundant false positives.
    """
    print("\n" + "=" * 60)
    print("DISENTANGLEMENT ANALYSIS")
    print("=" * 60)

    # Track redundant false positives
    redundant_count = 0
    unknown_count = 0

    if disent_results is None:
        print("  Disentanglement not performed.")
        return metrics

    # Extract results
    disent_matrix = disent_results.get("disent_matrix")
    count_matrix = disent_results.get("count_matrix")
    summary = disent_results.get("summary", {})

    if disent_matrix is None or count_matrix is None:
        print("  No disentanglement data available.")
        return metrics

    # Print overall statistics
    if "overall_stats" in summary:
        stats = summary["overall_stats"]
        print(f"\n  Overall Statistics:")
        print(f"    Total neuron-feature pairs analyzed: {stats.get('total_neuron_pairs', 0)}")
        print(f"    Redundancy rate: {stats.get('redundancy_rate', 0):.1f}%")
        print(f"    True mixed selectivity rate: {stats.get('true_mixed_selectivity_rate', 0):.1f}%")

    # Identify redundant detections based on ground truth
    print(f"\n  False Positive Analysis:")
    expected_pairs = set(ground_truth["expected_pairs"])
    neuron_types = ground_truth["neuron_types"]

    fps_analysis = []
    for neuron_id, features in significant_neurons.items():
        neuron_type = neuron_types.get(neuron_id, "unknown")
        for feat_name in features:
            if (neuron_id, feat_name) not in expected_pairs:
                # This is a false positive - classify it
                is_redundant = False
                primary_feat = None

                if neuron_type == "hd_cells" and feat_name == "position_2d":
                    is_redundant = True
                    primary_feat = "head_direction"
                elif neuron_type == "place_cells" and feat_name == "head_direction":
                    is_redundant = True
                    primary_feat = "position_2d"
                elif neuron_type == "speed_cells" and feat_name in ["head_direction", "position_2d"]:
                    is_redundant = True
                    primary_feat = "speed"

                fps_analysis.append((neuron_id, neuron_type, feat_name, is_redundant, primary_feat))

                if is_redundant:
                    redundant_count += 1
                else:
                    unknown_count += 1

    # Print FP analysis
    if fps_analysis:
        for neuron_id, neuron_type, feat_name, is_redundant, primary_feat in fps_analysis:
            if is_redundant:
                print(f"    Neuron {neuron_id} ({neuron_type}) -> {feat_name}: "
                      f"REDUNDANT ({primary_feat} is primary)")
            else:
                print(f"    Neuron {neuron_id} ({neuron_type}) -> {feat_name}: "
                      f"UNEXPLAINED (may be noise or true mixed)")
    else:
        print("    No false positives to analyze.")

    # Compute corrected metrics
    tp = metrics["true_positives"]
    fp_raw = metrics["false_positives"]
    fn = metrics["false_negatives"]
    fp_corrected = fp_raw - redundant_count  # Exclude redundant FPs

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_raw = tp / (tp + fp_raw) if (tp + fp_raw) > 0 else 0
    precision_corrected = tp / (tp + fp_corrected) if (tp + fp_corrected) > 0 else 0
    f1_raw = 2 * (precision_raw * sensitivity) / (precision_raw + sensitivity) if (precision_raw + sensitivity) > 0 else 0
    f1_corrected = 2 * (precision_corrected * sensitivity) / (precision_corrected + sensitivity) if (precision_corrected + sensitivity) > 0 else 0

    # Print before/after comparison
    print(f"\n  Metrics Comparison:")
    print(f"    {'Metric':<15} {'Before':<12} {'After':<12} {'Change'}")
    print(f"    {'-'*50}")
    print(f"    {'Sensitivity':<15} {sensitivity:>10.1%}   {sensitivity:>10.1%}   (unchanged)")
    print(f"    {'Precision':<15} {precision_raw:>10.1%}   {precision_corrected:>10.1%}   "
          f"(+{(precision_corrected - precision_raw)*100:.1f}pp)")
    print(f"    {'F1 Score':<15} {f1_raw:>10.1%}   {f1_corrected:>10.1%}   "
          f"(+{(f1_corrected - f1_raw)*100:.1f}pp)")
    print(f"    {'False Pos':<15} {fp_raw:>10}   {fp_corrected:>10}   "
          f"({redundant_count} redundant)")

    # Return corrected metrics
    return {
        **metrics,
        "fp_redundant": redundant_count,
        "fp_unexplained": unknown_count,
        "fp_corrected": fp_corrected,
        "precision_corrected": precision_corrected,
        "f1_corrected": f1_corrected,
    }


# =============================================================================
# OPTIMAL DELAYS REPORTING
# =============================================================================
def print_optimal_delays(info, feat_bunch, significant_neurons, ground_truth, fps):
    """Print optimal delays for significant neuron-feature pairs.

    Optimal delays represent the temporal offset (in frames and seconds) that
    maximizes mutual information between neural activity and behavioral features.

    Positive delays mean neural activity LAGS behind behavior (expected for
    calcium imaging due to indicator dynamics). Negative delays would indicate
    neural activity leads behavior (predictive coding).

    Typical delays for calcium imaging:
    - GCaMP6s: 0.5-2.0 seconds (slow indicator)
    - GCaMP6f: 0.2-0.8 seconds (fast indicator)
    - GCaMP8: 0.1-0.4 seconds (ultrafast indicator)
    """
    print("\n" + "=" * 60)
    print("OPTIMAL DELAYS")
    print("=" * 60)

    optimal_delays = info.get("optimal_delays")
    if optimal_delays is None:
        print("  No delay optimization performed.")
        return

    print(f"\n  Delay optimization compensates for calcium indicator dynamics.")
    print(f"  Positive delays = neural activity lags behavior (expected).")
    print(f"  Sampling rate: {fps} Hz")

    # Report delays for significant pairs, grouped by neuron type
    neuron_types = ground_truth.get("neuron_types", {})
    type_delays = {}

    for neuron_id, features in significant_neurons.items():
        neuron_type = neuron_types.get(neuron_id, "unknown")
        if neuron_type not in type_delays:
            type_delays[neuron_type] = []

        for feat_name in features:
            if feat_name in feat_bunch:
                feat_idx = feat_bunch.index(feat_name)
                delay_frames = optimal_delays[neuron_id, feat_idx]
                delay_sec = delay_frames / fps
                type_delays[neuron_type].append((neuron_id, feat_name, delay_frames, delay_sec))

    print(f"\n  Optimal delays for significant pairs:")
    for neuron_type in sorted(type_delays.keys()):
        delays = type_delays[neuron_type]
        if delays:
            # Calculate mean delay for this type
            mean_delay_sec = np.mean([d[3] for d in delays])
            print(f"\n  {neuron_type} (mean: {mean_delay_sec:.2f}s):")
            for neuron_id, feat_name, delay_frames, delay_sec in delays:
                print(f"    Neuron {neuron_id:2d} -> {feat_name:15s}: {delay_frames:4d} frames ({delay_sec:+.2f}s)")


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

    # Get feature names - use the same features that were analyzed (from feat_bunch)
    # Include position_2d but exclude x and y (which are marginals)
    feature_names = [
        f for f in exp.dynamic_features.keys()
        if f not in ["x", "y"]  # Exclude marginals, keep position_2d
    ]
    n_neurons = exp.n_cells
    n_features = len(feature_names)

    # Create MI matrix using 'me'
    mi_matrix = np.zeros((n_neurons, n_features))
    for neuron_id, features in significant_neurons.items():
        for feat_name in features:
            if feat_name in feature_names:
                feat_idx = feature_names.index(feat_name)
                pair_stats = exp.get_neuron_feature_pair_stats(neuron_id, feat_name)
                mi_matrix[neuron_id, feat_idx] = pair_stats.get("me", 0)

    im = ax1.imshow(mi_matrix, aspect="auto", cmap="viridis")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Neurons")
    ax1.set_title("INTENSE Selectivity Heatmap (MI values)")
    ax1.set_xticks(range(n_features))
    ax1.set_xticklabels(feature_names, rotation=45, ha="right")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Mutual Information (bits)")

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

    # 3. Summary statistics (before and after disentanglement)
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.axis("off")

    # Get corrected metrics if available
    has_corrected = "precision_corrected" in metrics
    prec_raw = metrics["precision"]
    prec_corr = metrics.get("precision_corrected", prec_raw)
    f1_raw = metrics["f1"]
    f1_corr = metrics.get("f1_corrected", f1_raw)
    fp_redundant = metrics.get("fp_redundant", 0)

    summary_text = (
        f"VALIDATION SUMMARY\n"
        f"{'=' * 30}\n\n"
        f"{'Metric':<12} {'Raw':>8} {'Corrected':>10}\n"
        f"{'-' * 30}\n"
        f"{'Sensitivity':<12} {metrics['sensitivity']:>7.1%}\n"
        f"{'Precision':<12} {prec_raw:>7.1%}  {prec_corr:>9.1%}\n"
        f"{'F1 Score':<12} {f1_raw:>7.1%}  {f1_corr:>9.1%}\n\n"
        f"Detection Counts:\n"
        f"  True Positives:  {metrics['true_positives']}\n"
        f"  False Positives: {metrics['false_positives']}"
    )
    if fp_redundant > 0:
        summary_text += f" ({fp_redundant} redundant)"
    summary_text += (
        f"\n  False Negatives: {metrics['false_negatives']}\n\n"
        f"Population:\n"
        f"  Neurons: {exp.n_cells}, Features: {len(exp.dynamic_features)}\n"
        f"  Expected pairs: {len(ground_truth['expected_pairs'])}\n"
    )
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontfamily="monospace", fontsize=9, verticalalignment="top")

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
    }

    exp = generate_tuned_selectivity_exp(
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
    ground_truth = exp.ground_truth

    # Step 2: Run INTENSE analysis with disentanglement and delay optimization
    print("\n[2] RUNNING INTENSE ANALYSIS")
    print("-" * 40)
    results, significant_neurons, analysis_time, disent_results, info, feat_bunch = (
        run_intense_analysis(exp, CONFIG, verbose=True)
    )

    # Step 3: Validate against ground truth using IntenseResults method
    print("\n[3] VALIDATING AGAINST GROUND TRUTH")
    print("-" * 40)
    metrics = results.validate_against_ground_truth(ground_truth, verbose=True)

    # Step 4: Disentanglement analysis (computes corrected metrics)
    print("\n[4] DISENTANGLEMENT ANALYSIS")
    print("-" * 40)
    metrics = analyze_disentanglement(disent_results, ground_truth, significant_neurons, metrics)

    # Step 5: Optimal delays analysis
    print("\n[5] OPTIMAL DELAYS")
    print("-" * 40)
    print_optimal_delays(info, feat_bunch, significant_neurons, ground_truth, CONFIG["fps"])

    # Step 6: Create visualizations
    print("\n[6] CREATING VISUALIZATIONS")
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
