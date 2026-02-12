#!/usr/bin/env python3
"""
INTENSE Validation Benchmark: SNR x skip_prob Sweep

Validates INTENSE selectivity detection across varying signal quality (SNR)
and response reliability (skip_prob). Two symmetric tests:
  1. Discrete: 5 event features
  2. Continuous: 5 FBM features

Each test uses 100 neurons (15 per feature + 25 nonselective).
Results are saved as per-run JSON files and aggregated into 2D heatmaps
of precision, recall, and F1.

Usage:
    python validate_intense_benchmark.py           # full sweep
    python validate_intense_benchmark.py --quick    # smoke test (1 config)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import driada
from driada.experiment.synthetic import generate_tuned_selectivity_exp


# =============================================================================
# CONFIGURATION
# =============================================================================

BASELINE_RATE = 0.03
SNR_VALUES = [4, 8, 16, 32, 64, 128]
SKIP_PROB_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_REALIZATIONS = 5

DURATION = 600
FPS = 20
CALCIUM_AMPLITUDE_RANGE = (0.5, 2.0)
CALCIUM_NOISE = 0.02
DECAY_TIME = 2.0

N_SHUFFLES_STAGE1 = 100
N_SHUFFLES_STAGE2 = 10000
PVAL_THR = 0.05
DS = 5
MULTICOMP_CORRECTION = "fdr_bh"

DISCRETE_POPULATION = [
    {"name": "event_0_cells", "count": 15, "features": ["event_0"]},
    {"name": "event_1_cells", "count": 15, "features": ["event_1"]},
    {"name": "event_2_cells", "count": 15, "features": ["event_2"]},
    {"name": "event_3_cells", "count": 15, "features": ["event_3"]},
    {"name": "event_4_cells", "count": 15, "features": ["event_4"]},
    {"name": "nonselective", "count": 25, "features": []},
]

CONTINUOUS_POPULATION = [
    {"name": "fbm_0_cells", "count": 15, "features": ["fbm_0"]},
    {"name": "fbm_1_cells", "count": 15, "features": ["fbm_1"]},
    {"name": "fbm_2_cells", "count": 15, "features": ["fbm_2"]},
    {"name": "fbm_3_cells", "count": 15, "features": ["fbm_3"]},
    {"name": "fbm_4_cells", "count": 15, "features": ["fbm_4"]},
    {"name": "nonselective", "count": 25, "features": []},
]


# =============================================================================
# UTILITIES
# =============================================================================

def snr_to_peak_rate(snr):
    """Convert SNR to peak firing rate. SNR = peak_rate / baseline_rate."""
    return BASELINE_RATE * snr


def compute_per_feature_metrics(metrics):
    """Compute precision, recall, F1 per feature from validation metrics."""
    tp_by_feat = defaultdict(int)
    fp_by_feat = defaultdict(int)
    fn_by_feat = defaultdict(int)

    for _, feat in metrics["tp_pairs"]:
        tp_by_feat[feat] += 1
    for _, feat in metrics["fp_pairs"]:
        fp_by_feat[feat] += 1
    for _, feat in metrics["fn_pairs"]:
        fn_by_feat[feat] += 1

    all_features = set(tp_by_feat) | set(fp_by_feat) | set(fn_by_feat)
    feature_metrics = {}

    for feat in sorted(all_features):
        tp = tp_by_feat[feat]
        fp = fp_by_feat[feat]
        fn = fn_by_feat[feat]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        feature_metrics[feat] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return feature_metrics


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def run_single_config(test_type, snr, skip_prob, realization_seed,
                      population, n_discrete):
    """Run one configuration: generate data, run INTENSE, validate."""
    peak_rate = snr_to_peak_rate(snr)

    # Generate experiment
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=DURATION,
        fps=FPS,
        baseline_rate=BASELINE_RATE,
        peak_rate=peak_rate,
        decay_time=DECAY_TIME,
        calcium_noise=CALCIUM_NOISE,
        calcium_amplitude_range=CALCIUM_AMPLITUDE_RANGE,
        n_discrete_features=n_discrete,
        skip_prob=skip_prob,
        seed=realization_seed,
        verbose=False,
    )

    # Build feature list (exclude x, y marginals)
    feat_bunch = [
        f for f in exp.dynamic_features.keys()
        if f not in ["x", "y"]
    ]

    # Run INTENSE
    stats, significance, info, results = driada.compute_cell_feat_significance(
        exp,
        feat_bunch=feat_bunch,
        mode="two_stage",
        n_shuffles_stage1=N_SHUFFLES_STAGE1,
        n_shuffles_stage2=N_SHUFFLES_STAGE2,
        find_optimal_delays=True,
        ds=DS,
        pval_thr=PVAL_THR,
        multicomp_correction=MULTICOMP_CORRECTION,
        with_disentanglement=False,
        verbose=False,
    )

    # Validate against ground truth
    metrics = results.validate_against_ground_truth(
        exp.ground_truth, verbose=False
    )

    # Per-feature metrics
    per_feature = compute_per_feature_metrics(metrics)

    return {
        "test_type": test_type,
        "snr": snr,
        "skip_prob": skip_prob,
        "seed": realization_seed,
        "peak_rate": round(peak_rate, 4),
        "overall": {
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["sensitivity"], 4),
            "f1": round(metrics["f1"], 4),
        },
        "per_feature": per_feature,
    }


def run_sweep(test_type, population, n_discrete, output_dir,
              snr_values=None, skip_prob_values=None, n_realizations=None):
    """Run full SNR x skip_prob sweep."""
    snr_values = snr_values or SNR_VALUES
    skip_prob_values = skip_prob_values or SKIP_PROB_VALUES
    n_realizations = n_realizations or N_REALIZATIONS

    results_dir = os.path.join(output_dir, f"results_{test_type}")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    total = len(snr_values) * len(skip_prob_values) * n_realizations
    done = 0

    for snr in snr_values:
        for skip_prob in skip_prob_values:
            for seed_idx in range(n_realizations):
                realization_seed = seed_idx
                done += 1

                label = (f"[{done}/{total}] {test_type} "
                         f"SNR={snr} skip={skip_prob} seed={realization_seed}")
                print(f"  {label} ...", end=" ", flush=True)

                t0 = time.time()
                result = run_single_config(
                    test_type, snr, skip_prob, realization_seed,
                    population, n_discrete,
                )
                elapsed = time.time() - t0

                # Save per-run JSON
                fname = f"snr{snr}_skip{skip_prob}_seed{realization_seed}.json"
                fpath = os.path.join(results_dir, fname)
                with open(fpath, "w") as f:
                    json.dump(result, f, indent=2)

                p = result["overall"]["precision"]
                r = result["overall"]["recall"]
                f1 = result["overall"]["f1"]
                print(f"P={p:.2f} R={r:.2f} F1={f1:.2f} ({elapsed:.1f}s)")

                all_results.append(result)

    return all_results


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results(all_results, snr_values=None, skip_prob_values=None):
    """Aggregate metrics across realizations for each (SNR, skip_prob) cell."""
    snr_values = snr_values or SNR_VALUES
    skip_prob_values = skip_prob_values or SKIP_PROB_VALUES

    n_snr = len(snr_values)
    n_skip = len(skip_prob_values)

    # Initialize arrays
    precision_mean = np.zeros((n_snr, n_skip))
    recall_mean = np.zeros((n_snr, n_skip))
    f1_mean = np.zeros((n_snr, n_skip))
    precision_std = np.zeros((n_snr, n_skip))
    recall_std = np.zeros((n_snr, n_skip))
    f1_std = np.zeros((n_snr, n_skip))

    for i, snr in enumerate(snr_values):
        for j, skip_prob in enumerate(skip_prob_values):
            cell_results = [
                r for r in all_results
                if r["snr"] == snr and r["skip_prob"] == skip_prob
            ]
            if not cell_results:
                continue

            precisions = [r["overall"]["precision"] for r in cell_results]
            recalls = [r["overall"]["recall"] for r in cell_results]
            f1s = [r["overall"]["f1"] for r in cell_results]

            precision_mean[i, j] = np.mean(precisions)
            recall_mean[i, j] = np.mean(recalls)
            f1_mean[i, j] = np.mean(f1s)
            precision_std[i, j] = np.std(precisions)
            recall_std[i, j] = np.std(recalls)
            f1_std[i, j] = np.std(f1s)

    return {
        "precision_mean": precision_mean, "precision_std": precision_std,
        "recall_mean": recall_mean, "recall_std": recall_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "snr_values": snr_values,
        "skip_prob_values": skip_prob_values,
    }


def save_summary_csv(aggregated, test_type, output_dir):
    """Save aggregated results as CSV."""
    snr_values = aggregated["snr_values"]
    skip_prob_values = aggregated["skip_prob_values"]

    fpath = os.path.join(output_dir, f"summary_{test_type}.csv")
    with open(fpath, "w") as f:
        f.write("snr,skip_prob,precision_mean,precision_std,"
                "recall_mean,recall_std,f1_mean,f1_std\n")
        for i, snr in enumerate(snr_values):
            for j, skip_prob in enumerate(skip_prob_values):
                f.write(
                    f"{snr},{skip_prob},"
                    f"{aggregated['precision_mean'][i,j]:.4f},"
                    f"{aggregated['precision_std'][i,j]:.4f},"
                    f"{aggregated['recall_mean'][i,j]:.4f},"
                    f"{aggregated['recall_std'][i,j]:.4f},"
                    f"{aggregated['f1_mean'][i,j]:.4f},"
                    f"{aggregated['f1_std'][i,j]:.4f}\n"
                )
    print(f"  Saved: {fpath}")


def save_summary_npz(aggregated, test_type, output_dir):
    """Save aggregated results as NPZ with 3D arrays (SNR x skip_prob x metric)."""
    snr_values = np.array(aggregated["snr_values"])
    skip_prob_values = np.array(aggregated["skip_prob_values"])

    # Stack into (n_snr, n_skip, 3) arrays — metrics order: precision, recall, f1
    mean = np.stack([
        aggregated["precision_mean"],
        aggregated["recall_mean"],
        aggregated["f1_mean"],
    ], axis=-1)

    std = np.stack([
        aggregated["precision_std"],
        aggregated["recall_std"],
        aggregated["f1_std"],
    ], axis=-1)

    fpath = os.path.join(output_dir, f"summary_{test_type}.npz")
    np.savez(
        fpath,
        mean=mean,
        std=std,
        snr_values=snr_values,
        skip_prob_values=skip_prob_values,
        metric_names=np.array(["precision", "recall", "f1"]),
    )
    print(f"  Saved: {fpath}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_heatmap(ax, data, snr_values, skip_prob_values, title, vmin=0, vmax=1):
    """Plot a single 2D heatmap on the given axis."""
    im = ax.imshow(
        data, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax,
        origin="lower",
    )
    ax.set_xticks(range(len(skip_prob_values)))
    ax.set_xticklabels([f"{sp:.1f}" for sp in skip_prob_values])
    ax.set_yticks(range(len(snr_values)))
    ax.set_yticklabels([str(s) for s in snr_values])
    ax.set_xlabel("skip_prob")
    ax.set_ylabel("SNR")
    ax.set_title(title)

    # Annotate cells with values
    for i in range(len(snr_values)):
        for j in range(len(skip_prob_values)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    return im


def plot_heatmaps(aggregated, test_name, output_dir):
    """Create 3-panel heatmap (precision, recall, F1) for one test."""
    snr_values = aggregated["snr_values"]
    skip_prob_values = aggregated["skip_prob_values"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title in zip(
        axes,
        ["precision_mean", "recall_mean", "f1_mean"],
        ["Precision", "Recall", "F1"],
    ):
        im = plot_single_heatmap(
            ax, aggregated[metric], snr_values, skip_prob_values,
            f"{test_name} — {title}",
        )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Score")
    fig.suptitle(f"INTENSE Validation: {test_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fpath = os.path.join(output_dir, f"heatmap_{test_name.lower()}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_combined_heatmaps(agg_discrete, agg_continuous, output_dir):
    """Create 2x3 combined heatmap for both tests."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, (agg, label) in enumerate([
        (agg_discrete, "Discrete"),
        (agg_continuous, "Continuous"),
    ]):
        for col, (metric, title) in enumerate([
            ("precision_mean", "Precision"),
            ("recall_mean", "Recall"),
            ("f1_mean", "F1"),
        ]):
            im = plot_single_heatmap(
                axes[row, col],
                agg[metric],
                agg["snr_values"],
                agg["skip_prob_values"],
                f"{label} — {title}",
            )

    fig.colorbar(im, ax=axes, shrink=0.6, label="Score")
    fig.suptitle(
        "INTENSE Validation Benchmark: SNR × skip_prob",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    fpath = os.path.join(output_dir, "heatmap_combined.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="INTENSE validation benchmark")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 1 SNR, 1 skip_prob, 1 realization per test",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))

    if args.quick:
        snr_values = [16]
        skip_prob_values = [0.0]
        n_realizations = 1
        print("=== QUICK MODE: 1 config per test ===\n")
    else:
        snr_values = SNR_VALUES
        skip_prob_values = SKIP_PROB_VALUES
        n_realizations = N_REALIZATIONS
        total_runs = len(snr_values) * len(skip_prob_values) * n_realizations * 2
        print(f"=== FULL SWEEP: {total_runs} total runs ===\n")

    # --- Discrete test ---
    print("=" * 60)
    print("TEST 1: DISCRETE (5 event features)")
    print("=" * 60)
    t0 = time.time()
    results_discrete = run_sweep(
        "discrete", DISCRETE_POPULATION, n_discrete=5, output_dir=output_dir,
        snr_values=snr_values, skip_prob_values=skip_prob_values,
        n_realizations=n_realizations,
    )
    t_discrete = time.time() - t0
    print(f"\nDiscrete test completed in {t_discrete:.1f}s\n")

    # --- Continuous test ---
    print("=" * 60)
    print("TEST 2: CONTINUOUS (5 FBM features)")
    print("=" * 60)
    t0 = time.time()
    results_continuous = run_sweep(
        "continuous", CONTINUOUS_POPULATION, n_discrete=0, output_dir=output_dir,
        snr_values=snr_values, skip_prob_values=skip_prob_values,
        n_realizations=n_realizations,
    )
    t_continuous = time.time() - t0
    print(f"\nContinuous test completed in {t_continuous:.1f}s\n")

    # --- Aggregate ---
    print("=" * 60)
    print("AGGREGATION & VISUALIZATION")
    print("=" * 60)

    agg_discrete = aggregate_results(
        results_discrete, snr_values, skip_prob_values
    )
    agg_continuous = aggregate_results(
        results_continuous, snr_values, skip_prob_values
    )

    save_summary_csv(agg_discrete, "discrete", output_dir)
    save_summary_csv(agg_continuous, "continuous", output_dir)
    save_summary_npz(agg_discrete, "discrete", output_dir)
    save_summary_npz(agg_continuous, "continuous", output_dir)

    plot_heatmaps(agg_discrete, "Discrete", output_dir)
    plot_heatmaps(agg_continuous, "Continuous", output_dir)

    if len(snr_values) > 1 and len(skip_prob_values) > 1:
        plot_combined_heatmaps(agg_discrete, agg_continuous, output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total time: {t_discrete + t_continuous:.1f}s")
    print(f"Output directory: {output_dir}")

    if not args.quick:
        # Print best/worst cells
        for test_name, agg in [("Discrete", agg_discrete), ("Continuous", agg_continuous)]:
            f1 = agg["f1_mean"]
            best = np.unravel_index(np.argmax(f1), f1.shape)
            worst = np.unravel_index(np.argmin(f1), f1.shape)
            print(f"\n{test_name}:")
            print(f"  Best  F1={f1[best]:.3f} at SNR={snr_values[best[0]]}, "
                  f"skip={skip_prob_values[best[1]]}")
            print(f"  Worst F1={f1[worst]:.3f} at SNR={snr_values[worst[0]]}, "
                  f"skip={skip_prob_values[worst[1]]}")


if __name__ == "__main__":
    main()
