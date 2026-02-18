"""
Behavior relations example -- feature-feature significance
===========================================================

Tests which behavioral variables are significantly correlated,
independent of neural data. Uses FFT-based circular shuffle to
account for temporal autocorrelation.

Sections:
1. Generate synthetic experiment
2. Add a derived feature (smoothed speed) to create a known correlation
3. compute_feat_feat_significance -- significance/similarity matrices
4. Display and visualize results
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.information import TimeSeries
from driada.intense import compute_feat_feat_significance


POPULATION = [
    {"name": "hd_cells", "count": 4, "features": ["head_direction"]},
    {"name": "place_cells", "count": 4, "features": ["position_2d"]},
    {"name": "speed_cells", "count": 4, "features": ["speed"]},
    {"name": "event_cells", "count": 4, "features": ["event_0"]},
    {"name": "nonselective", "count": 4, "features": []},
]


def add_derived_features(exp, fps):
    """Add a smoothed speed feature with known correlation to raw speed."""
    speed_data = exp.dynamic_features["speed"].data

    # 1-second moving average of speed.
    # Preserves enough variance for significant MI with the raw signal.
    kernel_size = int(1 * fps)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(speed_data, kernel, mode="same")
    exp.dynamic_features["speed_smoothed"] = TimeSeries(
        smoothed, ts_type="linear", name="speed_smoothed"
    )


def display_results(sim_mat, sig_mat, pval_mat, feature_names):
    """Print significant feature pairs and their MI values."""
    display_names = []
    for name in feature_names:
        if isinstance(name, (list, tuple)):
            display_names.append(", ".join(str(n) for n in name))
        else:
            display_names.append(str(name))

    n = len(feature_names)
    print(f"\n  Features analyzed: {n}")
    print(f"  Feature names: {display_names}")

    print(f"\n  Significant pairs:")
    n_sig = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sig_mat[i, j]:
                n_sig += 1
                print(
                    f"    {display_names[i]:20s} <-> {display_names[j]:20s}  "
                    f"MI={sim_mat[i, j]:.4f}  p={pval_mat[i, j]:.2e}"
                )
    if n_sig == 0:
        print("    (none)")
    print(f"\n  Total significant pairs: {n_sig}/{n * (n - 1) // 2}")

    return display_names


def create_heatmap(sim_mat, sig_mat, display_names, output_dir):
    """Create similarity matrix heatmap."""
    n = len(display_names)
    fig, ax = plt.subplots(figsize=(8, 7))

    plot_mat = sim_mat.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    im = ax.imshow(plot_mat, cmap="Blues", aspect="equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mutual information (bits)")

    for i in range(n):
        for j in range(n):
            if i != j and sig_mat[i, j]:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="red")

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=True, facecolor="0.85", edgecolor="none"))

    ax.set_xticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_title("Feature-feature mutual information (* = significant)")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "feature_feature_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("DRIADA behavior relations example")
    print("=" * 60)

    output_dir = os.path.dirname(__file__)
    fps = 20

    # Step 1: Generate experiment
    print("\n[1] Generating synthetic experiment")
    print("-" * 40)
    exp = generate_tuned_selectivity_exp(
        population=POPULATION,
        duration=600,
        fps=fps,
        n_discrete_features=2,
        seed=42,
        verbose=True,
    )
    print(f"  Features: {list(exp.dynamic_features.keys())}")

    # Step 2: Add derived feature with known correlation to speed
    print("\n[2] Adding derived feature")
    print("-" * 40)
    add_derived_features(exp, fps)
    print(f"  Added speed_smoothed (1-second moving average of speed)")

    # Step 3: Compute feat-feat significance.
    # Use feat_bunch to select features explicitly.
    # Exclude raw head_direction -- the pipeline uses head_direction_2d
    # (cos/sin encoding) which preserves circular topology.
    print("\n[3] Computing feature-feature significance")
    print("-" * 40)
    features_to_test = [
        "head_direction_2d", "speed", "position_2d",
        "event_0", "event_1", "speed_smoothed",
    ]
    print(f"  Testing: {features_to_test}")
    print(f"  (head_direction excluded -- use head_direction_2d for circular data)")

    sim_mat, sig_mat, pval_mat, feature_names, info = compute_feat_feat_significance(
        exp,
        feat_bunch=features_to_test,
        n_shuffles_stage1=100,
        n_shuffles_stage2=1000,
        pval_thr=0.01,
        verbose=True,
    )

    # Step 4: Display results
    print("\n[4] Results summary")
    print("-" * 40)
    display_names = display_results(sim_mat, sig_mat, pval_mat, feature_names)

    # Step 5: Visualization
    print("\n[5] Visualization")
    print("-" * 40)
    create_heatmap(sim_mat, sig_mat, display_names, output_dir)

    print("\n" + "=" * 60)
    print("Behavior relations example complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
