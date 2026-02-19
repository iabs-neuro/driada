#!/usr/bin/env python
"""
Functional organization of neural manifolds
============================================

This example demonstrates how to analyze which neurons drive which
dimensions of a population manifold. It uses DRIADA's integration
module to bridge dimensionality reduction with INTENSE selectivity
analysis -- the reverse direction of the INTENSE->DR pipeline.

Key concepts:
1. Treat embedding components as features and run INTENSE on them
2. Discover functional clusters (neurons with similar manifold roles)
3. Compare functional organization across DR methods (PCA vs UMAP)

APIs demonstrated:
- compute_embedding_selectivity  -- INTENSE on embedding components
- get_functional_organization    -- cluster and participation analysis
- compare_embeddings             -- cross-method comparison
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.intense import compute_embedding_selectivity
from driada.integration import get_functional_organization, compare_embeddings
from driada.utils.visual import DEFAULT_DPI, plot_embedding_comparison


def main(quick_test=False, enable_visualizations=True, seed=42):
    """Run the functional organization analysis.

    Parameters
    ----------
    quick_test : bool
        If True, use smaller parameters for faster execution.
    enable_visualizations : bool
        If True, create and save visualization plots.
    seed : int
        Random seed for reproducibility.
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("Functional organization of neural manifolds")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Generate synthetic population
    # ------------------------------------------------------------------
    print("\n1. Generating synthetic population...")

    if quick_test:
        n_scale = 0.5
        duration = 300
        n_shuffles_1 = 50
        n_shuffles_2 = 1000
        n_pca_components = 4
    else:
        n_scale = 1.0
        duration = 600
        n_shuffles_1 = 100
        n_shuffles_2 = 5000
        n_pca_components = 5

    population = [
        {"name": "hd_cells", "count": int(25 * n_scale),
         "features": ["head_direction"]},
        {"name": "place_cells", "count": int(20 * n_scale),
         "features": ["position_2d"]},
        {"name": "speed_cells", "count": int(15 * n_scale),
         "features": ["speed"]},
        {"name": "conjunctive", "count": int(10 * n_scale),
         "features": ["head_direction", "speed"], "combination": "and"},
        {"name": "non_selective", "count": int(30 * n_scale),
         "features": []},
    ]

    exp = generate_tuned_selectivity_exp(
        population, duration=duration, fps=20, seed=seed, verbose=True
    )

    # Build ground truth group map: neuron_index -> group_name
    gt_groups = {}
    idx = 0
    for group in population:
        for _ in range(group["count"]):
            gt_groups[idx] = group["name"]
            idx += 1

    n_total = exp.n_cells
    print(f"  {n_total} neurons, {exp.n_frames} frames")
    for group in population:
        print(f"    {group['name']:20s}: {group['count']} neurons")

    # ------------------------------------------------------------------
    # 2. Create embeddings
    # ------------------------------------------------------------------
    print("\n2. Creating embeddings...")

    pca_emb = exp.create_embedding("pca", n_components=n_pca_components, ds=5)
    print(f"  PCA: {pca_emb.shape} (n_frames, n_components)")

    umap_emb = exp.create_embedding(
        "umap", n_components=3, n_neighbors=50, min_dist=0.8, random_state=seed, ds=5
    )
    print(f"  UMAP: {umap_emb.shape}")

    # ------------------------------------------------------------------
    # 3. Compute embedding selectivity
    # ------------------------------------------------------------------
    #
    # This is the core step: each embedding component is temporarily
    # added as a dynamic feature, and INTENSE tests whether each neuron's
    # activity is significantly related to that component.
    #
    print("\n3. Computing embedding selectivity (INTENSE on components)...")

    results = compute_embedding_selectivity(
        exp,
        embedding_methods=["pca", "umap"],
        mode="two_stage",
        n_shuffles_stage1=n_shuffles_1,
        n_shuffles_stage2=n_shuffles_2,
        find_optimal_delays=False,
        pval_thr=0.01,
        ds=5,
        verbose=True,
        seed=seed,
    )

    for method in ["pca", "umap"]:
        r = results[method]
        n_sig = len(r["significant_neurons"])
        print(f"\n  {method.upper()} summary:")
        print(f"    {n_sig}/{n_total} neurons significantly selective "
              f"({100 * n_sig / n_total:.0f}%)")
        for comp_idx in range(r["n_components"]):
            n_sel = len(r["component_selectivity"][comp_idx])
            if n_sel > 0:
                print(f"    component {comp_idx}: {n_sel} selective neurons")

    # ------------------------------------------------------------------
    # 4. Analyze functional organization (PCA)
    # ------------------------------------------------------------------
    print("\n4. Functional organization (PCA)...")

    org = get_functional_organization(
        exp, "pca", intense_results=results["pca"]["intense_results"]
    )

    # Component importance
    print("\n  Component importance (variance explained):")
    for i, imp in enumerate(org["component_importance"]):
        print(f"    component {i}: {imp:.3f}")

    # Component specialization
    print(f"\n  Participating neurons: {org['n_participating_neurons']}/{n_total}")
    print(f"  Mean components per neuron: {org['mean_components_per_neuron']:.2f}")

    print("\n  Component specialization:")
    for comp_idx, spec in org["component_specialization"].items():
        n_sel = spec["n_selective_neurons"]
        rate = spec["selectivity_rate"]
        if n_sel > 0:
            # Cross-reference with ground truth groups
            group_counts = {}
            for nid in spec["selective_neurons"]:
                g = gt_groups.get(nid, "unknown")
                group_counts[g] = group_counts.get(g, 0) + 1
            groups_str = ", ".join(f"{g}={c}" for g, c in sorted(group_counts.items()))
            print(f"    component {comp_idx}: {n_sel} neurons ({rate:.0%}) -- {groups_str}")

    # Functional clusters
    print(f"\n  Functional clusters: {len(org['functional_clusters'])}")
    for i, cluster in enumerate(org["functional_clusters"]):
        comps = cluster["components"]
        size = cluster["size"]
        # What ground truth groups are in this cluster?
        group_counts = {}
        for nid in cluster["neurons"]:
            g = gt_groups.get(nid, "unknown")
            group_counts[g] = group_counts.get(g, 0) + 1
        groups_str = ", ".join(f"{g}={c}" for g, c in sorted(group_counts.items()))
        print(f"    cluster {i}: components {comps}, {size} neurons -- {groups_str}")

    # ------------------------------------------------------------------
    # 5. Compare PCA vs UMAP organization
    # ------------------------------------------------------------------
    print("\n5. Comparing PCA vs UMAP functional organization...")

    intense_dict = {
        m: results[m]["intense_results"] for m in ["pca", "umap"]
    }
    comparison = compare_embeddings(
        exp, ["pca", "umap"], intense_results_dict=intense_dict
    )

    for method in comparison["methods"]:
        n_part = comparison["n_participating_neurons"][method]
        mean_comp = comparison["mean_components_per_neuron"][method]
        n_clust = comparison["n_functional_clusters"][method]
        print(f"  {method.upper():6s}: {n_part} participating neurons, "
              f"{mean_comp:.2f} mean components, {n_clust} clusters")

    if "participation_overlap" in comparison:
        for pair, overlap in comparison["participation_overlap"].items():
            print(f"  Participation overlap ({pair}): {overlap:.2f}")

    # ------------------------------------------------------------------
    # 6. Visualize
    # ------------------------------------------------------------------
    if enable_visualizations:
        print("\n6. Creating visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Functional organization analysis", fontsize=14)

        # (a) Component importance
        ax = axes[0, 0]
        comp_imp = org["component_importance"]
        ax.bar(range(len(comp_imp)), comp_imp, color="steelblue", edgecolor="white")
        ax.set_xlabel("PCA component")
        ax.set_ylabel("Variance explained (fraction)")
        ax.set_title("Component importance")
        ax.set_xticks(range(len(comp_imp)))

        # (b) Component specialization by neuron group
        ax = axes[0, 1]
        group_names = [g["name"] for g in population]
        group_colors = plt.cm.Set2(np.linspace(0, 1, len(group_names)))
        color_map = dict(zip(group_names, group_colors))

        comp_indices = sorted(org["component_specialization"].keys())
        bottom = np.zeros(len(comp_indices))
        for gname in group_names:
            counts = []
            for comp_idx in comp_indices:
                spec = org["component_specialization"][comp_idx]
                c = sum(1 for nid in spec["selective_neurons"]
                        if gt_groups.get(nid) == gname)
                counts.append(c)
            ax.bar(comp_indices, counts, bottom=bottom,
                   label=gname, color=color_map[gname], edgecolor="white")
            bottom += counts
        ax.set_xlabel("PCA component")
        ax.set_ylabel("Selective neurons")
        ax.set_title("Component specialization by group")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks(comp_indices)

        # (c) Neuron participation histogram
        ax = axes[1, 0]
        participation = org.get("neuron_participation", {})
        if participation:
            n_comps_per_neuron = [len(comps) for comps in participation.values()]
            max_comps = max(n_comps_per_neuron) if n_comps_per_neuron else 1
            bins = np.arange(0.5, max_comps + 1.5, 1)
            ax.hist(n_comps_per_neuron, bins=bins, color="steelblue",
                    edgecolor="white", rwidth=0.8)
            ax.set_xlabel("Number of components")
            ax.set_ylabel("Number of neurons")
            ax.set_title("Neuron participation distribution")
            ax.set_xticks(range(1, max_comps + 1))
        else:
            ax.text(0.5, 0.5, "No participating neurons",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Neuron participation distribution")

        # (d) PCA vs UMAP: participating neurons per method
        ax = axes[1, 1]
        methods = comparison["methods"]
        n_parts = [comparison["n_participating_neurons"][m] for m in methods]
        n_clusts = [comparison["n_functional_clusters"][m] for m in methods]
        x = np.arange(len(methods))
        w = 0.35
        ax.bar(x - w / 2, n_parts, w, label="Participating neurons",
               color="steelblue", edgecolor="white")
        ax.bar(x + w / 2, n_clusts, w, label="Functional clusters",
               color="coral", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylabel("Count")
        ax.set_title("PCA vs UMAP comparison")
        ax.legend(fontsize=8)

        if "participation_overlap" in comparison:
            for pair, overlap in comparison["participation_overlap"].items():
                ax.annotate(f"Overlap: {overlap:.2f}",
                            xy=(0.5, 0.95), xycoords="axes fraction",
                            ha="center", fontsize=9, color="gray")

        plt.tight_layout()
        save_path = os.path.join(output_dir, "functional_organization.png")
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()

        # Embedding scatter plots colored by behavioral features
        ds_emb = 5  # must match ds used in create_embedding
        hd = exp.dynamic_features["head_direction"].data[::ds_emb]
        spd = exp.dynamic_features["speed"].data[::ds_emb]

        plot_embedding_comparison(
            embeddings={"PCA": pca_emb[:, :2], "UMAP": umap_emb[:, :2]},
            features={"head_direction": hd, "speed": spd},
            with_trajectory=False,
            compute_metrics=False,
            scatter_size=8,
            save_path=os.path.join(output_dir, "embedding_comparison.png"),
        )
        print(f"  Saved: {os.path.join(output_dir, 'embedding_comparison.png')}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[OK] Functional organization analysis complete")
    print("=" * 65)

    return exp, results, org, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Functional organization of neural manifolds"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test with reduced parameters"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Disable visualization generation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    main(
        quick_test=args.quick,
        enable_visualizations=not args.no_viz,
        seed=args.seed,
    )
