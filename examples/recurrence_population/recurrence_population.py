#!/usr/bin/env python3
"""
Recovering functional modules from recurrence dynamics

Scientific question: Can we identify which neurons belong to the same
functional module purely from their temporal dynamics, without ever
computing mutual information or looking at behavioral variables?

This example mirrors the cell-cell network analysis example, which uses
MI-based significance testing to build functional networks.  Here we use
a completely different instrument -- recurrence analysis -- to recover
the same modular structure from the same synthetic population.

Approach:
  1. Generate 120 neurons in 6 functional modules (same as cell-cell example):
     3 single-event modules (30 neurons each) + 3 dual-event OR modules (10 each)
  2. Build recurrence graphs for each neuron from calcium traces alone
  3. Quantify dynamical signatures per neuron (RQA measures)
  4. Measure pairwise similarity of recurrence patterns (Jaccard index)
  5. Detect communities in the similarity network (Network of Networks)
  6. Compare detected communities to ground-truth functional modules

Key insight: neurons selective to the same event(s) are driven by the
same behavioral variable.  When an event is active at two time points,
selective neurons recur together -- creating shared recurrence structure.
This makes same-module neurons more similar in Jaccard space, enabling
recovery of functional groups from dynamics alone.

Figures produced:
  - Recurrence plots for representative neurons from each module
  - Three graph views of a single neuron (RG, HVG, OPN)
  - Jaccard similarity network with convex hulls (colored by module)
  - Jaccard matrices ordered by module vs detected community
  - Confusion matrix: ground-truth modules vs detected communities
"""

import os
import sys
import pickle
import hashlib
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import networkx.algorithms.community as nx_comm
from sklearn.metrics import adjusted_rand_score
import networkx as nx

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.recurrence import (
    RecurrenceGraph,
    plot_recurrence,
    pairwise_jaccard_sparse,
)
from driada.network import Network


# Ground-truth module colors (consistent with eLife fig 2)
MODULE_COLORS = {
    "event_0":  "#1a5acd",   # dark blue
    "event_1":  "#ffaa00",   # orange
    "event_2":  "#33cc33",   # green
    "event_0|1": "#cc44cc",  # magenta (bridge E0-E1)
    "event_0|2": "#00dddd",  # cyan    (bridge E0-E2)
    "event_1|2": "#ff4444",  # red     (bridge E1-E2)
}
MODULE_SHORT = {
    "event_0": "E1", "event_1": "E2", "event_2": "E3",
    "event_0|1": "E1|E2", "event_0|2": "E1|E3",
    "event_1|2": "E2|E3",
}


# =============================================================================
# CONFIGURATION
# =============================================================================
# Six functional modules mirroring the cell-cell network example:
# - 3 single-event modules (30 neurons each): event_0, event_1, event_2
# - 3 dual-event OR modules (10 neurons each): event_0|1, event_0|2, event_1|2
POPULATION = [
    {"name": "event_0", "count": 30, "features": ["event_0"]},
    {"name": "event_1", "count": 30, "features": ["event_1"]},
    {"name": "event_2", "count": 30, "features": ["event_2"]},
    {"name": "event_0|1", "count": 10, "features": ["event_0", "event_1"],
     "combination": "or"},
    {"name": "event_0|2", "count": 10, "features": ["event_0", "event_2"],
     "combination": "or"},
    {"name": "event_1|2", "count": 10, "features": ["event_1", "event_2"],
     "combination": "or"},
]

CONFIG = {
    "duration": 600,        # 10-minute recording (seconds)
    "fps": 5,
    "seed": 42,             # same seed as cell-cell example
    "n_discrete_features": 3,
    # Recurrence graph
    "k": 50,                # k-NN neighbors; higher k = denser RP
    "max_shift": 60,        # max lag for TDMI tau estimation (samples)
    "max_dim": 15,          # max FNN dimension to test
    # Network of Networks: keep top 10% of Jaccard pairs as edges
    "jaccard_percentile": 90,
}


# =============================================================================
# HELPERS
# =============================================================================
def get_neuron_modules(population):
    """Map neuron index -> module name from population config."""
    modules = {}
    idx = 0
    for group in population:
        for _ in range(group["count"]):
            modules[idx] = group["name"]
            idx += 1
    return modules


def module_indices(neuron_modules):
    """Group neuron indices by module."""
    groups = {}
    for idx, m in neuron_modules.items():
        groups.setdefault(m, []).append(idx)
    return groups


def _config_hash():
    """Deterministic hash of CONFIG + POPULATION for cache invalidation."""
    blob = json.dumps({"config": CONFIG, "population": POPULATION},
                      sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _cache_path():
    return os.path.join(os.path.dirname(__file__), "_cache.pkl")


def _save_cache(data):
    data["_config_hash"] = _config_hash()
    with open(_cache_path(), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [CACHE] Saved to {_cache_path()}")


def _load_cache():
    path = _cache_path()
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data.get("_config_hash") != _config_hash():
        print("  [CACHE] Config changed, recomputing...")
        return None
    # Invalidate old caches missing new fields
    required_keys = {"perm_p_value", "perm_ratios"}
    if not required_keys.issubset(data.keys()):
        print("  [CACHE] Missing new fields, recomputing...")
        return None
    print(f"  [CACHE] Loaded from {path}")
    return data


def _set_rp_time_axes(ax, n_points, fps_eff):
    """Convert RP axes from sample index to seconds."""
    # Choose nice tick spacing
    total_sec = n_points / fps_eff
    if total_sec > 200:
        tick_step = 60
    elif total_sec > 50:
        tick_step = 20
    else:
        tick_step = 10
    tick_sec = np.arange(0, total_sec + 1, tick_step)
    tick_idx = tick_sec * fps_eff
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{s:.0f}" for s in tick_sec], fontsize=7)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{s:.0f}" for s in tick_sec], fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Time (s)", fontsize=9)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Recovering functional modules from recurrence dynamics")
    print("=" * 70)

    output_dir = os.path.dirname(__file__)
    neuron_modules = get_neuron_modules(POPULATION)
    groups = module_indices(neuron_modules)
    module_names = sorted(groups.keys())
    n_total = sum(g["count"] for g in POPULATION)

    # -----------------------------------------------------------------
    # Try loading cached computation
    # -----------------------------------------------------------------
    cache = _load_cache()

    if cache is not None:
        # Unpack cached data
        calcium = cache["calcium"]
        n_neurons, n_frames = calcium.shape
        fps_eff = CONFIG["fps"]
        taus = cache["taus"]
        dims = cache["dims"]
        rqa_results = cache["rqa_results"]
        sim_matrix = cache["sim_matrix"]
        demo_rg_adj = cache["demo_rg_adj"]
        demo_vg_adj = cache["demo_vg_adj"]
        demo_opn_adj = cache["demo_opn_adj"]
        demo_opn_pattern_ids = cache["demo_opn_pattern_ids"]
        demo_idx = cache["demo_idx"]
        demo_tau = cache["demo_tau"]
        demo_dim = cache["demo_dim"]
        example_rp_adjs = cache["example_rp_adjs"]
        example_idxs = cache["example_idxs"]
        communities = cache["communities"]
        nodes_in_net = cache["nodes_in_net"]
        ari = cache["ari"]
        mod_real = cache["mod_real"]
        mod_rand = cache["mod_rand"]
        n_detected = cache["n_detected"]
        net_adj = cache["net_adj"]
        thr = cache["thr"]
        within_vals = cache["within_vals"]
        between_vals = cache["between_vals"]
        ratio = cache["ratio"]
        perm_p_value = cache["perm_p_value"]
        perm_ratios = cache["perm_ratios"]

        # Rebuild lightweight objects from cached adjacency
        net = Network(adj=net_adj, preprocessing=None, create_nx_graph=True)

        print(f"\n  {n_neurons} neurons, {n_frames} frames ({fps_eff:.0f} Hz)")
        print(f"  ARI={ari:.3f}, modularity={mod_real:.3f}, "
              f"Jaccard ratio={ratio:.2f}x, perm p={perm_p_value:.4f}")

    else:
        # =============================================================
        # FULL COMPUTATION (cached for next run)
        # =============================================================

        # -----------------------------------------------------------------
        # Step 1: Generate synthetic population with known modules
        # -----------------------------------------------------------------
        print("\n[1] Generating synthetic population...")

        exp = generate_tuned_selectivity_exp(
            population=POPULATION,
            duration=CONFIG["duration"],
            fps=CONFIG["fps"],
            seed=CONFIG["seed"],
            n_discrete_features=CONFIG["n_discrete_features"],
            baseline_rate=0.05,
            peak_rate=2.0,
            decay_time=2.0,
            calcium_noise=0.02,
            verbose=False,
        )

        calcium = exp.calcium.data
        n_neurons, n_frames = calcium.shape
        fps_eff = CONFIG["fps"]

        print(f"  {n_neurons} neurons, {n_frames} frames ({fps_eff:.0f} Hz)")
        for g in POPULATION:
            feat_str = " OR ".join(g["features"]) if g["features"] else "none"
            print(f"    {g['count']:>2} {g['name']:<12} -> {feat_str}")

        ts_list = exp.calcium.ts_list

        # -----------------------------------------------------------------
        # Step 2: Per-neuron embedding and RQA
        # -----------------------------------------------------------------
        # For each neuron:
        #   tau  = time delay (first minimum of time-delayed mutual information)
        #   dim  = embedding dimension (false nearest neighbors criterion)
        #   RQA  = recurrence quantification (DET, LAM, ENTR from the RP)
        print("\n[2] Per-neuron embedding and RQA...")

        taus = np.empty(n_neurons, dtype=int)
        dims = np.empty(n_neurons, dtype=int)
        rqa_results = []

        for i, ts in enumerate(ts_list):
            taus[i] = ts.estimate_tau(max_shift=CONFIG["max_shift"])
            dims[i] = ts.estimate_embedding_dim(tau=taus[i], max_dim=CONFIG["max_dim"])
            rqa_results.append(ts.rqa(tau=taus[i], m=dims[i], k=CONFIG["k"]))

        print(f"\n  {'Module':<15} {'n':>3} {'tau':>6} {'dim':>5}"
              f" {'DET':>7} {'LAM':>7} {'ENTR':>7}")
        print(f"  {'-'*55}")

        for m in module_names:
            idxs = groups[m]
            det = np.array([rqa_results[i]["DET"] for i in idxs])
            lam = np.array([rqa_results[i]["LAM"] for i in idxs])
            entr = np.array([rqa_results[i]["ENTR"] for i in idxs])
            print(f"  {m:<15} {len(idxs):>3} {taus[idxs].mean():>5.0f}"
                  f" {dims[idxs].mean():>5.1f}"
                  f" {det.mean():>6.3f} {lam.mean():>6.3f} {entr.mean():>6.3f}")

        # -----------------------------------------------------------------
        # Step 3: Demo graphs for single neuron
        # -----------------------------------------------------------------
        print("\n[3] Building demo graphs for single neuron...")

        example_modules = ["event_0", "event_1", "event_2", "event_0|1"]
        example_idxs = {m: groups[m][0] for m in example_modules}

        # Build example RGs for recurrence plots
        example_rp_adjs = {}
        for m in example_modules:
            idx = example_idxs[m]
            rg = ts_list[idx].recurrence_graph(
                tau=taus[idx], m=dims[idx], k=CONFIG["k"])
            example_rp_adjs[m] = rg.adj

        demo_idx = example_idxs["event_0"]
        demo_ts = ts_list[demo_idx]
        demo_tau = int(taus[demo_idx])
        demo_dim = int(dims[demo_idx])

        # Three different graph representations of the same signal:
        # - RecurrenceGraph: connects time points whose embedded states are close
        # - HorizontalVisibilityGraph: connects time points with unobstructed
        #   horizontal line-of-sight (encodes amplitude structure)
        # - OrdinalPartitionNetwork: connects consecutive ordinal rank patterns
        #   (encodes temporal ordering structure)
        demo_rg = demo_ts.recurrence_graph(tau=demo_tau, m=demo_dim, k=10)
        demo_vg = demo_ts.visibility_graph(method="horizontal")
        demo_opn = demo_ts.ordinal_partition_network(tau=demo_tau)
        demo_rg_adj = demo_rg.adj
        demo_vg_adj = demo_vg.adj
        demo_opn_adj = demo_opn.adj
        demo_opn_pattern_ids = demo_opn._pattern_ids

        # -----------------------------------------------------------------
        # Step 4: Build per-neuron recurrence graphs
        # -----------------------------------------------------------------
        print("\n[4] Building recurrence graphs...")

        per_neuron_rgs = []
        for i, ts in enumerate(ts_list):
            rg = ts.recurrence_graph(tau=taus[i], m=dims[i], k=CONFIG["k"])
            per_neuron_rgs.append(rg)
        print(f"  {n_neurons} graphs built")

        # -----------------------------------------------------------------
        # Step 5: Pairwise Jaccard similarity
        # -----------------------------------------------------------------
        print("\n[5] Pairwise Jaccard similarity...")

        # Accepts RecurrenceGraph objects directly; adaptive trim (default)
        # handles different embedded lengths from per-neuron tau/dim.
        sim_matrix, _mask = pairwise_jaccard_sparse(per_neuron_rgs)

        # Split pairwise Jaccard into within-module and between-module pairs
        within_vals = []
        between_vals = []
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                if neuron_modules[i] == neuron_modules[j]:
                    within_vals.append(sim_matrix[i, j])
                else:
                    between_vals.append(sim_matrix[i, j])

        within_vals = np.array(within_vals)
        between_vals = np.array(between_vals)
        ratio = (within_vals.mean() / between_vals.mean()
                 if between_vals.mean() > 0 else float("inf"))

        print(f"  Within/between ratio: {ratio:.2f}x")

        # -----------------------------------------------------------------
        # Step 5b: Permutation test for within/between Jaccard ratio
        # -----------------------------------------------------------------
        # Shuffle module labels and recompute within/between ratio 1000 times
        # to get a null distribution. p-value = fraction of shuffles >= observed.
        print("\n[5b] Permutation test (1000 permutations)...")

        pairs_i, pairs_j = np.triu_indices(n_neurons, k=1)
        pair_sims = sim_matrix[pairs_i, pairs_j]
        all_labels_arr = np.array([neuron_modules[i] for i in range(n_neurons)])

        observed_ratio = within_vals.mean() / between_vals.mean()
        n_perms = 1000
        perm_ratios = np.empty(n_perms)
        rng = np.random.default_rng(CONFIG["seed"])
        for p in range(n_perms):
            shuffled = rng.permutation(all_labels_arr)
            same_mask = shuffled[pairs_i] == shuffled[pairs_j]
            w_mean = pair_sims[same_mask].mean()
            b_mean = pair_sims[~same_mask].mean()
            perm_ratios[p] = w_mean / b_mean if b_mean > 0 else 1.0
        perm_p_value = float((perm_ratios >= observed_ratio).mean())

        print(f"  Observed ratio: {observed_ratio:.3f}, "
              f"p-value: {perm_p_value:.4f} (1000 permutations)")

        # -----------------------------------------------------------------
        # Step 6: Network of Networks -- community detection
        # -----------------------------------------------------------------
        print("\n[6] Network of Networks (community detection)...")

        # Threshold the Jaccard matrix: keep only the strongest pairs as edges.
        # This creates a sparse "Network of Networks" where each node is a neuron
        # and edges connect neurons with similar recurrence structure.
        n_neurons = calcium.shape[0]
        upper = sim_matrix[np.triu_indices(n_neurons, k=1)]
        thr = np.percentile(upper, CONFIG["jaccard_percentile"])
        sim_thresholded = sim_matrix.copy()
        sim_thresholded[sim_thresholded < thr] = 0
        np.fill_diagonal(sim_thresholded, 0)

        # Extract giant connected component and detect communities with Louvain
        net_adj = sp.csr_matrix(sim_thresholded)
        net = Network(adj=net_adj, preprocessing="giant_cc", create_nx_graph=True)

        communities = nx_comm.louvain_communities(
            net.graph, weight="weight", seed=CONFIG["seed"])
        communities = sorted(communities, key=len, reverse=True)
        # Convert frozensets to sets for pickle
        communities = [set(c) for c in communities]
        n_detected = len(communities)

        mod_real = nx_comm.modularity(net.graph, communities, weight="weight")

        # Baseline: randomize edge weights (preserving degree) and re-run Louvain
        net_rand = net.randomize(rmode="adj_iom")
        comms_rand = nx_comm.louvain_communities(
            net_rand.graph, weight="weight", seed=CONFIG["seed"])
        mod_rand = nx_comm.modularity(net_rand.graph, comms_rand, weight="weight")

        nodes_in_net = set()
        for comm in communities:
            nodes_in_net.update(comm)

        true_labels = []
        detected_labels = []
        for ci, comm in enumerate(communities):
            for node in comm:
                detected_labels.append(ci)
                true_labels.append(neuron_modules.get(node, "unknown"))

        mod_to_int = {m: i for i, m in enumerate(module_names)}
        true_int = [mod_to_int.get(t, -1) for t in true_labels]
        ari = adjusted_rand_score(true_int, detected_labels)

        print(f"  ARI={ari:.3f}, modularity={mod_real:.3f} "
              f"(shuffled: {mod_rand:.3f})")

        # Save net adjacency for cache (before giant_cc preprocessing)
        net_adj = net.adj

        # -----------------------------------------------------------------
        # Save cache
        # -----------------------------------------------------------------
        _save_cache({
            "calcium": calcium,
            "taus": taus,
            "dims": dims,
            "rqa_results": rqa_results,
            "sim_matrix": sim_matrix,
            "demo_rg_adj": demo_rg_adj,
            "demo_vg_adj": demo_vg_adj,
            "demo_opn_adj": demo_opn_adj,
            "demo_opn_pattern_ids": demo_opn_pattern_ids,
            "demo_idx": demo_idx,
            "demo_tau": demo_tau,
            "demo_dim": demo_dim,
            "example_rp_adjs": example_rp_adjs,
            "example_idxs": example_idxs,
            "communities": communities,
            "nodes_in_net": nodes_in_net,
            "ari": ari,
            "mod_real": mod_real,
            "mod_rand": mod_rand,
            "n_detected": n_detected,
            "net_adj": net_adj,
            "thr": thr,
            "within_vals": within_vals,
            "between_vals": between_vals,
            "ratio": ratio,
            "perm_p_value": perm_p_value,
            "perm_ratios": perm_ratios,
        })

    # =================================================================
    # PLOTTING (always runs, uses cached or fresh data)
    # =================================================================
    n_neurons, n_frames = calcium.shape
    fps_eff = CONFIG["fps"]
    example_modules = ["event_0", "event_1", "event_2", "event_0|1"]
    mod_to_int = {m: i for i, m in enumerate(module_names)}

    # -----------------------------------------------------------------
    # Figure 1: Recurrence plots for representative neurons (time in s)
    # -----------------------------------------------------------------
    print("\n[Plot] Recurrence plots (representative neurons)...")

    fig_rp, axes_rp = plt.subplots(1, len(example_modules),
                                    figsize=(18, 4.5))
    for ax, m in zip(axes_rp, example_modules):
        idx = example_idxs[m]
        rg = RecurrenceGraph.from_adjacency(example_rp_adjs[m])
        plot_recurrence(rg, ax=ax, markersize=0.2)
        det = rqa_results[idx]["DET"]
        ax.set_title(f"{m} (n{idx}, DET={det:.2f})", fontsize=10)
        _set_rp_time_axes(ax, rg.adj.shape[0], fps_eff)
    fig_rp.suptitle("Recurrence plots by module", fontsize=13, y=1.02)
    fig_rp.tight_layout()

    # -----------------------------------------------------------------
    # Figure 2: Three graph representations of a single neuron
    # -----------------------------------------------------------------
    print("[Plot] Three graph representations of a single neuron...")

    demo_ca = calcium[demo_idx]

    fig_graphs, axes_g = plt.subplots(1, 3, figsize=(18, 6))

    from scipy.stats import rankdata

    graph_items = [
        (demo_rg_adj, "RecurrenceGraph (k=10)", "rg"),
        (demo_vg_adj, "HorizontalVisibilityGraph", "vg"),
        (demo_opn_adj, "OrdinalPartitionNetwork", "opn"),
    ]

    for ax, (adj, title, gtype) in zip(axes_g, graph_items):
        G = nx.from_scipy_sparse_array(adj)

        if gtype == "rg":
            gcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(gcc).copy()

        nodes = sorted(G.nodes())

        # Color nodes by calcium activity: for RG/HVG nodes are time indices,
        # for OPN nodes are pattern IDs -- average the calcium values that
        # produced each pattern
        if gtype == "opn":
            pattern_ids = demo_opn_pattern_ids
            node_vals = np.zeros(len(nodes))
            for vi, node in enumerate(nodes):
                mask = pattern_ids == node
                if mask.any():
                    node_vals[vi] = np.mean(demo_ca[:len(pattern_ids)][mask])
        else:
            node_vals = demo_ca[nodes]

        pos = nx.spring_layout(G, seed=CONFIG["seed"], iterations=80)

        pct_vals = rankdata(node_vals, method="average") / len(node_vals)

        edge_lines = [[pos[u], pos[v]] for u, v in G.edges()]
        if edge_lines:
            lc = mcoll.LineCollection(
                edge_lines, colors="0.8", linewidths=0.15, alpha=0.15)
            ax.add_collection(lc)

        xy = np.array([pos[n] for n in nodes])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=pct_vals, cmap="inferno",
                        s=8, vmin=0, vmax=1, zorder=2,
                        edgecolors="none", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        plt.colorbar(sc, ax=ax, label="Calcium percentile", shrink=0.8)

    fig_graphs.suptitle(
        f"Three graph views of neuron {demo_idx} (event_0)",
        fontsize=13, y=1.02)
    fig_graphs.tight_layout()

    # -----------------------------------------------------------------
    # Print analysis summary
    # -----------------------------------------------------------------
    print(f"\n[Summary] Within/between Jaccard: {ratio:.2f}x")
    print(f"  Permutation test p-value: {perm_p_value:.4f}")
    print(f"  Threshold: {thr:.5f} ({CONFIG['jaccard_percentile']}th pctl)")
    print(f"  {n_detected} communities, ARI={ari:.3f}, "
          f"modularity={mod_real:.3f} (shuffled: {mod_rand:.3f})")

    for ci, comm in enumerate(communities):
        mod_counts = {}
        for node in comm:
            m = neuron_modules.get(node, "unknown")
            mod_counts[m] = mod_counts.get(m, 0) + 1
        dominant = max(mod_counts, key=mod_counts.get)
        purity = mod_counts[dominant] / len(comm)
        composition = ", ".join(f"{m}:{c}" for m, c in sorted(mod_counts.items()))
        print(f"    C{ci}: {len(comm):>3} neurons [{composition}]"
              f"  (dominant: {dominant}, purity: {purity:.0%})")

    legend_order = ["event_0", "event_1", "event_2",
                    "event_0|1", "event_0|2", "event_1|2"]

    # -----------------------------------------------------------------
    # Figure 4: Population network graph with convex hulls
    # -----------------------------------------------------------------
    print("\n[Plot] Population network graph...")

    fig_net, ax_net = plt.subplots(figsize=(7, 7))

    # Two-step layout: spectral gives a stable initial placement,
    # then spring-layout refines for visual clarity
    pos_init = nx.spectral_layout(net.graph)
    pos_net = nx.spring_layout(
        net.graph, pos=pos_init, k=0.02 / np.sqrt(len(net.graph)),
        iterations=200, seed=CONFIG["seed"])

    # Draw edges
    edge_lines = [[pos_net[u], pos_net[v]] for u, v in net.graph.edges()]
    if edge_lines:
        lc_net = mcoll.LineCollection(
            edge_lines, colors="0.65", linewidths=0.4, alpha=0.3)
        ax_net.add_collection(lc_net)

    # Draw nodes colored by ground-truth module
    node_list = list(net.graph.nodes())
    xy_net = np.array([pos_net[n] for n in node_list])
    node_colors = [MODULE_COLORS[neuron_modules[n]] for n in node_list]
    ax_net.scatter(xy_net[:, 0], xy_net[:, 1], c=node_colors,
                   s=40, edgecolors="k", linewidths=0.3,
                   zorder=2, alpha=0.85)

    # Legend (column-major 2x3)
    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=MODULE_COLORS[g],
                      markeredgecolor="k", markeredgewidth=0.3,
                      markersize=7, label=MODULE_SHORT[g])
               for g in legend_order]
    # Reorder for column-major layout: E1,E1uE2,E2,E1uE3,E3,E2uE3
    handles = [handles[0], handles[3], handles[1],
               handles[4], handles[2], handles[5]]
    ax_net.legend(handles=handles, fontsize=9, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ax_net.set_title(
        f"Jaccard similarity network ({net.n} neurons, "
        f"{net.graph.number_of_edges()} edges, ARI={ari:.2f})",
        fontsize=12)
    ax_net.set_axis_off()

    # -----------------------------------------------------------------
    # Figure 5: Jaccard matrices with inferno + colored sidebars
    # -----------------------------------------------------------------
    print("[Plot] Jaccard summary matrices...")

    fig_sum, axes_sum = plt.subplots(1, 2, figsize=(14, 6))

    # Stretch colormap contrast: clip at 5th/99th percentile so the
    # block-diagonal structure stands out against the background
    upper_tri = sim_matrix[np.triu_indices(n_neurons, k=1)]
    vmin_j = np.percentile(upper_tri, 5)
    vmax_j = np.percentile(upper_tri, 99)

    # -- Helper: draw colored sidebar strip --
    def _draw_sidebar(ax, order, neuron_modules_map, side="left"):
        """Draw a colored strip showing module identity along an axis."""
        n = len(order)
        mod_arr = [neuron_modules_map.get(idx, "event_0") for idx in order]
        # Build unique color list preserving MODULE_COLORS order
        unique_mods = list(MODULE_COLORS.keys())
        mod_to_idx = {m: i for i, m in enumerate(unique_mods)}
        color_list = [MODULE_COLORS[m] for m in unique_mods]
        cmap_sidebar = ListedColormap(color_list)

        sidebar_data = np.array([mod_to_idx[m] for m in mod_arr])

        # Get the position of the main imshow axes
        pos = ax.get_position()

        if side == "left":
            # Left sidebar
            sb_ax = ax.figure.add_axes(
                [pos.x0 - 0.018, pos.y0, 0.012, pos.height])
            sb_ax.imshow(sidebar_data.reshape(-1, 1), cmap=cmap_sidebar,
                         aspect="auto", interpolation="nearest",
                         vmin=0, vmax=len(unique_mods) - 1)
            sb_ax.set_xticks([])
            sb_ax.set_yticks([])
            sb_ax.spines[:].set_visible(False)

        # Bottom sidebar
        sb_ax_b = ax.figure.add_axes(
            [pos.x0, pos.y0 - 0.018, pos.width, 0.012])
        sb_ax_b.imshow(sidebar_data.reshape(1, -1), cmap=cmap_sidebar,
                       aspect="auto", interpolation="nearest",
                       vmin=0, vmax=len(unique_mods) - 1)
        sb_ax_b.set_xticks([])
        sb_ax_b.set_yticks([])
        sb_ax_b.spines[:].set_visible(False)

    # Left panel: neurons ordered by ground-truth module.
    # If recurrence captures function, diagonal blocks should be bright.
    mod_order = []
    for m in module_names:
        mod_order.extend(sorted(groups[m]))
    mod_arr = np.array(mod_order)
    sim_by_mod = sim_matrix[np.ix_(mod_arr, mod_arr)]
    np.fill_diagonal(sim_by_mod, 0)

    im1 = axes_sum[0].imshow(
        sim_by_mod, cmap="inferno", aspect="auto",
        vmin=vmin_j, vmax=vmax_j)
    cumsum_mod = np.cumsum([0] + [len(groups[m]) for m in module_names])
    for b in cumsum_mod[1:-1]:
        axes_sum[0].axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.7)
        axes_sum[0].axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.7)
    axes_sum[0].set_xticks([])
    axes_sum[0].set_yticks([])
    axes_sum[0].set_title("Jaccard similarity (ordered by module)")
    plt.colorbar(im1, ax=axes_sum[0], label="Jaccard", shrink=0.8)

    # Right panel: reordered by Louvain communities (no ground truth used).
    # Similar block structure = communities match modules.
    comm_order = []
    for comm in communities:
        comm_order.extend(sorted(comm))
    missing = [i for i in range(n_neurons) if i not in nodes_in_net]
    comm_order.extend(missing)
    comm_arr = np.array(comm_order)
    sim_by_comm = sim_matrix[np.ix_(comm_arr, comm_arr)]
    np.fill_diagonal(sim_by_comm, 0)

    im2 = axes_sum[1].imshow(
        sim_by_comm, cmap="inferno", aspect="auto",
        vmin=vmin_j, vmax=vmax_j)
    cumsum_comm = np.cumsum(
        [0] + [len(c) for c in communities] + [len(missing)])
    for b in cumsum_comm[1:-1]:
        axes_sum[1].axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.7)
        axes_sum[1].axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.7)
    axes_sum[1].set_xticks([])
    axes_sum[1].set_yticks([])
    axes_sum[1].set_title("Jaccard similarity (ordered by community)")
    plt.colorbar(im2, ax=axes_sum[1], label="Jaccard", shrink=0.8)

    fig_sum.tight_layout()

    # Draw sidebars after tight_layout so positions are final
    _draw_sidebar(axes_sum[0], mod_order, neuron_modules, side="left")
    _draw_sidebar(axes_sum[1], comm_order, neuron_modules, side="left")

    # -----------------------------------------------------------------
    # Figure 7: Confusion matrix (ground-truth vs detected communities)
    # -----------------------------------------------------------------
    print("[Plot] Confusion matrix (module vs community)...")

    # Build confusion matrix: rows=ground-truth, cols=detected community
    n_comm = len(communities)
    confusion = np.zeros((len(module_names), n_comm), dtype=int)
    for ci, comm in enumerate(communities):
        for node in comm:
            m = neuron_modules.get(node, "unknown")
            if m in module_names:
                mi = module_names.index(m)
                confusion[mi, ci] += 1

    fig_conf, ax_conf = plt.subplots(
        figsize=(max(6, n_comm * 0.8 + 2), max(5, len(module_names) * 0.7 + 1)))
    im_conf = ax_conf.imshow(confusion, cmap="Blues", aspect="auto")
    # Text annotations
    for i in range(len(module_names)):
        for j in range(n_comm):
            val = confusion[i, j]
            if val > 0:
                color = "white" if val > confusion.max() / 2 else "black"
                ax_conf.text(j, i, str(val), ha="center", va="center",
                             fontsize=10, color=color, fontweight="bold")
    ax_conf.set_xticks(range(n_comm))
    ax_conf.set_xticklabels([f"C{i}" for i in range(n_comm)], fontsize=9)
    ax_conf.set_yticks(range(len(module_names)))
    ax_conf.set_yticklabels([MODULE_SHORT.get(m, m) for m in module_names],
                            fontsize=9)
    ax_conf.set_xlabel("Detected community", fontsize=11)
    ax_conf.set_ylabel("Ground-truth module", fontsize=11)
    ax_conf.set_title(f"Community detection confusion matrix (ARI={ari:.3f})",
                      fontsize=12)
    plt.colorbar(im_conf, ax=ax_conf, label="Neuron count", shrink=0.8)
    fig_conf.tight_layout()

    # -----------------------------------------------------------------
    # Save all figures
    # -----------------------------------------------------------------
    print("\n[Saving] Figures...")

    rp_path = os.path.join(output_dir, "recurrence_population_rp.png")
    fig_rp.savefig(rp_path, dpi=400, bbox_inches="tight")
    plt.close(fig_rp)
    print(f"  [OK] {rp_path}")

    graphs_path = os.path.join(output_dir, "recurrence_population_graphs.png")
    fig_graphs.savefig(graphs_path, dpi=400, bbox_inches="tight")
    plt.close(fig_graphs)
    print(f"  [OK] {graphs_path}")

    net_path = os.path.join(output_dir, "recurrence_population_network.png")
    fig_net.savefig(net_path, dpi=400, bbox_inches="tight")
    plt.close(fig_net)
    print(f"  [OK] {net_path}")

    sum_path = os.path.join(output_dir, "recurrence_population_summary.png")
    fig_sum.savefig(sum_path, dpi=400, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"  [OK] {sum_path}")


    conf_path = os.path.join(output_dir, "recurrence_population_confusion.png")
    fig_conf.savefig(conf_path, dpi=400, bbox_inches="tight")
    plt.close(fig_conf)
    print(f"  [OK] {conf_path}")

    # -----------------------------------------------------------------
    # Conclusions
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Conclusions")
    print("=" * 70)

    print(f"\n  Population: {n_neurons} neurons in {len(POPULATION)} modules")
    print(f"  (mirrors cell-cell network example population)")

    print(f"\n  1. Recurrence similarity reflects shared function:")
    print(f"     Within-module Jaccard = {within_vals.mean():.5f}, "
          f"between = {between_vals.mean():.5f} ({ratio:.2f}x)")
    print(f"     Permutation test p-value = {perm_p_value:.4f} "
          f"(n=1000 permutations)")
    if ratio > 1.5:
        print(f"     -> Clear separation: same-module pairs share more"
              f" recurrence structure.")
    elif ratio > 1.1:
        print(f"     -> Modest separation: same-module pairs are somewhat"
              f" more similar.")
    else:
        print(f"     -> Weak separation: recurrence similarity alone may"
              f" not distinguish modules at this recording length.")

    print(f"\n  2. Community detection recovers functional modules:")
    print(f"     {n_detected} communities detected, ARI = {ari:.3f}, "
          f"modularity = {mod_real:.3f} (shuffled: {mod_rand:.3f})")
    if ari > 0.4:
        print(f"     -> Good recovery of ground-truth modules.")
    elif ari > 0.2:
        print(f"     -> Partial recovery -- single-event modules likely"
              f" recovered, OR modules may merge with parents.")
    else:
        print(f"     -> Weak recovery -- longer recordings or different"
              f" parameters may improve results.")

    print(f"\n  3. Single-event vs OR-combination modules:")
    single_mods = [m for m in module_names if "|" not in m]
    or_mods = [m for m in module_names if "|" in m]
    single_within = [sim_matrix[i, j]
                     for m in single_mods
                     for i in groups[m] for j in groups[m] if i < j]
    or_within = [sim_matrix[i, j]
                 for m in or_mods
                 for i in groups[m] for j in groups[m] if i < j]
    if single_within and or_within:
        print(f"     Single-event within-module Jaccard:"
              f" {np.mean(single_within):.5f}")
        print(f"     OR-combination within-module Jaccard:"
              f" {np.mean(or_within):.5f}")
        if np.mean(or_within) > np.mean(single_within):
            print(f"     -> OR modules show higher within-similarity"
                  f" (driven by multiple events).")
        else:
            print(f"     -> OR modules show lower cohesion"
                  f" (heterogeneous event responses).")

    print(f"\n  Compare with cell-cell network example (MI-based):")
    print(f"  Same population, different instrument. MI significance")
    print(f"  tests each neuron pair directly; recurrence analysis")
    print(f"  compares dynamical fingerprints without behavioral labels.")

    print(f"\n  Output figures:")
    for p in [rp_path, graphs_path, net_path,
              sum_path, conf_path]:
        print(f"    {os.path.basename(p)}")


if __name__ == "__main__":
    main()
