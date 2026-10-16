"""
Fig 2 data — generate synthetic population, run INTENSE + network + Isomap.

Usage:
    python publications/elife/fig2/fig2_data.py

Outputs: fig2_*.npz in the cache directory (see elife_paths.py)
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths

# Limit MKL threading overhead for FFT-heavy INTENSE
os.environ.setdefault("OMP_NUM_THREADS", "1")

CACHE_DIR = elife_paths.CACHE_DIR
SEED = 42
DURATION = 600       # seconds
FPS = 20.0
DS = 5               # downsampling for INTENSE

# ── Population configuration ────────────────────────────────
EVENT_POPULATION = [
    {"name": "event_0_cells", "count": 60, "features": ["event_0"]},
    {"name": "event_1_cells", "count": 60, "features": ["event_1"]},
    {"name": "event_2_cells", "count": 60, "features": ["event_2"]},
    {"name": "event_0_or_1",  "count": 20, "features": ["event_0", "event_1"], "combination": "or"},
    {"name": "event_0_or_2",  "count": 20, "features": ["event_0", "event_2"], "combination": "or"},
    {"name": "event_1_or_2",  "count": 20, "features": ["event_1", "event_2"], "combination": "or"},
    {"name": "nonselective",  "count": 60, "features": []},
]
N_HD = 60
N_NONSEL_EXTRA = 40   # additional non-selective padding


def generate_population():
    """Generate merged synthetic population: events + HD + non-selective."""
    from driada.experiment.synthetic.generators import (
        generate_tuned_selectivity_exp,
        generate_circular_manifold_exp,
    )

    print("=" * 60)
    print("Generating event-modular population...")
    exp_events = generate_tuned_selectivity_exp(
        population=EVENT_POPULATION,
        n_discrete_features=3,
        duration=DURATION,
        fps=FPS,
        seed=SEED,
        verbose=True,
    )

    print("\nGenerating head-direction population...")
    exp_hd, hd_info = generate_circular_manifold_exp(
        n_neurons=N_HD,
        kappa=4.0,
        duration=DURATION,
        fps=FPS,
        seed=SEED + 1,
        return_info=True,
        verbose=True,
    )

    # ── Merge into single calcium array ──
    ca_events = exp_events.calcium.data
    ca_hd = exp_hd.calcium.data
    n_frames = min(ca_events.shape[1], ca_hd.shape[1])
    ca_events = ca_events[:, :n_frames]
    ca_hd = ca_hd[:, :n_frames]

    # Non-selective padding (Gaussian noise matching calcium stats)
    rng = np.random.default_rng(SEED + 2)
    noise_std = np.std(ca_events) * 0.3
    ca_nonsel = np.abs(rng.normal(0, noise_std, (N_NONSEL_EXTRA, n_frames)))

    calcium_merged = np.vstack([ca_events, ca_hd, ca_nonsel])

    # ── Build ground-truth labels ──
    n_event = ca_events.shape[0]
    n_total = calcium_merged.shape[0]

    gt_event_types = exp_events.ground_truth["neuron_types"]
    group_names = []
    for i in range(n_total):
        if i < n_event:
            group_names.append(gt_event_types[i])
        elif i < n_event + N_HD:
            group_names.append("hd_cells")
        else:
            group_names.append("nonselective")

    # Expected cell-feature pairs for selectivity validation
    expected_pairs = list(exp_events.ground_truth["expected_pairs"])
    for j in range(N_HD):
        expected_pairs.append((n_event + j, "head_direction"))

    hd_trajectory = hd_info["head_direction"][:n_frames]
    preferred_dirs = hd_info["preferred_directions"]

    print(f"\nMerged population: {n_total} neurons "
          f"({n_event} event + {N_HD} HD + {N_NONSEL_EXTRA} non-selective)")
    print(f"Frames: {n_frames} ({n_frames/FPS:.0f}s at {FPS} Hz)")
    print(f"Expected selective pairs: {len(expected_pairs)}")

    return {
        "calcium": calcium_merged,
        "n_event": n_event,
        "n_hd": N_HD,
        "n_nonsel": N_NONSEL_EXTRA,
        "n_frames": n_frames,
        "group_names": group_names,
        "expected_pairs": expected_pairs,
        "hd_trajectory": hd_trajectory,
        "preferred_dirs": preferred_dirs,
        "exp_events": exp_events,
        "exp_hd": exp_hd,
    }


def run_intense(exp_events):
    """Run INTENSE selectivity detection on event experiment.

    Returns
    -------
    mi_matrix : ndarray (n_neurons, n_features) — MI values
    sig_matrix : ndarray (n_neurons, n_features) — binary significance
    feat_ids : list of str — feature names (column order)
    """
    from driada.intense.pipelines import compute_cell_feat_significance

    print("\n" + "=" * 60)
    print("Running INTENSE selectivity detection...")

    stats, significance, info, intense_res, _ = compute_cell_feat_significance(
        exp_events,
        data_type="calcium",
        mi_estimator="gcmi",
        mode="two_stage",
        n_shuffles_stage1=100,
        n_shuffles_stage2=10000,
        ds=DS,
        pval_thr=0.01,
        multicomp_correction="holm",
        verbose=True,
        seed=SEED,
    )

    # Extract matrices from info dict
    # stage_2_significance['stage2'] is (n_neurons, n_features) binary
    sig_tables = info["stage_2_significance"]
    sig_matrix = sig_tables["stage2"].astype(float)

    # stage_2_stats['me'] is (n_neurons, n_features) MI values
    stats_tables = info["stage_2_stats"]
    mi_matrix = stats_tables["me"]

    # Feature names in column order
    feat_ids = [ts.name for ts in exp_events.dynamic_features.values()]

    print(f"Shape: {mi_matrix.shape} (neurons x features)")
    print(f"Features: {feat_ids}")
    print(f"Significant pairs: {int(np.sum(sig_matrix))}")

    return mi_matrix, sig_matrix, feat_ids


def run_isomap(calcium_hd, hd_trajectory):
    """Run Isomap on HD-selective neurons to recover circular manifold."""
    from driada.dim_reduction.data import MVData

    print("\n" + "=" * 60)
    print("Running Isomap on HD neurons...")

    mvdata = MVData(calcium_hd, downsampling=DS)
    embedding = mvdata.get_embedding(method="isomap", dim=2, n_neighbors=50)
    coords = embedding.coords.T   # (n_samples, 2)

    # Downsample ground truth to match
    hd_ds = hd_trajectory[::DS][:coords.shape[0]]

    print(f"Embedding shape: {coords.shape}")
    return coords, hd_ds


def run_network(exp_events):
    """Run cell-cell MI network analysis on event experiment."""
    from driada.intense.pipelines import compute_cell_cell_significance
    import networkx as nx

    print("\n" + "=" * 60)
    print("Running cell-cell network analysis...")

    sim_mat, sig_mat, pval_mat, cells, info = compute_cell_cell_significance(
        exp_events,
        data_type="calcium",
        ds=DS,
        n_shuffles_stage1=100,
        n_shuffles_stage2=1000,
        pval_thr=0.01,
        multicomp_correction="holm",
        verbose=True,
        seed=SEED,
    )

    # Build Network and detect communities
    from driada.network import Network
    import scipy.sparse as sp
    import networkx.algorithms.community as nx_comm

    adj = sp.csr_matrix(sig_mat)
    net = Network(adj=adj, preprocessing="giant_cc")

    communities = nx_comm.louvain_communities(net.graph, seed=SEED)
    module_assignment = {}
    for mod_idx, comm in enumerate(communities):
        for node in comm:
            module_assignment[node] = mod_idx

    detected_labels = np.array([module_assignment.get(i, -1)
                                for i in range(len(cells))])

    # Remove low-degree nodes from layout (they distort spring positioning)
    min_degree = 5
    low_deg = [n for n, d in net.graph.degree() if d < min_degree]
    graph_viz = net.graph.copy()
    graph_viz.remove_nodes_from(low_deg)

    # Compute spring layout for graph visualization
    pos = nx.spring_layout(
        graph_viz,
        k=1.5 / np.sqrt(max(len(graph_viz), 1)),
        iterations=200,
        seed=SEED,
    )
    # Store as (n_nodes, 2) array with node order
    nodes = sorted(graph_viz.nodes())
    pos_array = np.array([pos[n] for n in nodes])
    node_ids = np.array(nodes)
    edges = np.array(list(graph_viz.edges()))
    print(f"Removed {len(low_deg)} low-degree nodes from graph layout")

    print(f"Significant connections: {np.sum(sig_mat) / 2:.0f}")
    print(f"Detected {len(communities)} communities")

    return sim_mat, sig_mat, detected_labels, cells, pos_array, node_ids, edges


def compute_metrics(pop_data, sig_matrix, feat_ids, detected_labels):
    """Compute F1 for selectivity and ARI/NMI for network communities."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        adjusted_rand_score, normalized_mutual_info_score,
    )

    # ── Selectivity F1 ──
    expected = set()
    for neuron_idx, feat_name in pop_data["expected_pairs"]:
        if neuron_idx < sig_matrix.shape[0] and feat_name in feat_ids:
            col = feat_ids.index(feat_name)
            expected.add((neuron_idx, col))

    gt_flat = np.zeros_like(sig_matrix, dtype=int)
    for r, c in expected:
        gt_flat[r, c] = 1

    f1 = f1_score(gt_flat.ravel(), sig_matrix.ravel().astype(int))
    precision = precision_score(gt_flat.ravel(), sig_matrix.ravel().astype(int),
                                zero_division=0)
    recall = recall_score(gt_flat.ravel(), sig_matrix.ravel().astype(int))

    print(f"\nSelectivity: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    # ── Network ARI/NMI ──
    n_event = pop_data["n_event"]
    group_names = pop_data["group_names"][:n_event]
    unique_groups = sorted(set(group_names))
    gt_labels = np.array([unique_groups.index(g) for g in group_names])

    # Only compare neurons present in giant CC
    n_compare = min(len(gt_labels), len(detected_labels))
    gt_sub = gt_labels[:n_compare]
    det_sub = detected_labels[:n_compare]

    ari = adjusted_rand_score(gt_sub, det_sub)
    nmi = normalized_mutual_info_score(gt_sub, det_sub)

    print(f"Network: ARI={ari:.3f}, NMI={nmi:.3f}")

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "ari": ari, "nmi": nmi,
    }


def save_cache(pop_data, mi_intense, sig_intense, feat_ids,
               isomap_coords, hd_ds, sim_net, sig_net,
               detected_labels, metrics, net_pos, net_nodes, net_edges):
    """Save all results to NPZ cache files."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Population metadata
    np.savez(
        os.path.join(CACHE_DIR, "fig2_population.npz"),
        calcium=pop_data["calcium"],
        n_event=pop_data["n_event"],
        n_hd=pop_data["n_hd"],
        n_nonsel=pop_data["n_nonsel"],
        group_names=np.array(pop_data["group_names"], dtype=object),
        hd_trajectory=pop_data["hd_trajectory"],
        preferred_dirs=pop_data["preferred_dirs"],
    )

    # INTENSE results
    np.savez(
        os.path.join(CACHE_DIR, "fig2_intense.npz"),
        mi_mat=mi_intense,
        sig_mat=sig_intense,
        feat_ids=np.array(feat_ids, dtype=object),
    )

    # Isomap results
    np.savez(
        os.path.join(CACHE_DIR, "fig2_isomap.npz"),
        coords=isomap_coords,
        hd_downsampled=hd_ds,
    )

    # Network results (including graph layout for visualization)
    np.savez(
        os.path.join(CACHE_DIR, "fig2_network.npz"),
        sim_mat=sim_net,
        sig_mat=sig_net,
        detected_labels=detected_labels,
        graph_pos=net_pos,
        graph_nodes=net_nodes,
        graph_edges=net_edges,
    )

    # Metrics
    np.savez(
        os.path.join(CACHE_DIR, "fig2_metrics.npz"),
        **metrics,
    )

    # Save example neuron data for Panel B
    exp_events = pop_data["exp_events"]
    exp_hd = pop_data["exp_hd"]

    # Event neuron: first neuron in event_0 group (index 0)
    event_neuron_calcium = exp_events.calcium.data[0, :]
    event_feature_ts = exp_events.dynamic_features["event_0"].data

    # HD neuron: first neuron in HD group
    hd_neuron_calcium = exp_hd.calcium.data[0, :]
    hd_angles = pop_data["hd_trajectory"]

    np.savez(
        os.path.join(CACHE_DIR, "fig2_example_neurons.npz"),
        event_calcium=event_neuron_calcium,
        event_feature=event_feature_ts,
        hd_calcium=hd_neuron_calcium,
        hd_angles=hd_angles,
        preferred_dir_0=pop_data["preferred_dirs"][0],
        fps=FPS,
    )

    print(f"\nAll cache files saved to {CACHE_DIR}/")


def main():
    """Run full Fig 2 data pipeline."""
    # 1. Generate population
    pop_data = generate_population()

    # 2. Run INTENSE on event experiment
    mi_intense, sig_intense, feat_ids = run_intense(pop_data["exp_events"])

    # 3. Run Isomap on HD neurons
    calcium_hd = pop_data["calcium"][pop_data["n_event"]:pop_data["n_event"] + N_HD, :]
    isomap_coords, hd_ds = run_isomap(calcium_hd, pop_data["hd_trajectory"])

    # 4. Run network analysis on event experiment
    sim_net, sig_net, detected_labels, cells, net_pos, net_nodes, net_edges = \
        run_network(pop_data["exp_events"])

    # 5. Compute metrics
    metrics = compute_metrics(pop_data, sig_intense, feat_ids, detected_labels)

    # 6. Save everything
    save_cache(
        pop_data, mi_intense, sig_intense, feat_ids,
        isomap_coords, hd_ds, sim_net, sig_net,
        detected_labels, metrics, net_pos, net_nodes, net_edges,
    )

    print("\n" + "=" * 60)
    print("Fig 2 data pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
