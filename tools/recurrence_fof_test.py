#!/usr/bin/env python
"""
Recurrence analysis validation on real FOF (Familiar Open Field) calcium imaging data.

Full pipeline: tau estimation -> embedding dim -> per-neuron RecurrenceGraphs -> RQA ->
population JRP -> Network of Networks (Jaccard) -> spectral analysis.

Usage:
    python tools/recurrence_fof_test.py --session FOF_F36_1D --ds 5 --k 30
    python tools/recurrence_fof_test.py --load --output-dir recurrence/fof_output
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))

from driada.recurrence import (
    takens_embedding,
    RecurrenceGraph,
    estimate_tau,
    estimate_embedding_dim,
)
from driada.recurrence.population import pairwise_jaccard_sparse
from driada.recurrence.plotting import plot_recurrence
from driada.network import Network
from driada.utils.parallel import parallel_executor
from joblib import delayed
from fof_validation.loader import load_fof_session


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='FOF recurrence analysis pipeline')
    p.add_argument('--session', default='FOF_F36_1D',
                   help='FOF session name (default: FOF_F36_1D)')
    p.add_argument('--ds', type=int, default=5,
                   help='Downsample factor (default: 5)')
    p.add_argument('--k', type=int, default=5,
                   help='k-NN neighbours for recurrence graph (default: 5)')
    p.add_argument('--max-shift', type=int, default=60,
                   help='Max lag for tau estimation (default: 60)')
    p.add_argument('--max-dim', type=int, default=10,
                   help='Max embedding dimension to test (default: 10)')
    p.add_argument('--output-dir', default='recurrence_fof_output',
                   help='Output directory for figures (default: recurrence_fof_output)')
    p.add_argument('--jrp-threshold', type=float, default=0.02,
                   help='JRP agreement fraction for main plot (default: 0.02)')
    p.add_argument('--smooth-sigma', type=float, default=2.0,
                   help='Gaussian smoothing sigma before embedding (default: 2.0, 0=none)')
    p.add_argument('--load', action='store_true',
                   help='Load per-neuron results from output dir instead of recomputing')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='Parallel jobs; -1 = all cores (default: -1)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Save / Load helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_adjacencies(trimmed, output_dir):
    """Save trimmed binary adjacency matrices as compressed edge lists."""
    all_rows, all_cols, neuron_ptrs = [], [], [0]
    for rg in trimmed:
        coo = sp.triu(rg.adj).tocoo()
        all_rows.append(coo.row.astype(np.int16))
        all_cols.append(coo.col.astype(np.int16))
        neuron_ptrs.append(neuron_ptrs[-1] + len(coo.row))
    n = trimmed[0].adj.shape[0]
    np.savez_compressed(Path(output_dir) / 'adjacencies.npz',
                        rows=np.concatenate(all_rows),
                        cols=np.concatenate(all_cols),
                        ptrs=np.array(neuron_ptrs, dtype=np.int64),
                        n=np.int32(n), n_neurons=np.int32(len(trimmed)))


def load_adjacencies(output_dir):
    """Reconstruct trimmed RecurrenceGraphs from saved edge lists."""
    data = np.load(Path(output_dir) / 'adjacencies.npz')
    rows, cols = data['rows'], data['cols']
    ptrs = data['ptrs']
    n = int(data['n'])
    n_neurons = int(data['n_neurons'])
    trimmed = []
    for i in range(n_neurons):
        r = rows[ptrs[i]:ptrs[i + 1]].astype(np.int32)
        c = cols[ptrs[i]:ptrs[i + 1]].astype(np.int32)
        ones = np.ones(len(r), dtype=float)
        upper = sp.csr_matrix((ones, (r, c)), shape=(n, n))
        adj = upper + upper.T
        trimmed.append(RecurrenceGraph.from_adjacency(adj))
    return trimmed


def save_neuron_results(output_dir, taus, dims, rqa_arrays):
    """Save per-neuron parameters and RQA arrays."""
    kw = dict(taus=taus, dims=dims)
    for m, v in rqa_arrays.items():
        kw[f'rqa_{m}'] = v
    np.savez_compressed(Path(output_dir) / 'neuron_results.npz', **kw)


def load_neuron_results(output_dir):
    """Load per-neuron parameters and RQA arrays."""
    data = np.load(Path(output_dir) / 'neuron_results.npz')
    taus = data['taus']
    dims = data['dims']
    measures = ['RR', 'DET', 'LAM', 'L_mean', 'L_max', 'TT', 'ENTR']
    rqa_arrays = {m: data[f'rqa_{m}'] for m in measures}
    return taus, dims, rqa_arrays


def save_population_results(output_dir, sim_matrix, mod_real, mod_rand,
                            cluster_range, jrp_thresholds, jrp_stats):
    """Save population-level results."""
    kw = dict(
        jaccard_matrix=sim_matrix,
        mod_real=np.array(mod_real),
        mod_rand=np.array(mod_rand),
        cluster_range=np.array(list(cluster_range)),
        jrp_thresholds=np.array(jrp_thresholds),
    )
    for m in ['RR', 'DET', 'LAM', 'ENTR']:
        kw[f'jrp_{m}'] = np.array([s[m] for s in jrp_stats])
    np.savez_compressed(Path(output_dir) / 'population_results.npz', **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════
def savefig(fig, name, output_dir):
    path = Path(output_dir) / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    def log(msg):
        print(msg)
        summary_lines.append(msg)

    # ───────────────────────────────────────────────────────────────────
    # Load Data
    # ───────────────────────────────────────────────────────────────────
    log(f'\n{"="*60}')
    log(f'  FOF Recurrence Analysis: {args.session}')
    log(f'{"="*60}')

    exp, intense, disent = load_fof_session(args.session)
    calcium = exp.calcium.data[:, ::args.ds]
    n_neurons, n_frames = calcium.shape
    fps_eff = 20.0 / args.ds
    duration = n_frames / fps_eff

    log(f'\nSession:    {args.session}')
    log(f'Neurons:    {n_neurons}')
    log(f'Frames:     {n_frames} (ds={args.ds}, effective {fps_eff:.1f} fps)')
    log(f'Duration:   {duration:.1f} s')

    if args.smooth_sigma > 0:
        from driada.utils.signals import filter_signals
        calcium = filter_signals(calcium, method='gaussian', sigma=args.smooth_sigma)
        log(f'Smoothing:  Gaussian sigma={args.smooth_sigma}')

    # ───────────────────────────────────────────────────────────────────
    # Per-Neuron Tau + Dim + RecurrenceGraph
    # ───────────────────────────────────────────────────────────────────
    can_load = (args.load
                and (output_dir / 'adjacencies.npz').exists()
                and (output_dir / 'neuron_results.npz').exists())

    if can_load:
        log(f'\n--- Loading per-neuron results from {output_dir} ---')
        t1 = time.time()
        taus, dims, rqa_arrays = load_neuron_results(output_dir)
        trimmed = load_adjacencies(output_dir)
        n_neurons = len(trimmed)
        min_n = trimmed[0].adj.shape[0]
        dt_build = time.time() - t1
        log(f'  Loaded {n_neurons} neurons ({min_n} time points) in {dt_build:.1f}s')
    else:
        log(f'\n--- Tau + Dim + RecurrenceGraph Construction ---')
        t1 = time.time()

        def build_neuron(i):
            signal = calcium[i]
            tau_i = estimate_tau(signal, max_shift=args.max_shift)
            m_i = estimate_embedding_dim(signal, tau=int(tau_i),
                                         max_dim=args.max_dim)
            emb = takens_embedding(signal, tau=int(tau_i), m=int(m_i))
            theiler = int(tau_i) * (int(m_i) - 1) + 1
            rg = RecurrenceGraph(emb, method='knn', k=args.k,
                                 theiler_window=theiler)
            return int(tau_i), int(m_i), rg

        with parallel_executor(args.n_jobs, backend='threading') as par:
            results = par(delayed(build_neuron)(i) for i in range(n_neurons))

        taus = np.array([r[0] for r in results])
        dims = np.array([r[1] for r in results])
        graphs = [r[2] for r in results]

        # Trim to common size
        sizes = [rg.n for rg in graphs]
        min_n = min(sizes)
        max_n = max(sizes)
        trimmed = []
        for rg in graphs:
            if rg.n > min_n:
                trimmed.append(RecurrenceGraph.from_adjacency(
                    rg.adj[:min_n, :min_n], theiler_window=rg.theiler_window))
            else:
                trimmed.append(rg)

        dt_build = time.time() - t1
        log(f'  Sizes: min={min_n}, max={max_n} -> trimmed to {min_n}')
        log(f'  Time: {dt_build:.1f}s')

        # Per-neuron RQA
        log(f'\n--- RQA Analysis ---')
        t_rqa = time.time()
        rqa_results = [rg.rqa() for rg in trimmed]
        measures = ['RR', 'DET', 'LAM', 'L_mean', 'L_max', 'TT', 'ENTR']
        rqa_arrays = {m: np.array([r[m] for r in rqa_results]) for m in measures}
        dt_rqa = time.time() - t_rqa
        log(f'  Time: {dt_rqa:.1f}s')

        # Save per-neuron results + adjacencies
        save_neuron_results(output_dir, taus, dims, rqa_arrays)
        save_adjacencies(trimmed, output_dir)
        log(f'  Saved neuron_results.npz + adjacencies.npz')

    # Print per-neuron stats
    log(f'\n  Per-neuron parameters (n={n_neurons}):')
    log(f'  Tau:  mean={taus.mean():.1f}, std={taus.std():.1f}, '
        f'range=[{taus.min()}, {taus.max()}]')
    log(f'  Dim:  mean={dims.mean():.1f}, std={dims.std():.1f}, '
        f'range=[{dims.min()}, {dims.max()}]')
    measures = ['RR', 'DET', 'LAM', 'L_mean', 'L_max', 'TT', 'ENTR']
    log(f'\n  RQA Summary:')
    log(f'  {"Measure":<10} {"Mean":>8} {"Std":>8} {"Median":>8} {"Min":>8} {"Max":>8}')
    log(f'  {"-"*50}')
    for m in measures:
        v = rqa_arrays[m]
        log(f'  {m:<10} {v.mean():8.3f} {v.std():8.3f} {np.median(v):8.3f} '
            f'{v.min():8.3f} {v.max():8.3f}')

    # Fig 1: Tau and dim histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(taus, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Tau (samples)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Tau distribution (n={n_neurons})')
    ax1.axvline(np.median(taus), color='red', ls='--', label=f'median={np.median(taus):.0f}')
    ax1.legend()

    ax2.hist(dims, bins=range(1, args.max_dim + 2), edgecolor='black', alpha=0.7,
             align='left')
    ax2.set_xlabel('Embedding dimension')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Dim distribution (n={n_neurons})')
    ax2.axvline(np.median(dims), color='red', ls='--', label=f'median={np.median(dims):.0f}')
    ax2.legend()
    fig.tight_layout()
    savefig(fig, 'fig01_tau_dim_distributions.png', output_dir)

    # Fig 2: RQA distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, ['DET', 'LAM', 'ENTR']):
        ax.hist(rqa_arrays[m], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(m)
        ax.set_ylabel('Count')
        ax.set_title(f'{m} distribution')
        ax.axvline(np.median(rqa_arrays[m]), color='red', ls='--',
                    label=f'median={np.median(rqa_arrays[m]):.3f}')
        ax.legend(fontsize=8)
    fig.suptitle(f'RQA distributions ({args.session}, n={n_neurons})')
    fig.tight_layout()
    savefig(fig, 'fig02_rqa_distributions.png', output_dir)

    # ───────────────────────────────────────────────────────────────────
    # Population JRP (all neurons)
    # ───────────────────────────────────────────────────────────────────
    log(f'\n--- Population JRP ({n_neurons} neurons) ---')
    t4 = time.time()

    # Sum all adjacency matrices once, derive mean and thresholded JRPs
    summed = sp.csr_matrix((min_n, min_n), dtype=float)
    for rg in trimmed:
        summed = summed + rg.adj.astype(float)
    mean_matrix = (summed / n_neurons).toarray()

    # Fig 3: Population mean recurrence heatmap
    median_tau = int(np.median(taus))
    diag_mask = np.abs(np.arange(min_n)[:, None] - np.arange(min_n)[None, :]) < median_tau
    display_matrix = mean_matrix.copy()
    display_matrix[diag_mask] = 0
    vmax = np.percentile(display_matrix[display_matrix > 0], 99) if np.any(display_matrix > 0) else 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(display_matrix, cmap='hot', aspect='equal', origin='lower',
                   vmin=0, vmax=vmax, interpolation='none')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    ax.set_title(f'Population Recurrence ({n_neurons} neurons, agreement fraction)')
    plt.colorbar(im, ax=ax, label='Fraction of neurons', shrink=0.8)
    savefig(fig, 'fig03_population_recurrence.png', output_dir)

    # Threshold sweep with RQA (reuse summed matrix)
    jrp_thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    jrp_stats = []
    jrp_at_chosen = None
    summed_coo = summed.tocoo()

    log(f'\n  JRP threshold sweep:')
    log(f'  {"Thresh":>8} {"nnz":>10} {"RR":>10} {"DET":>8} {"LAM":>8} {"ENTR":>8}')
    log(f'  {"-"*56}')

    def _jrp_from_sum(threshold):
        min_count = threshold * n_neurons
        mask = summed_coo.data >= min_count
        adj = sp.csr_matrix(
            (np.ones(mask.sum()), (summed_coo.row[mask], summed_coo.col[mask])),
            shape=(min_n, min_n))
        return RecurrenceGraph.from_adjacency(adj)

    for thr in jrp_thresholds:
        pop = _jrp_from_sum(thr)
        rqa = pop.rqa()
        jrp_stats.append(rqa)
        log(f'  {thr:8.3f} {pop.adj.nnz:10d} {rqa["RR"]:10.6f} '
            f'{rqa["DET"]:8.3f} {rqa["LAM"]:8.3f} {rqa["ENTR"]:8.3f}')
        if abs(thr - args.jrp_threshold) < 0.001:
            jrp_at_chosen = pop

    if jrp_at_chosen is None:
        jrp_at_chosen = _jrp_from_sum(args.jrp_threshold)

    dt_jrp = time.time() - t4
    log(f'  Time: {dt_jrp:.1f}s')

    # Fig 4: JRP at chosen threshold
    jrp_rqa = jrp_at_chosen.rqa()
    log(f'\n  JRP RQA (threshold={args.jrp_threshold}):')
    for m in measures:
        log(f'    {m}: {jrp_rqa[m]:.4f}')

    if jrp_at_chosen.adj.nnz > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_recurrence(jrp_at_chosen, ax=ax, markersize=0.3)
        ax.set_title(f'Joint Recurrence Plot (threshold={args.jrp_threshold}, '
                     f'{n_neurons} neurons, nnz={jrp_at_chosen.adj.nnz})')
        savefig(fig, 'fig04_jrp_recurrence_plot.png', output_dir)
    else:
        log(f'  JRP at threshold={args.jrp_threshold} is empty, skipping plot.')

    # ───────────────────────────────────────────────────────────────────
    # Network of Networks (Jaccard Similarity)
    # ───────────────────────────────────────────────────────────────────
    log(f'\n--- Network of Networks (Jaccard Similarity) ---')
    t5 = time.time()

    n_pairs = n_neurons * (n_neurons - 1) // 2
    sim_matrix, _mask = pairwise_jaccard_sparse([rg.adj for rg in trimmed])

    dt_jac = time.time() - t5
    upper = sim_matrix[np.triu_indices(n_neurons, k=1)]
    log(f'  Pairs: {n_pairs}')
    log(f'  Jaccard: mean={upper.mean():.4f}, std={upper.std():.4f}, '
        f'max={upper.max():.4f}')
    log(f'  Time: {dt_jac:.1f}s')

    # ───────────────────────────────────────────────────────────────────
    # NoN Spectral Analysis
    # ───────────────────────────────────────────────────────────────────
    log(f'\n--- NoN Spectral Analysis ---')
    t6 = time.time()

    thr_97 = np.percentile(upper, 97)
    log(f'  Similarity threshold (97th pct): {thr_97:.4f}')

    sim_thresholded = sim_matrix.copy()
    sim_thresholded[sim_thresholded < thr_97] = 0
    np.fill_diagonal(sim_thresholded, 0)
    adj_non = sp.csr_matrix(sim_thresholded)

    net = Network(adj=adj_non, preprocessing='giant_cc', create_nx_graph=True)
    log(f'  NoN: {net.n} nodes (giant CC), {net.adj.nnz} edges')

    # Louvain community detection for reordering
    import networkx.algorithms.community as nx_comm
    communities = nx_comm.louvain_communities(net.graph, weight='weight', seed=args.seed)
    communities = sorted(communities, key=len, reverse=True)
    module_assignment = {}
    for mod_idx, comm in enumerate(communities):
        for node in comm:
            module_assignment[node] = mod_idx
    n_modules = len(communities)
    log(f'  Louvain: {n_modules} communities '
        f'(sizes: {", ".join(str(len(c)) for c in communities)})')

    # Fig 5: Similarity matrix reordered by community
    node_order = []
    for comm in communities:
        node_order.extend(sorted(comm))
    reorder_idx = np.array(node_order)
    sim_reordered = sim_matrix[np.ix_(reorder_idx, reorder_idx)]
    np.fill_diagonal(sim_reordered, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_reordered, cmap='hot', aspect='auto',
                   vmin=0, vmax=np.percentile(upper, 99))
    cumsum = np.cumsum([0] + [len(c) for c in communities])
    for boundary in cumsum[1:-1]:
        ax.axhline(boundary - 0.5, color='cyan', linewidth=1, alpha=0.8)
        ax.axvline(boundary - 0.5, color='cyan', linewidth=1, alpha=0.8)
    ax.set_xlabel('Neuron (reordered)')
    ax.set_ylabel('Neuron (reordered)')
    ax.set_title(f'Jaccard Similarity ({n_neurons} neurons, {n_modules} communities)')
    plt.colorbar(im, ax=ax, label='Jaccard index')
    savefig(fig, 'fig05_similarity_matrix.png', output_dir)

    net_rand = net.randomize(rmode='adj_iom')

    from sklearn.cluster import SpectralClustering
    nx_modularity = nx_comm.modularity

    cluster_range = range(2, 11)
    mod_real = []
    mod_rand = []

    for nc in cluster_range:
        sc = SpectralClustering(n_clusters=nc, affinity='precomputed',
                                random_state=args.seed, n_init=10)

        nodes_real = list(net.graph.nodes())
        aff_real = net.adj.toarray().astype(float)
        np.fill_diagonal(aff_real, 0)
        labels = sc.fit_predict(aff_real)
        comms = [set(nodes_real[i] for i in np.where(labels == c)[0]) for c in range(nc)]
        mod_real.append(nx_modularity(net.graph, comms))

        nodes_rand = list(net_rand.graph.nodes())
        aff_rand = net_rand.adj.toarray().astype(float)
        np.fill_diagonal(aff_rand, 0)
        labels_r = sc.fit_predict(aff_rand)
        comms_r = [set(nodes_rand[i] for i in np.where(labels_r == c)[0]) for c in range(nc)]
        mod_rand.append(nx_modularity(net_rand.graph, comms_r))

    log(f'  Modularity (n=5): real={mod_real[3]:.3f}, shuffled={mod_rand[3]:.3f}')

    # Fig 6: Modularity vs n_clusters
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(cluster_range), mod_real, 'b-o', label='Real')
    ax.plot(list(cluster_range), mod_rand, 'r--o', label='Shuffled')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Modularity')
    ax.set_title(f'Modularity -- Network of Networks ({net.n} nodes)')
    ax.legend()
    ax.set_ylim(bottom=0)
    savefig(fig, 'fig06_modularity.png', output_dir)

    # Thermodynamic entropy + specific heat
    tvals = np.logspace(0, 3, 400)
    entropy_real = np.array(net.calculate_thermodynamic_entropy(tvals, norm=True))
    entropy_rand = np.array(net_rand.calculate_thermodynamic_entropy(tvals, norm=True))

    dt_vals = np.diff(np.log(tvals))
    dS_real = np.diff(entropy_real)
    dS_rand = np.diff(entropy_rand)
    t_mid = tvals[:-1] * np.exp(dt_vals / 2)
    C_real = -t_mid * dS_real / np.diff(tvals)
    C_rand = -t_mid * dS_rand / np.diff(tvals)

    C_real_max = np.max(np.abs(C_real)) if np.max(np.abs(C_real)) > 0 else 1.0
    C_rand_max = np.max(np.abs(C_rand)) if np.max(np.abs(C_rand)) > 0 else 1.0

    dt_non = time.time() - t6
    log(f'  Time: {dt_non:.1f}s')

    # Fig 7: Specific heat
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(t_mid, C_real / C_real_max, 'b-', label='Real', linewidth=1.5)
    ax.semilogx(t_mid, C_rand / C_rand_max, 'r--', label='Shuffled', linewidth=1.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Specific heat (normalized)')
    ax.set_title(f'Specific Heat -- Network of Networks ({net.n} nodes)')
    ax.legend()
    savefig(fig, 'fig07_specific_heat.png', output_dir)

    # ───────────────────────────────────────────────────────────────────
    # Save population results
    # ───────────────────────────────────────────────────────────────────
    save_population_results(output_dir, sim_matrix, mod_real, mod_rand,
                            cluster_range, jrp_thresholds, jrp_stats)
    log(f'\n  Saved population_results.npz')

    # ───────────────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    log(f'\n{"="*60}')
    log(f'  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)')
    log(f'{"="*60}')

    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f'\nSummary saved to {summary_path}')
    print(f'Figures saved to {output_dir}/')


if __name__ == '__main__':
    main()
