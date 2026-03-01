#!/usr/bin/env python
"""
Recurrence analysis validation on real FOF (Familiar Open Field) calcium imaging data.

Full pipeline: tau estimation → embedding dim → per-neuron RecurrenceGraphs → RQA →
population JRP → Network of Networks (Jaccard) → spectral analysis.

Usage:
    python tools/recurrence_fof_test.py --session FOF_F36_1D --ds 5
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
from driada.recurrence.population import population_recurrence_graph, pairwise_jaccard_sparse
from driada.recurrence.plotting import plot_recurrence
from driada.information.info_base import get_tdmi
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
    p.add_argument('--exemplar', type=int, default=50,
                   help='Exemplar neuron index for deep dive (default: 50)')
    p.add_argument('--output-dir', default='recurrence_fof_output',
                   help='Output directory for figures (default: recurrence_fof_output)')
    p.add_argument('--jrp-threshold', type=float, default=0.5,
                   help='JRP threshold for main recurrence plot (default: 0.5)')
    p.add_argument('--n-jrp', type=int, default=10,
                   help='Number of neurons for JRP subset (default: 10)')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='Parallel jobs; -1 = all cores (default: -1)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def savefig(fig, name, output_dir):
    path = Path(output_dir) / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def circular_shift_shuffle(adj, rng):
    """Shuffle a sparse adjacency matrix by circular-shifting each row."""
    n = adj.shape[0]
    adj_csr = adj.tocsr()
    rows, cols, data = [], [], []
    for i in range(n):
        row_start = adj_csr.indptr[i]
        row_end = adj_csr.indptr[i + 1]
        if row_start == row_end:
            continue
        row_cols = adj_csr.indices[row_start:row_end].copy()
        shift = rng.integers(1, n)
        shifted = (row_cols + shift) % n
        rows.extend([i] * len(shifted))
        cols.extend(shifted.tolist())
        data.extend([1] * len(shifted))
    shuffled = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Make symmetric
    shuffled = ((shuffled + shuffled.T) > 0).astype(float)
    return shuffled


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
    # Section 2: Load Data
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

    # ───────────────────────────────────────────────────────────────────
    # Section 3+4: Per-Neuron Tau + Dim + RecurrenceGraph (parallel)
    # ───────────────────────────────────────────────────────────────────
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
    log(f'  Neurons: {n_neurons}, k={args.k}')
    log(f'  Tau:  mean={taus.mean():.1f}, std={taus.std():.1f}, '
        f'range=[{taus.min()}, {taus.max()}]')
    log(f'  Dim:  mean={dims.mean():.1f}, std={dims.std():.1f}, '
        f'range=[{dims.min()}, {dims.max()}]')
    log(f'  Sizes: min={min_n}, max={max_n} -> trimmed to {min_n}')
    log(f'  Time: {dt_build:.1f}s')

    # Figure 1: Tau and dim histograms
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

    # ───────────────────────────────────────────────────────────────────
    # Section 5: Per-Neuron RQA + Summary
    # ───────────────────────────────────────────────────────────────────
    log(f'\n--- RQA Analysis ---')
    t3 = time.time()

    with parallel_executor(args.n_jobs, backend='threading') as par:
        rqa_results = par(delayed(lambda rg: rg.rqa())(rg) for rg in trimmed)

    measures = ['RR', 'DET', 'LAM', 'L_mean', 'L_max', 'TT', 'ENTR']
    rqa_arrays = {}
    for m in measures:
        rqa_arrays[m] = np.array([r[m] for r in rqa_results])

    dt_rqa = time.time() - t3
    log(f'\n  RQA Summary (n={n_neurons} neurons):')
    log(f'  {"Measure":<10} {"Mean":>8} {"Std":>8} {"Median":>8} {"Min":>8} {"Max":>8}')
    log(f'  {"-"*50}')
    for m in measures:
        v = rqa_arrays[m]
        log(f'  {m:<10} {v.mean():8.3f} {v.std():8.3f} {np.median(v):8.3f} '
            f'{v.min():8.3f} {v.max():8.3f}')
    log(f'  Time: {dt_rqa:.1f}s')

    # Figure 2: RQA distributions
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
    # Section 6: Exemplar Neuron Deep Dive
    # ───────────────────────────────────────────────────────────────────
    ex = min(args.exemplar, n_neurons - 1)
    log(f'\n--- Exemplar Neuron #{ex} Deep Dive ---')
    log(f'  tau={taus[ex]}, dim={dims[ex]}, DET={rqa_arrays["DET"][ex]:.3f}, '
        f'LAM={rqa_arrays["LAM"][ex]:.3f}')

    rg_ex = trimmed[ex]

    # Fig 3: Recurrence plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_recurrence(rg_ex, ax=ax, markersize=0.3)
    ax.set_title(f'Recurrence Plot — Neuron #{ex} '
                 f'(τ={taus[ex]}, m={dims[ex]}, k={args.k})')
    savefig(fig, 'fig03_recurrence_plot.png', output_dir)

    # Fig 4: TDMI curve
    signal_ex = calcium[ex]
    tdmi = get_tdmi(signal_ex, min_shift=1, max_shift=args.max_shift)
    fig, ax = plt.subplots(figsize=(8, 4))
    lags = np.arange(1, len(tdmi) + 1)
    ax.plot(lags, tdmi, 'b-', linewidth=1.5)
    ax.axvline(taus[ex], color='red', ls='--', label=f'τ = {taus[ex]}')
    ax.set_xlabel('Time lag (samples)')
    ax.set_ylabel('TDMI (bits)')
    ax.set_title(f'TDMI — Neuron #{ex}')
    ax.legend()
    savefig(fig, 'fig04_tdmi_curve.png', output_dir)

    # Fig 5: Spectrum comparison (real vs shuffle)
    spectrum_real = rg_ex.get_spectrum('adj')
    adj_shuffled = circular_shift_shuffle(rg_ex.adj, rng)
    rg_shuffled = RecurrenceGraph.from_adjacency(adj_shuffled)
    spectrum_shuffled = rg_shuffled.get_spectrum('adj')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(spectrum_real, bins=50, alpha=0.6, density=True, label='Real')
    ax.hist(spectrum_shuffled, bins=50, alpha=0.6, density=True, label='Shuffled')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title(f'Adjacency Spectrum — Neuron #{ex}')
    ax.legend()
    savefig(fig, 'fig05_spectrum_comparison.png', output_dir)

    # Fig 6: Degree distribution
    deg_real = np.asarray(rg_ex.adj.sum(axis=1)).ravel()
    deg_shuffled = np.asarray(adj_shuffled.sum(axis=1)).ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deg_real, bins=30, alpha=0.6, label='Real')
    ax.hist(deg_shuffled, bins=30, alpha=0.6, label='Shuffled')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title(f'Degree Distribution — Neuron #{ex}')
    ax.legend()
    savefig(fig, 'fig06_degree_comparison.png', output_dir)

    # ───────────────────────────────────────────────────────────────────
    # Section 7: Population JRP (random subset)
    # ───────────────────────────────────────────────────────────────────
    n_jrp = min(args.n_jrp, n_neurons)
    jrp_indices = rng.choice(n_neurons, size=n_jrp, replace=False)
    jrp_subset = [trimmed[i] for i in sorted(jrp_indices)]
    log(f'\n--- Population JRP ({n_jrp} of {n_neurons} neurons) ---')
    t4 = time.time()
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    log(f'  {"Threshold":>10} {"nnz":>12} {"RR":>10}')
    log(f'  {"-"*35}')

    jrp_at_chosen = None
    for thr in thresholds:
        pop = population_recurrence_graph(jrp_subset, method='joint', threshold=thr)
        n_pop = pop.adj.shape[0]
        rr = pop.adj.nnz / (n_pop * n_pop) if n_pop > 0 else 0
        log(f'  {thr:10.2f} {pop.adj.nnz:12d} {rr:10.6f}')
        if abs(thr - args.jrp_threshold) < 0.01:
            jrp_at_chosen = pop

    dt_jrp = time.time() - t4
    log(f'  Time: {dt_jrp:.1f}s')

    # Fig 7: JRP recurrence plot
    if jrp_at_chosen is not None and jrp_at_chosen.adj.nnz > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_recurrence(jrp_at_chosen, ax=ax, markersize=0.3)
        ax.set_title(f'Joint Recurrence Plot (threshold={args.jrp_threshold}, '
                     f'{n_jrp} neurons)')
        savefig(fig, 'fig07_jrp_recurrence_plot.png', output_dir)

        jrp_rqa = jrp_at_chosen.rqa()
        log(f'\n  JRP RQA (threshold={args.jrp_threshold}):')
        for m in measures:
            log(f'    {m}: {jrp_rqa[m]:.4f}')
    else:
        log(f'  JRP at threshold={args.jrp_threshold} is empty, skipping plot.')

    # ───────────────────────────────────────────────────────────────────
    # Section 8: Network of Networks (Jaccard Similarity)
    # ───────────────────────────────────────────────────────────────────
    log(f'\n--- Network of Networks (Jaccard Similarity) ---')
    t5 = time.time()

    n_pairs = n_neurons * (n_neurons - 1) // 2
    sim_matrix = pairwise_jaccard_sparse([rg.adj for rg in trimmed])

    dt_jac = time.time() - t5
    upper = sim_matrix[np.triu_indices(n_neurons, k=1)]
    log(f'  Pairs: {n_pairs}')
    log(f'  Jaccard: mean={upper.mean():.4f}, std={upper.std():.4f}, '
        f'max={upper.max():.4f}')
    log(f'  Time: {dt_jac:.1f}s')

    # Fig 8: Similarity matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='hot', aspect='auto')
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Jaccard Similarity ({n_neurons} neurons)')
    plt.colorbar(im, ax=ax, label='Jaccard index')
    savefig(fig, 'fig08_similarity_matrix.png', output_dir)

    # ───────────────────────────────────────────────────────────────────
    # Section 9: NoN Spectral Analysis
    # ───────────────────────────────────────────────────────────────────
    log(f'\n--- NoN Spectral Analysis ---')
    t6 = time.time()

    # Threshold at ~97th percentile
    thr_97 = np.percentile(upper, 97)
    log(f'  Similarity threshold (97th pct): {thr_97:.4f}')

    sim_thresholded = sim_matrix.copy()
    sim_thresholded[sim_thresholded < thr_97] = 0
    np.fill_diagonal(sim_thresholded, 0)
    adj_non = sp.csr_matrix(sim_thresholded)

    net = Network(adj=adj_non, preprocessing='giant_cc', create_nx_graph=True)
    log(f'  NoN: {net.n} nodes (giant CC), {net.adj.nnz} edges')

    # Randomized null model
    net_rand = net.randomize(rmode='adj_iom')

    # Modularity vs n_clusters
    from sklearn.cluster import SpectralClustering
    from networkx.algorithms.community import modularity as nx_modularity

    cluster_range = range(2, 11)
    mod_real = []
    mod_rand = []

    for nc in cluster_range:
        sc = SpectralClustering(n_clusters=nc, affinity='precomputed',
                                random_state=args.seed, n_init=10)

        # Real network
        nodes_real = list(net.graph.nodes())
        aff_real = net.adj.toarray().astype(float)
        np.fill_diagonal(aff_real, 0)
        labels = sc.fit_predict(aff_real)
        comms = [set(nodes_real[i] for i in np.where(labels == c)[0]) for c in range(nc)]
        mod_real.append(nx_modularity(net.graph, comms))

        # Randomized
        nodes_rand = list(net_rand.graph.nodes())
        aff_rand = net_rand.adj.toarray().astype(float)
        np.fill_diagonal(aff_rand, 0)
        labels_r = sc.fit_predict(aff_rand)
        comms_r = [set(nodes_rand[i] for i in np.where(labels_r == c)[0]) for c in range(nc)]
        mod_rand.append(nx_modularity(net_rand.graph, comms_r))

    log(f'  Modularity (n=5): real={mod_real[3]:.3f}, shuffled={mod_rand[3]:.3f}')

    # Fig 9: Modularity vs n_clusters
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(cluster_range), mod_real, 'b-o', label='Real')
    ax.plot(list(cluster_range), mod_rand, 'r--o', label='Shuffled')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Modularity')
    ax.set_title(f'Modularity — Network of Networks ({net.n} nodes)')
    ax.legend()
    ax.set_ylim(bottom=0)
    savefig(fig, 'fig09_modularity.png', output_dir)

    # Thermodynamic entropy + specific heat
    tvals = np.logspace(0, 3, 400)
    entropy_real = np.array(net.calculate_thermodynamic_entropy(tvals, norm=True))
    entropy_rand = np.array(net_rand.calculate_thermodynamic_entropy(tvals, norm=True))

    # Specific heat: C(t) = -t * dS/dt
    dt_vals = np.diff(np.log(tvals))
    dS_real = np.diff(entropy_real)
    dS_rand = np.diff(entropy_rand)
    t_mid = tvals[:-1] * np.exp(dt_vals / 2)
    C_real = -t_mid * dS_real / np.diff(tvals)
    C_rand = -t_mid * dS_rand / np.diff(tvals)

    # Normalize
    C_real_max = np.max(np.abs(C_real)) if np.max(np.abs(C_real)) > 0 else 1.0
    C_rand_max = np.max(np.abs(C_rand)) if np.max(np.abs(C_rand)) > 0 else 1.0

    dt_non = time.time() - t6
    log(f'  Time: {dt_non:.1f}s')

    # Fig 10: Specific heat
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(t_mid, C_real / C_real_max, 'b-', label='Real', linewidth=1.5)
    ax.semilogx(t_mid, C_rand / C_rand_max, 'r--', label='Shuffled', linewidth=1.5)
    ax.set_xlabel('Temperature (β)')
    ax.set_ylabel('Specific heat (normalized)')
    ax.set_title(f'Specific Heat — Network of Networks ({net.n} nodes)')
    ax.legend()
    savefig(fig, 'fig10_specific_heat.png', output_dir)

    # ───────────────────────────────────────────────────────────────────
    # Section 10: Summary
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
