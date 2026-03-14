#!/usr/bin/env python
"""
Generate Notebook 06: Recurrence analysis
==========================================

Assembles a Colab-ready Jupyter notebook covering recurrence analysis
fundamentals on classic signals (sine, Lorenz, noise), then applies the
workflow to recover functional modules in a synthetic neural population.
"""

import os
import nbformat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md_cell(source):
    """Create a markdown cell."""
    return nbformat.v4.new_markdown_cell(source.strip())


def code_cell(source):
    """Create a code cell."""
    return nbformat.v4.new_code_cell(source.strip())


# ---------------------------------------------------------------------------
# Build cells
# ---------------------------------------------------------------------------

cells = []

# ===== HEADER + SETUP =====================================================

cells.append(md_cell(
"# Recurrence analysis\n"
"\n"
"Recurrence analysis maps temporal dynamics to graphs, revealing\n"
"structure invisible to linear methods. This notebook covers\n"
"fundamentals on classic signals, then applies the workflow to recover\n"
"functional modules in a synthetic neural population.\n"
"\n"
"| Step | Notebook | What it does |\n"
"|---|---|---|\n"
"| **Overview** | [00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/dev/notebooks/00_driada_overview.ipynb) | Core data structures, quick tour of INTENSE, DR, networks |\n"
"| Neuron analysis | [01 -- Neuron analysis](https://colab.research.google.com/github/iabs-neuro/driada/blob/dev/notebooks/01_data_loading_and_neurons.ipynb) | Spike reconstruction, kinetics optimization, quality metrics, surrogates |\n"
"| Single-neuron selectivity | [02 -- INTENSE](https://colab.research.google.com/github/iabs-neuro/driada/blob/dev/notebooks/02_selectivity_detection_intense.ipynb) | Detect which neurons encode which behavioral variables |\n"
"| Population geometry | [03 -- Dimensionality reduction](https://colab.research.google.com/github/iabs-neuro/driada/blob/dev/notebooks/03_population_geometry_dr.ipynb) | Extract low-dimensional manifolds from population activity |\n"
"| Network analysis | [04 -- Network analysis](https://colab.research.google.com/github/iabs-neuro/driada/blob/dev/notebooks/04_network_analysis.ipynb) | Build and analyze interaction graphs |\n"
"| Putting it together | [05 -- Advanced](https://colab.research.google.com/github/iabs-neuro/driada/blob/dev/notebooks/05_advanced_capabilities.ipynb) | Combine INTENSE + DR, leave-one-out importance, RSA, RNN analysis |\n"
"| **Recurrence analysis** | **06 -- this notebook** | Recurrence graphs, RQA, graph representations, population modules |\n"
"\n"
"**In this notebook:**\n"
"\n"
"1. **Recurrence fundamentals** -- generate classic signals, select\n"
"   embedding parameters, build recurrence plots, compute RQA measures,\n"
"   compare three graph representations (RG, HVG, OPN), and track regime\n"
"   changes with windowed RQA.\n"
"2. **Population analysis** -- generate a synthetic modular population,\n"
"   build per-neuron recurrence graphs, measure pairwise Jaccard\n"
"   similarity, detect communities, and compare to ground truth."
))

cells.append(code_cell(
"# TODO: revert to '!pip install -q driada' after v1.0.0 PyPI release\n"
"# TODO: revert @dev to @main after merging recurrence module to main\n"
"!pip install -q git+https://github.com/iabs-neuro/driada.git@dev\n"
"%matplotlib inline\n"
"\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import scipy.sparse as sp\n"
"import networkx as nx\n"
"import networkx.algorithms.community as nx_comm\n"
"from sklearn.metrics import adjusted_rand_score\n"
"from scipy.spatial import cKDTree\n"
"from scipy.stats import rankdata\n"
"from mpl_toolkits.mplot3d import Axes3D  # noqa: F401\n"
"\n"
"from driada.recurrence import (\n"
"    takens_embedding, estimate_tau, estimate_embedding_dim,\n"
"    RecurrenceGraph, plot_recurrence, compute_rqa,\n"
"    pairwise_jaccard_sparse,\n"
")\n"
"from driada.information import TimeSeries\n"
"from driada.information.info_base import get_tdmi\n"
"from driada.experiment.synthetic import generate_tuned_selectivity_exp\n"
"from driada.network import Network"
))

# ===== SECTION 1: RECURRENCE FUNDAMENTALS =================================

cells.append(md_cell(
"## 1. Recurrence fundamentals"
))

# --- 1.1 Generate signals ---

cells.append(md_cell(
"### 1.1 Generate signals\n"
"\n"
"Three classic signals -- periodic (sine), chaotic (Lorenz), stochastic\n"
"(noise). Standard testbed for recurrence analysis."
))

cells.append(code_cell(
"def _lorenz_series(n, dt=0.02, seed=42):\n"
'    """Generate x-component of Lorenz attractor via RK4 integration."""\n'
"    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0\n"
"    rng = np.random.default_rng(seed)\n"
"\n"
"    def lorenz(state):\n"
"        x, y, z = state\n"
"        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])\n"
"\n"
"    state = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)\n"
"\n"
"    # Warm-up: discard transient\n"
"    warmup = 2000\n"
"    for _ in range(warmup):\n"
"        k1 = lorenz(state)\n"
"        k2 = lorenz(state + dt / 2 * k1)\n"
"        k3 = lorenz(state + dt / 2 * k2)\n"
"        k4 = lorenz(state + dt * k3)\n"
"        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)\n"
"\n"
"    # Collect samples\n"
"    x_series = np.empty(n)\n"
"    for i in range(n):\n"
"        k1 = lorenz(state)\n"
"        k2 = lorenz(state + dt / 2 * k1)\n"
"        k3 = lorenz(state + dt / 2 * k2)\n"
"        k4 = lorenz(state + dt * k3)\n"
"        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)\n"
"        x_series[i] = state[0]\n"
"\n"
"    return x_series\n"
"\n"
"\n"
"rng = np.random.default_rng(42)\n"
"N = 800\n"
"\n"
"t = np.arange(N)\n"
"sine = np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.05, N)\n"
"lorenz_x = _lorenz_series(N, dt=0.02, seed=42)\n"
"noise = rng.normal(size=N)\n"
"\n"
"signals = {\n"
'    "Sine (periodic)": sine,\n'
'    "Lorenz (chaotic)": lorenz_x,\n'
'    "Noise (stochastic)": noise,\n'
"}\n"
"signal_colors = {\n"
'    "Sine (periodic)": "#1f77b4",\n'
'    "Lorenz (chaotic)": "#d62728",\n'
'    "Noise (stochastic)": "#7f7f7f",\n'
"}\n"
"\n"
"print(f\"Generated {N} points for each signal:\")\n"
"for name in signals:\n"
"    sig = signals[name]\n"
"    print(f\"  {name}: range [{sig.min():.2f}, {sig.max():.2f}]\")"
))

# --- 1.2 Embedding parameter selection ---

cells.append(md_cell(
"### 1.2 Embedding parameter selection\n"
"\n"
"Takens' theorem: reconstruct the attractor from a 1D observable using\n"
"time-delay embedding. Two parameters are needed:\n"
"\n"
"- **tau** (time delay) -- first minimum of time-delayed mutual\n"
"  information (TDMI)\n"
"- **dim** (embedding dimension) -- where the false nearest neighbours\n"
"  (FNN) fraction drops below 5%"
))

cells.append(code_cell(
"def compute_fnn_fractions(data, tau, max_dim=10, r_tol=10.0, a_tol=2.0):\n"
'    """Compute FNN fraction for each candidate embedding dimension."""\n'
"    data = np.asarray(data, dtype=float).ravel()\n"
"    attractor_size = np.std(data)\n"
"    dist_tol = attractor_size * 1e-8\n"
"    fractions = []\n"
"    for m in range(2, max_dim + 1):\n"
"        emb_m = takens_embedding(data, tau, m).T\n"
"        emb_m1 = takens_embedding(data, tau, m + 1).T\n"
"        n_m1 = emb_m1.shape[0]\n"
"        emb_m_trimmed = emb_m[:n_m1]\n"
"        tree = cKDTree(emb_m_trimmed)\n"
"        dists, indices = tree.query(emb_m_trimmed, k=2)\n"
"        nn_dists_m = dists[:, 1]\n"
"        nn_indices = indices[:, 1]\n"
"        nn_dists_m1 = np.linalg.norm(emb_m1 - emb_m1[nn_indices], axis=1)\n"
"        valid = nn_dists_m > dist_tol\n"
"        if not np.any(valid):\n"
"            fractions.append((m, 0.0))\n"
"            continue\n"
"        ratio = np.zeros(n_m1)\n"
"        ratio[valid] = np.abs(nn_dists_m1[valid] - nn_dists_m[valid]) / nn_dists_m[valid]\n"
"        c1 = ratio > r_tol\n"
"        c2 = (nn_dists_m1 / attractor_size) > a_tol\n"
"        fractions.append((m, np.sum(c1 | c2) / n_m1))\n"
"    return fractions\n"
"\n"
"\n"
"# Estimate parameters for each signal\n"
"params = {}\n"
"tdmi_curves = {}\n"
"fnn_curves = {}\n"
"\n"
"for name, sig in signals.items():\n"
"    tdmi = get_tdmi(sig, min_shift=1, max_shift=81, estimator=\"gcmi\")\n"
"    tdmi_curves[name] = tdmi\n"
"    tau = estimate_tau(sig, max_shift=80)\n"
"    dim = estimate_embedding_dim(sig, tau=tau, max_dim=10)\n"
"    fnn = compute_fnn_fractions(sig, tau, max_dim=10)\n"
"    fnn_curves[name] = fnn\n"
"    params[name] = (tau, dim)\n"
"    print(f\"{name}: tau={tau}, dim={dim}\")\n"
"\n"
"# 2x3 grid: top=TDMI, bottom=FNN\n"
"fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
"names = list(signals.keys())\n"
"\n"
"for col, name in enumerate(names):\n"
"    tau, dim = params[name]\n"
"    color = signal_colors[name]\n"
"\n"
"    # TDMI curve\n"
"    ax = axes[0, col]\n"
"    tdmi = tdmi_curves[name]\n"
"    shifts = np.arange(1, len(tdmi) + 1)\n"
"    ax.plot(shifts, tdmi, color=color, linewidth=1.5)\n"
"    ax.axvline(tau, color=\"k\", linestyle=\"--\", linewidth=1, alpha=0.7,\n"
"               label=f\"tau = {tau}\")\n"
"    ax.set_title(name, fontsize=11)\n"
"    ax.set_xlabel(\"Lag (samples)\")\n"
"    ax.legend(fontsize=9, loc=\"upper right\")\n"
"    if col == 0:\n"
"        ax.set_ylabel(\"TDMI (bits)\")\n"
"\n"
"    # FNN curve\n"
"    ax = axes[1, col]\n"
"    fnn = fnn_curves[name]\n"
"    fnn_dims = [f[0] for f in fnn]\n"
"    fnn_fracs = [f[1] for f in fnn]\n"
"    ax.plot(fnn_dims, fnn_fracs, \"o-\", color=color, linewidth=1.5, markersize=5)\n"
"    ax.axhline(0.05, color=\"k\", linestyle=\"--\", linewidth=1, alpha=0.7,\n"
"               label=\"5% threshold\")\n"
"    ax.axvline(dim, color=\"k\", linestyle=\":\", linewidth=1, alpha=0.5,\n"
"               label=f\"dim = {dim}\")\n"
"    ax.set_xlabel(\"Embedding dimension\")\n"
"    ax.set_ylim(-0.02, max(0.3, max(fnn_fracs) * 1.15) if fnn_fracs else 0.3)\n"
"    ax.legend(fontsize=9, loc=\"upper right\")\n"
"    if col == 0:\n"
"        ax.set_ylabel(\"FNN fraction\")\n"
"\n"
"fig.suptitle(\"Embedding parameter selection: TDMI and FNN diagnostics\",\n"
"             fontsize=13, y=0.98)\n"
"fig.tight_layout(rect=[0, 0, 1, 0.95])\n"
"plt.show()"
))

# --- 1.3 3D delay embedding ---

cells.append(md_cell(
"### 1.3 3D delay embedding\n"
"\n"
"Delay-embedded Lorenz attractor recovers the butterfly shape from a\n"
"scalar observable -- Takens' theorem in action."
))

cells.append(code_cell(
"tau_lorenz, dim_lorenz = params[\"Lorenz (chaotic)\"]\n"
"emb = takens_embedding(lorenz_x, tau_lorenz, max(dim_lorenz, 3))\n"
"n_emb = emb.shape[1]\n"
"\n"
"fig = plt.figure(figsize=(9, 7))\n"
"ax = fig.add_subplot(111, projection=\"3d\")\n"
"\n"
"xs, ys, zs = emb[0], emb[1], emb[2]\n"
"time_color = np.arange(n_emb)\n"
"\n"
"sc = ax.scatter(xs, ys, zs, c=time_color, cmap=\"inferno\", s=2, alpha=0.7)\n"
"cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)\n"
"cbar.set_label(\"Time index\", fontsize=10)\n"
"\n"
"ax.set_xlabel(r\"$x(t)$\", fontsize=10)\n"
"ax.set_ylabel(r\"$x(t + \\tau)$\", fontsize=10)\n"
"ax.set_zlabel(r\"$x(t + 2\\tau)$\", fontsize=10)\n"
"ax.set_title(f\"3D delay embedding of Lorenz attractor \"\n"
"             f\"(tau={tau_lorenz}, dim={max(dim_lorenz, 3)})\", fontsize=12)\n"
"ax.view_init(elev=25, azim=135)\n"
"\n"
"fig.tight_layout()\n"
"plt.show()"
))

# --- 1.4 Recurrence plots ---

cells.append(md_cell(
"### 1.4 Recurrence plots\n"
"\n"
"A recurrence plot (RP) is a binary matrix where a dot at (i, j) means\n"
"states at times i and j are close in embedding space.\n"
"\n"
"- **Diagonal lines** = determinism\n"
"- **Vertical lines** = laminarity\n"
"- **Uniform scatter** = noise (no temporal structure)"
))

cells.append(code_cell(
"# Build k-NN recurrence graphs (k=5) for all three signals\n"
"graphs = {}\n"
"for name, sig in signals.items():\n"
"    tau, dim = params[name]\n"
"    emb = takens_embedding(sig, tau=tau, m=dim)\n"
"    theiler = tau * (dim - 1) + 1\n"
"    rg = RecurrenceGraph(emb, method=\"knn\", k=5, theiler_window=theiler)\n"
"    graphs[name] = rg\n"
"    print(f\"{name}: {rg.n} embedded points, {rg.adj.nnz} recurrence entries\")\n"
"\n"
"# 2x3 grid: top=time series, bottom=recurrence plots\n"
"names = list(signals.keys())\n"
"fig, axes = plt.subplots(2, 3, figsize=(15, 8),\n"
"                          gridspec_kw={\"height_ratios\": [1, 3]})\n"
"\n"
"for col, name in enumerate(names):\n"
"    sig = signals[name]\n"
"    rg = graphs[name]\n"
"    color = signal_colors[name]\n"
"    n_emb = rg.n\n"
"\n"
"    # Time series (top)\n"
"    ax_ts = axes[0, col]\n"
"    ax_ts.plot(np.arange(n_emb), sig[:n_emb], color=color, linewidth=0.6)\n"
"    ax_ts.set_title(name, fontsize=11)\n"
"    ax_ts.set_xlim(0, n_emb - 1)\n"
"    ax_ts.tick_params(labelbottom=False)\n"
"    if col == 0:\n"
"        ax_ts.set_ylabel(\"Amplitude\")\n"
"\n"
"    # Recurrence plot (bottom)\n"
"    ax_rp = axes[1, col]\n"
"    plot_recurrence(rg, ax=ax_rp, markersize=0.3, color=\"k\")\n"
"    ax_rp.set_title(\"\")\n"
"    ax_rp.set_xlim(0, n_emb - 1)\n"
"    ax_rp.set_ylim(n_emb - 1, 0)\n"
"    ax_rp.set_xlabel(\"Time index\")\n"
"    if col == 0:\n"
"        ax_rp.set_ylabel(\"Time index\")\n"
"\n"
"fig.suptitle(\"Recurrence plots with marginal time series\",\n"
"             fontsize=13, y=0.98)\n"
"fig.tight_layout(rect=[0, 0, 1, 0.95])\n"
"plt.show()"
))

# --- 1.5 RQA comparison ---

cells.append(md_cell(
"### 1.5 RQA comparison\n"
"\n"
"Recurrence Quantification Analysis (RQA) extracts diagonal and vertical\n"
"line structures from the RP:\n"
"\n"
"| Measure | Definition | Interpretation |\n"
"|---|---|---|\n"
"| DET | Fraction of recurrence in diagonal lines (l >= 2) | Determinism |\n"
"| LAM | Fraction in vertical lines (v >= 2) | Laminarity |\n"
"| ENTR | Shannon entropy of diagonal line lengths | Complexity |"
))

cells.append(code_cell(
"# Compute RQA for each signal\n"
"rqa_data = {}\n"
"measures = [\"DET\", \"LAM\", \"ENTR\"]\n"
"\n"
"print(f\"{'Signal':<24} {'DET':>8} {'LAM':>8} {'ENTR':>8}\")\n"
"print(f\"{'-' * 48}\")\n"
"for name, rg in graphs.items():\n"
"    rqa = rg.rqa()\n"
"    rqa_data[name] = rqa\n"
"    print(f\"{name:<24} {rqa['DET']:>8.4f} {rqa['LAM']:>8.4f} {rqa['ENTR']:>8.4f}\")\n"
"\n"
"# Grouped bar chart\n"
"names = list(graphs.keys())\n"
"n_measures = len(measures)\n"
"n_signals = len(names)\n"
"x = np.arange(n_measures)\n"
"width = 0.8 / n_signals\n"
"\n"
"fig, ax = plt.subplots(figsize=(8, 5))\n"
"for i, name in enumerate(names):\n"
"    vals = [rqa_data[name][m] for m in measures]\n"
"    offset = (i - (n_signals - 1) / 2) * width\n"
"    ax.bar(x + offset, vals, width * 0.9,\n"
"           label=name, color=signal_colors[name], edgecolor=\"white\",\n"
"           linewidth=0.5)\n"
"\n"
"ax.set_xticks(x)\n"
"ax.set_xticklabels(measures, fontsize=11)\n"
"ax.set_ylabel(\"Value\", fontsize=11)\n"
"ax.set_title(\"RQA measures comparison\", fontsize=13)\n"
"ax.legend(fontsize=9)\n"
"ax.grid(axis=\"y\", alpha=0.3)\n"
"fig.tight_layout()\n"
"plt.show()"
))

# --- 1.6 Three graph representations ---

cells.append(md_cell(
"### 1.6 Three graph representations"
))

# 1.6a RecurrenceGraph

cells.append(md_cell(
"#### RecurrenceGraph\n"
"\n"
"RG connects time points whose embedded states are k-nearest neighbours.\n"
"The adjacency matrix IS the recurrence plot. Nodes = time points,\n"
"edges = recurrence."
))

cells.append(code_cell(
"tau_sine, dim_sine = params[\"Sine (periodic)\"]\n"
"\n"
"# Build RG for sine with k=10\n"
"emb_sine = takens_embedding(sine, tau=tau_sine, m=dim_sine)\n"
"theiler_sine = tau_sine * (dim_sine - 1) + 1\n"
"rg_sine = RecurrenceGraph(emb_sine, method=\"knn\", k=10,\n"
"                           theiler_window=theiler_sine)\n"
"\n"
"# Convert to networkx and visualize\n"
"G_rg = nx.from_scipy_sparse_array(rg_sine.adj)\n"
"# Take giant connected component\n"
"gcc = max(nx.connected_components(G_rg), key=len)\n"
"G_rg = G_rg.subgraph(gcc).copy()\n"
"\n"
"pos_rg = nx.spring_layout(G_rg, seed=42, iterations=80)\n"
"node_list = sorted(G_rg.nodes())\n"
"node_vals = rankdata(sine[node_list], method=\"average\") / len(node_list)\n"
"\n"
"fig, ax = plt.subplots(figsize=(8, 6))\n"
"nx.draw_networkx_edges(G_rg, pos_rg, alpha=0.1, ax=ax)\n"
"sc = nx.draw_networkx_nodes(G_rg, pos_rg, nodelist=node_list,\n"
"                             node_color=node_vals, cmap=\"inferno\",\n"
"                             node_size=15, ax=ax)\n"
"ax.set_title(\"RecurrenceGraph (k=10, sine)\", fontsize=12)\n"
"ax.axis(\"off\")\n"
"plt.colorbar(sc, ax=ax, label=\"Signal percentile\", shrink=0.8)\n"
"plt.tight_layout()\n"
"plt.show()"
))

# 1.6b HorizontalVisibilityGraph

cells.append(md_cell(
"#### HorizontalVisibilityGraph\n"
"\n"
"Two time points are connected if a horizontal line-of-sight is\n"
"unobstructed (no intermediate value exceeds both endpoints). The degree\n"
"distribution differentiates periodic, chaotic, and random signals\n"
"(Lacasa et al. 2008). O(N) construction via monotone stack."
))

cells.append(code_cell(
"ts_sine = TimeSeries(sine)\n"
"vg = ts_sine.visibility_graph(method=\"horizontal\")\n"
"\n"
"G_vg = nx.from_scipy_sparse_array(vg.adj)\n"
"pos_vg = nx.spring_layout(G_vg, seed=42, iterations=80)\n"
"\n"
"node_list_vg = sorted(G_vg.nodes())\n"
"node_vals_vg = rankdata(sine[node_list_vg], method=\"average\") / len(node_list_vg)\n"
"\n"
"fig, ax = plt.subplots(figsize=(8, 6))\n"
"nx.draw_networkx_edges(G_vg, pos_vg, alpha=0.1, ax=ax)\n"
"sc = nx.draw_networkx_nodes(G_vg, pos_vg, nodelist=node_list_vg,\n"
"                             node_color=node_vals_vg, cmap=\"inferno\",\n"
"                             node_size=15, ax=ax)\n"
"ax.set_title(\"HorizontalVisibilityGraph (sine)\", fontsize=12)\n"
"ax.axis(\"off\")\n"
"plt.colorbar(sc, ax=ax, label=\"Signal percentile\", shrink=0.8)\n"
"plt.tight_layout()\n"
"plt.show()"
))

# 1.6c OrdinalPartitionNetwork

cells.append(md_cell(
"#### OrdinalPartitionNetwork\n"
"\n"
"Delay-embedded windows are reduced to rank patterns (ordinal patterns).\n"
"Directed edges connect consecutive patterns. Permutation entropy\n"
"measures the diversity of visited patterns."
))

cells.append(code_cell(
"opn = ts_sine.ordinal_partition_network(tau=tau_sine)\n"
"\n"
"G_opn = nx.from_scipy_sparse_array(opn.adj)\n"
"pos_opn = nx.spring_layout(G_opn, seed=42, iterations=80)\n"
"\n"
"# Color OPN nodes by mean signal value of time points producing each pattern\n"
"pattern_ids = opn._pattern_ids\n"
"opn_nodes = sorted(G_opn.nodes())\n"
"opn_vals = np.zeros(len(opn_nodes))\n"
"for vi, node in enumerate(opn_nodes):\n"
"    mask = pattern_ids == node\n"
"    if mask.any():\n"
"        opn_vals[vi] = np.mean(sine[:len(pattern_ids)][mask])\n"
"\n"
"fig, ax = plt.subplots(figsize=(8, 6))\n"
"nx.draw_networkx_edges(G_opn, pos_opn, alpha=0.3, ax=ax)\n"
"sc = nx.draw_networkx_nodes(G_opn, pos_opn, nodelist=opn_nodes,\n"
"                             node_color=opn_vals, cmap=\"inferno\",\n"
"                             node_size=80, ax=ax)\n"
"ax.set_title(\"OrdinalPartitionNetwork (sine)\", fontsize=12)\n"
"ax.axis(\"off\")\n"
"plt.colorbar(sc, ax=ax, label=\"Mean signal value\", shrink=0.8)\n"
"\n"
"pe = ts_sine.permutation_entropy(tau=tau_sine)\n"
"print(f\"Permutation entropy: {pe:.4f}\")\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

# --- 1.7 Windowed RQA ---

cells.append(md_cell(
"### 1.7 Windowed RQA\n"
"\n"
"Sliding window over the RP diagonal, computing DET in each window.\n"
"Tracks regime transitions in non-stationary signals."
))

cells.append(code_cell(
"# Concatenate sine + Lorenz + sine (2400 points)\n"
"seg_len = N\n"
"rng2 = np.random.default_rng(99)\n"
"sine_seg1 = np.sin(2 * np.pi * np.arange(seg_len) / 50) + rng.normal(0, 0.05, seg_len)\n"
"lorenz_seg = _lorenz_series(seg_len, dt=0.02, seed=99)\n"
"sine_seg2 = np.sin(2 * np.pi * np.arange(seg_len) / 50) + rng.normal(0, 0.05, seg_len)\n"
"\n"
"# Normalize Lorenz to similar amplitude range\n"
"lorenz_seg_norm = (lorenz_seg - lorenz_seg.mean()) / lorenz_seg.std()\n"
"nonstat_signal = np.concatenate([sine_seg1, lorenz_seg_norm, sine_seg2])\n"
"print(f\"Non-stationary signal: {len(nonstat_signal)} points (sine-Lorenz-sine)\")\n"
"\n"
"# Fixed embedding parameters for mixed signal\n"
"nonstat_tau, nonstat_dim = 12, 5\n"
"nonstat_emb = takens_embedding(nonstat_signal, tau=nonstat_tau, m=nonstat_dim)\n"
"nonstat_theiler = nonstat_tau * (nonstat_dim - 1) + 1\n"
"nonstat_rg = RecurrenceGraph(nonstat_emb, method=\"knn\", k=5,\n"
"                              theiler_window=nonstat_theiler)\n"
"print(f\"Embedded: {nonstat_rg.n} points, {nonstat_rg.adj.nnz} recurrence entries\")\n"
"\n"
"\n"
"def windowed_det(adj_csr, window_size, step):\n"
'    """Compute DET in sliding windows along the RP diagonal."""\n'
"    n = adj_csr.shape[0]\n"
"    positions, values = [], []\n"
"    for start in range(0, n - window_size, step):\n"
"        end = start + window_size\n"
"        sub = adj_csr[start:end, start:end]\n"
"        if sub.nnz > 0:\n"
"            rqa = compute_rqa(sub)\n"
"            values.append(rqa['DET'])\n"
"        else:\n"
"            values.append(0.0)\n"
"        positions.append((start + end) / 2)\n"
"    return np.array(positions), np.array(values)\n"
"\n"
"\n"
"adj_csr = nonstat_rg.adj.tocsr()\n"
"positions, det_values = windowed_det(adj_csr, window_size=150, step=10)\n"
"n_emb = nonstat_rg.n\n"
"\n"
"# Plot: top=signal with colored background, bottom=windowed DET\n"
"fig, (ax_ts, ax_det) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,\n"
"                                     gridspec_kw={\"height_ratios\": [1, 1.2]})\n"
"\n"
"# Top: time series with regime shading\n"
"t_emb = np.arange(n_emb)\n"
"ax_ts.plot(t_emb, nonstat_signal[:n_emb], color=\"#9467bd\", linewidth=0.5)\n"
"\n"
"regime_colors = [\"#1f77b4\", \"#d62728\", \"#1f77b4\"]\n"
"regime_labels = [\"Sine\", \"Lorenz\", \"Sine\"]\n"
"for i in range(3):\n"
"    start = i * seg_len\n"
"    end = min((i + 1) * seg_len, n_emb)\n"
"    if start < n_emb:\n"
"        ax_ts.axvspan(start, end, alpha=0.12, color=regime_colors[i])\n"
"        mid = (start + min(end, n_emb)) / 2\n"
"        ax_ts.text(mid, ax_ts.get_ylim()[0] if i > 0 else 0, regime_labels[i],\n"
"                   ha=\"center\", va=\"bottom\", fontsize=9, color=regime_colors[i],\n"
"                   fontweight=\"bold\")\n"
"\n"
"ax_ts.set_ylabel(\"Amplitude\")\n"
"ax_ts.set_title(\"Non-stationary signal: sine - Lorenz - sine\", fontsize=12)\n"
"\n"
"# Bottom: windowed DET\n"
"ax_det.plot(positions, det_values, color=\"#9467bd\", linewidth=1.5)\n"
"ax_det.fill_between(positions, det_values, alpha=0.2, color=\"#9467bd\")\n"
"for i in range(3):\n"
"    start = i * seg_len\n"
"    end = min((i + 1) * seg_len, n_emb)\n"
"    if start < n_emb:\n"
"        ax_det.axvspan(start, end, alpha=0.08, color=regime_colors[i])\n"
"\n"
"ax_det.set_xlabel(\"Time index\")\n"
"ax_det.set_ylabel(\"Windowed DET\")\n"
"ax_det.set_ylim(-0.02, 1.05)\n"
"ax_det.set_title(\"Windowed determinism (DET) tracks regime changes\", fontsize=12)\n"
"ax_det.grid(axis=\"y\", alpha=0.3)\n"
"\n"
"fig.tight_layout()\n"
"plt.show()"
))

# ===== SECTION 2: RECOVERING FUNCTIONAL MODULES ===========================

cells.append(md_cell(
"## 2. Recovering functional modules from population dynamics\n"
"\n"
"Scientific question: can we identify functional modules from dynamics\n"
"alone? 120 neurons in 6 modules -- same population as notebook 04. No\n"
"behavioral variables are used."
))

# --- 2.1 Generate population ---

cells.append(md_cell(
"### 2.1 Generate population"
))

cells.append(code_cell(
"population = [\n"
"    {\"name\": \"event_0\", \"count\": 30, \"features\": [\"event_0\"]},\n"
"    {\"name\": \"event_1\", \"count\": 30, \"features\": [\"event_1\"]},\n"
"    {\"name\": \"event_2\", \"count\": 30, \"features\": [\"event_2\"]},\n"
"    {\"name\": \"event_0|1\", \"count\": 10, \"features\": [\"event_0\", \"event_1\"],\n"
"     \"combination\": \"or\"},\n"
"    {\"name\": \"event_0|2\", \"count\": 10, \"features\": [\"event_0\", \"event_2\"],\n"
"     \"combination\": \"or\"},\n"
"    {\"name\": \"event_1|2\", \"count\": 10, \"features\": [\"event_1\", \"event_2\"],\n"
"     \"combination\": \"or\"},\n"
"]\n"
"\n"
"print(\"Generating synthetic experiment...\")\n"
"exp = generate_tuned_selectivity_exp(\n"
"    population=population,\n"
"    n_discrete_features=3,\n"
"    duration=600,\n"
"    fps=20.0,\n"
"    baseline_rate=0.05,\n"
"    peak_rate=2.0,\n"
"    decay_time=2.0,\n"
"    calcium_noise=0.02,\n"
"    seed=42,\n"
"    verbose=True,\n"
")\n"
"\n"
"# Downsample calcium to 5 Hz\n"
"ds = 4\n"
"calcium = exp.calcium.data[:, ::ds]\n"
"n_neurons, n_frames = calcium.shape\n"
"fps_eff = 20.0 / ds\n"
"\n"
"print(f\"\\n{n_neurons} neurons, {n_frames} frames ({fps_eff:.0f} Hz)\")\n"
"for g in population:\n"
"    feat_str = \" OR \".join(g[\"features\"])\n"
"    print(f\"  {g['count']:>2} {g['name']:<12} -> {feat_str}\")"
))

# --- 2.2 Per-neuron embedding + RQA ---

cells.append(md_cell(
"### 2.2 Per-neuron embedding and RQA\n"
"\n"
"For each neuron: estimate tau (TDMI), dim (FNN), build k-NN recurrence\n"
"graph (k=50), compute RQA."
))

cells.append(code_cell(
"def get_neuron_modules(population):\n"
'    """Map neuron index to module name."""\n'
"    modules = {}\n"
"    idx = 0\n"
"    for group in population:\n"
"        for _ in range(group[\"count\"]):\n"
"            modules[idx] = group[\"name\"]\n"
"            idx += 1\n"
"    return modules\n"
"\n"
"\n"
"neuron_modules = get_neuron_modules(population)\n"
"groups = {}\n"
"for idx, m in neuron_modules.items():\n"
"    groups.setdefault(m, []).append(idx)\n"
"module_names = sorted(groups.keys())\n"
"\n"
"# Wrap each neuron as TimeSeries\n"
"ts_list = [TimeSeries(calcium[i]) for i in range(n_neurons)]\n"
"\n"
"print(\"Computing per-neuron embedding and RQA...\")\n"
"taus = np.empty(n_neurons, dtype=int)\n"
"dims = np.empty(n_neurons, dtype=int)\n"
"rqa_results = []\n"
"\n"
"for i, ts in enumerate(ts_list):\n"
"    taus[i] = ts.estimate_tau(max_shift=60)\n"
"    dims[i] = ts.estimate_embedding_dim(tau=taus[i], max_dim=15)\n"
"    rqa_results.append(ts.rqa(tau=taus[i], m=dims[i], k=50))\n"
"\n"
"print(f\"\\n{'Module':<15} {'n':>3} {'tau':>6} {'dim':>5}\"\n"
"      f\" {'DET':>7} {'LAM':>7} {'ENTR':>7}\")\n"
"print(f\"{'-' * 55}\")\n"
"\n"
"for m in module_names:\n"
"    idxs = groups[m]\n"
"    det = np.array([rqa_results[i]['DET'] for i in idxs])\n"
"    lam = np.array([rqa_results[i]['LAM'] for i in idxs])\n"
"    entr = np.array([rqa_results[i]['ENTR'] for i in idxs])\n"
"    print(f\"{m:<15} {len(idxs):>3} {taus[idxs].mean():>5.0f}\"\n"
"          f\" {dims[idxs].mean():>5.1f}\"\n"
"          f\" {det.mean():>6.3f} {lam.mean():>6.3f} {entr.mean():>6.3f}\")"
))

# --- 2.3 Pairwise Jaccard ---

cells.append(md_cell(
"### 2.3 Pairwise Jaccard similarity\n"
"\n"
"Jaccard = |intersection| / |union| of binary RG entries. Same-module\n"
"neurons recur together, producing higher Jaccard."
))

cells.append(code_cell(
"# Build recurrence graphs and trim to common size\n"
"print(\"Building per-neuron recurrence graphs...\")\n"
"per_neuron_rgs = []\n"
"for i, ts in enumerate(ts_list):\n"
"    rg = ts.recurrence_graph(tau=taus[i], m=dims[i], k=50)\n"
"    per_neuron_rgs.append(rg)\n"
"\n"
"sizes = [rg.n for rg in per_neuron_rgs]\n"
"min_n = min(sizes)\n"
"trimmed_adjs = []\n"
"for rg in per_neuron_rgs:\n"
"    if rg.n > min_n:\n"
"        trimmed_adjs.append(rg.adj[:min_n, :min_n])\n"
"    else:\n"
"        trimmed_adjs.append(rg.adj)\n"
"print(f\"{n_neurons} graphs, trimmed to {min_n} time points\")\n"
"\n"
"# Pairwise Jaccard\n"
"print(\"Computing pairwise Jaccard similarity...\")\n"
"sim_matrix = pairwise_jaccard_sparse(trimmed_adjs)\n"
"\n"
"# Within-module vs between-module\n"
"within_vals = []\n"
"between_vals = []\n"
"for i in range(n_neurons):\n"
"    for j in range(i + 1, n_neurons):\n"
"        if neuron_modules[i] == neuron_modules[j]:\n"
"            within_vals.append(sim_matrix[i, j])\n"
"        else:\n"
"            between_vals.append(sim_matrix[i, j])\n"
"\n"
"within_vals = np.array(within_vals)\n"
"between_vals = np.array(between_vals)\n"
"ratio = within_vals.mean() / between_vals.mean() if between_vals.mean() > 0 else float(\"inf\")\n"
"\n"
"print(f\"Within-module Jaccard:  {within_vals.mean():.5f}\")\n"
"print(f\"Between-module Jaccard: {between_vals.mean():.5f}\")\n"
"print(f\"Within/between ratio:   {ratio:.2f}x\")"
))

# --- 2.4 Permutation test ---

cells.append(md_cell(
"### 2.4 Permutation test\n"
"\n"
"Shuffle module labels 1000 times, recompute the within/between ratio\n"
"each time to build a null distribution. p-value = fraction of shuffled\n"
"ratios >= observed."
))

cells.append(code_cell(
"pairs_i, pairs_j = np.triu_indices(n_neurons, k=1)\n"
"pair_sims = sim_matrix[pairs_i, pairs_j]\n"
"all_labels_arr = np.array([neuron_modules[i] for i in range(n_neurons)])\n"
"\n"
"observed_ratio = within_vals.mean() / between_vals.mean()\n"
"n_perms = 1000\n"
"perm_ratios = np.empty(n_perms)\n"
"rng_perm = np.random.default_rng(42)\n"
"\n"
"for p in range(n_perms):\n"
"    shuffled = rng_perm.permutation(all_labels_arr)\n"
"    same_mask = shuffled[pairs_i] == shuffled[pairs_j]\n"
"    w_mean = pair_sims[same_mask].mean()\n"
"    b_mean = pair_sims[~same_mask].mean()\n"
"    perm_ratios[p] = w_mean / b_mean if b_mean > 0 else 1.0\n"
"\n"
"perm_p_value = float((perm_ratios >= observed_ratio).mean())\n"
"\n"
"print(f\"Observed within/between ratio: {observed_ratio:.3f}\")\n"
"print(f\"Null distribution: {perm_ratios.mean():.3f} +/- {perm_ratios.std():.3f}\")\n"
"print(f\"p-value: {perm_p_value:.4f} (1000 permutations)\")"
))

# --- 2.5 Community detection ---

cells.append(md_cell(
"### 2.5 Community detection\n"
"\n"
"Threshold Jaccard at 90th percentile, extract giant connected\n"
"component, run Louvain, and compare to ground truth with ARI."
))

cells.append(code_cell(
"# Threshold the Jaccard matrix\n"
"upper = sim_matrix[np.triu_indices(n_neurons, k=1)]\n"
"thr = np.percentile(upper, 90)\n"
"sim_thresholded = sim_matrix.copy()\n"
"sim_thresholded[sim_thresholded < thr] = 0\n"
"np.fill_diagonal(sim_thresholded, 0)\n"
"print(f\"Threshold: {thr:.5f} (90th percentile)\")\n"
"\n"
"# Build network and extract giant CC\n"
"net_adj = sp.csr_matrix(sim_thresholded)\n"
"net = Network(adj=net_adj, preprocessing=\"giant_cc\", create_nx_graph=True)\n"
"print(f\"Network: {net.n} nodes, {net.graph.number_of_edges()} edges\")\n"
"\n"
"# Louvain community detection\n"
"communities = nx_comm.louvain_communities(net.graph, weight=\"weight\", seed=42)\n"
"communities = sorted(communities, key=len, reverse=True)\n"
"communities = [set(c) for c in communities]\n"
"n_detected = len(communities)\n"
"\n"
"# Real modularity\n"
"mod_real = nx_comm.modularity(net.graph, communities, weight=\"weight\")\n"
"\n"
"# Shuffled modularity baseline\n"
"net_rand = net.randomize(rmode=\"adj_iom\")\n"
"comms_rand = nx_comm.louvain_communities(net_rand.graph, weight=\"weight\", seed=42)\n"
"mod_rand = nx_comm.modularity(net_rand.graph, comms_rand, weight=\"weight\")\n"
"\n"
"# ARI\n"
"nodes_in_net = set()\n"
"for comm in communities:\n"
"    nodes_in_net.update(comm)\n"
"\n"
"true_labels = []\n"
"detected_labels = []\n"
"for ci, comm in enumerate(communities):\n"
"    for node in comm:\n"
"        detected_labels.append(ci)\n"
"        true_labels.append(neuron_modules.get(node, \"unknown\"))\n"
"\n"
"mod_to_int = {m: i for i, m in enumerate(module_names)}\n"
"true_int = [mod_to_int.get(t, -1) for t in true_labels]\n"
"ari = adjusted_rand_score(true_int, detected_labels)\n"
"\n"
"print(f\"{n_detected} communities detected\")\n"
"print(f\"ARI = {ari:.3f}\")\n"
"print(f\"Modularity: {mod_real:.3f} (shuffled: {mod_rand:.3f})\")\n"
"\n"
"# Print composition per community\n"
"for ci, comm in enumerate(communities):\n"
"    mod_counts = {}\n"
"    for node in comm:\n"
"        m = neuron_modules.get(node, \"unknown\")\n"
"        mod_counts[m] = mod_counts.get(m, 0) + 1\n"
"    composition = \", \".join(f\"{m}:{c}\" for m, c in sorted(mod_counts.items()))\n"
"    print(f\"  C{ci}: {len(comm):>3} neurons [{composition}]\")"
))

# --- 2.6 Visualizations ---

cells.append(md_cell(
"### 2.6 Visualizations\n"
"\n"
"Three views of results: network graph, Jaccard matrices, and confusion\n"
"matrix."
))

cells.append(code_cell(
"MODULE_COLORS = {\n"
"    \"event_0\": \"#1a5acd\", \"event_1\": \"#ffaa00\", \"event_2\": \"#33cc33\",\n"
"    \"event_0|1\": \"#cc44cc\", \"event_0|2\": \"#00dddd\", \"event_1|2\": \"#ff4444\",\n"
"}\n"
"MODULE_SHORT = {\n"
"    \"event_0\": \"E1\", \"event_1\": \"E2\", \"event_2\": \"E3\",\n"
"    \"event_0|1\": \"E1|E2\", \"event_0|2\": \"E1|E3\", \"event_1|2\": \"E2|E3\",\n"
"}\n"
"\n"
"legend_order = [\"event_0\", \"event_1\", \"event_2\",\n"
"                \"event_0|1\", \"event_0|2\", \"event_1|2\"]\n"
"\n"
"# --- Figure 1: Network graph ---\n"
"from matplotlib.lines import Line2D\n"
"\n"
"fig_net, ax_net = plt.subplots(figsize=(7, 7))\n"
"\n"
"pos_init = nx.spectral_layout(net.graph)\n"
"pos_net = nx.spring_layout(\n"
"    net.graph, pos=pos_init, k=0.02 / np.sqrt(len(net.graph)),\n"
"    iterations=200, seed=42)\n"
"\n"
"nx.draw_networkx_edges(net.graph, pos_net, alpha=0.1, ax=ax_net)\n"
"\n"
"node_list_net = list(net.graph.nodes())\n"
"node_colors_net = [MODULE_COLORS[neuron_modules[n]] for n in node_list_net]\n"
"nx.draw_networkx_nodes(net.graph, pos_net, nodelist=node_list_net,\n"
"                        node_color=node_colors_net, node_size=40,\n"
"                        edgecolors=\"k\", linewidths=0.3, ax=ax_net, alpha=0.85)\n"
"\n"
"handles = [Line2D([0], [0], marker=\"o\", color=\"none\",\n"
"                   markerfacecolor=MODULE_COLORS[g],\n"
"                   markeredgecolor=\"k\", markeredgewidth=0.3,\n"
"                   markersize=7, label=MODULE_SHORT[g])\n"
"           for g in legend_order]\n"
"ax_net.legend(handles=handles, fontsize=9, frameon=False,\n"
"              loc=\"lower center\", bbox_to_anchor=(0.5, -0.06), ncol=3)\n"
"ax_net.set_title(\n"
"    f\"Jaccard similarity network ({net.n} neurons, \"\n"
"    f\"{net.graph.number_of_edges()} edges, ARI={ari:.2f})\",\n"
"    fontsize=12)\n"
"ax_net.axis(\"off\")\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(code_cell(
"# --- Figure 2: Jaccard matrices (1x2 subplots) ---\n"
"fig_sum, axes_sum = plt.subplots(1, 2, figsize=(14, 6))\n"
"\n"
"upper_tri = sim_matrix[np.triu_indices(n_neurons, k=1)]\n"
"vmin_j = np.percentile(upper_tri, 5)\n"
"vmax_j = np.percentile(upper_tri, 99)\n"
"\n"
"# Left: ordered by module\n"
"mod_order = []\n"
"for m in module_names:\n"
"    mod_order.extend(sorted(groups[m]))\n"
"mod_arr = np.array(mod_order)\n"
"sim_by_mod = sim_matrix[np.ix_(mod_arr, mod_arr)]\n"
"np.fill_diagonal(sim_by_mod, 0)\n"
"\n"
"im1 = axes_sum[0].imshow(sim_by_mod, cmap=\"inferno\", aspect=\"auto\",\n"
"                          vmin=vmin_j, vmax=vmax_j)\n"
"cumsum_mod = np.cumsum([0] + [len(groups[m]) for m in module_names])\n"
"for b in cumsum_mod[1:-1]:\n"
"    axes_sum[0].axhline(b - 0.5, color=\"white\", linewidth=0.8, alpha=0.7)\n"
"    axes_sum[0].axvline(b - 0.5, color=\"white\", linewidth=0.8, alpha=0.7)\n"
"axes_sum[0].set_xticks([])\n"
"axes_sum[0].set_yticks([])\n"
"axes_sum[0].set_title(\"Jaccard similarity (ordered by module)\")\n"
"plt.colorbar(im1, ax=axes_sum[0], label=\"Jaccard\", shrink=0.8)\n"
"\n"
"# Right: ordered by community\n"
"comm_order = []\n"
"for comm in communities:\n"
"    comm_order.extend(sorted(comm))\n"
"missing = [i for i in range(n_neurons) if i not in nodes_in_net]\n"
"comm_order.extend(missing)\n"
"comm_arr = np.array(comm_order)\n"
"sim_by_comm = sim_matrix[np.ix_(comm_arr, comm_arr)]\n"
"np.fill_diagonal(sim_by_comm, 0)\n"
"\n"
"im2 = axes_sum[1].imshow(sim_by_comm, cmap=\"inferno\", aspect=\"auto\",\n"
"                          vmin=vmin_j, vmax=vmax_j)\n"
"cumsum_comm = np.cumsum(\n"
"    [0] + [len(c) for c in communities] + [len(missing)])\n"
"for b in cumsum_comm[1:-1]:\n"
"    axes_sum[1].axhline(b - 0.5, color=\"white\", linewidth=0.8, alpha=0.7)\n"
"    axes_sum[1].axvline(b - 0.5, color=\"white\", linewidth=0.8, alpha=0.7)\n"
"axes_sum[1].set_xticks([])\n"
"axes_sum[1].set_yticks([])\n"
"axes_sum[1].set_title(\"Jaccard similarity (ordered by community)\")\n"
"plt.colorbar(im2, ax=axes_sum[1], label=\"Jaccard\", shrink=0.8)\n"
"\n"
"fig_sum.tight_layout()\n"
"plt.show()"
))

cells.append(code_cell(
"# --- Figure 3: Confusion matrix ---\n"
"n_comm = len(communities)\n"
"confusion = np.zeros((len(module_names), n_comm), dtype=int)\n"
"for ci, comm in enumerate(communities):\n"
"    for node in comm:\n"
"        m = neuron_modules.get(node, \"unknown\")\n"
"        if m in module_names:\n"
"            mi = module_names.index(m)\n"
"            confusion[mi, ci] += 1\n"
"\n"
"fig_conf, ax_conf = plt.subplots(\n"
"    figsize=(max(6, n_comm * 0.8 + 2), max(5, len(module_names) * 0.7 + 1)))\n"
"im_conf = ax_conf.imshow(confusion, cmap=\"Blues\", aspect=\"auto\")\n"
"\n"
"for i in range(len(module_names)):\n"
"    for j in range(n_comm):\n"
"        val = confusion[i, j]\n"
"        if val > 0:\n"
"            color = \"white\" if val > confusion.max() / 2 else \"black\"\n"
"            ax_conf.text(j, i, str(val), ha=\"center\", va=\"center\",\n"
"                         fontsize=10, color=color, fontweight=\"bold\")\n"
"\n"
"ax_conf.set_xticks(range(n_comm))\n"
"ax_conf.set_xticklabels([f\"C{i}\" for i in range(n_comm)], fontsize=9)\n"
"ax_conf.set_yticks(range(len(module_names)))\n"
"ax_conf.set_yticklabels([MODULE_SHORT.get(m, m) for m in module_names],\n"
"                         fontsize=9)\n"
"ax_conf.set_xlabel(\"Detected community\", fontsize=11)\n"
"ax_conf.set_ylabel(\"Ground-truth module\", fontsize=11)\n"
"ax_conf.set_title(f\"Community detection confusion matrix (ARI={ari:.3f})\",\n"
"                   fontsize=12)\n"
"plt.colorbar(im_conf, ax=ax_conf, label=\"Neuron count\", shrink=0.8)\n"
"fig_conf.tight_layout()\n"
"plt.show()"
))

# --- 2.7 Further reading ---

cells.append(md_cell(
"## Further reading\n"
"\n"
"Standalone examples (run directly, no external data needed):\n"
"\n"
"- [recurrence_basic](https://github.com/iabs-neuro/driada/tree/main/examples/recurrence_basic) -- Recurrence fundamentals on classic signals\n"
"- [recurrence_population](https://github.com/iabs-neuro/driada/tree/main/examples/recurrence_population) -- Recovering functional modules from population dynamics\n"
"\n"
"Reference: Marwan, N., Romano, M. C., Thiel, M. & Kurths, J. (2007).\n"
"Recurrence plots for the analysis of complex systems. *Physics Reports*,\n"
"438(5--6), 237--329.\n"
"\n"
"[All examples](https://github.com/iabs-neuro/driada/tree/main/examples)"
))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

nb = nbformat.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
})
nb.cells = cells

out_path = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "06_recurrence_analysis.ipynb"
)
out_path = os.path.normpath(out_path)

with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
