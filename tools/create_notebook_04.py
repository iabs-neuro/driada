#!/usr/bin/env python
"""
Generate Notebook 04: Network analysis
==========================================

Assembles a Colab-ready Jupyter notebook covering general-purpose graph
analysis with the Network class: construction from adjacency matrices and
NetworkX graphs, structural properties, community detection, spectral
decomposition, thermodynamic entropy, and null-model comparison.
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
"# Network analysis\n"
"\n"
"The [`Network`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network)\n"
"class provides general-purpose graph analysis: spectral decomposition,\n"
"thermodynamic entropy, community detection, and visualization for any\n"
"graph -- structural connectomes, correlation matrices, functional\n"
"connectivity, or DR proximity graphs (see\n"
"[Notebook 03](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb)).\n"
"This tutorial demonstrates the API on a cell-cell mutual-information\n"
"network.\n"
"\n"
"| Step | Notebook | What it does |\n"
"|---|---|---|\n"
"| **Overview** | [00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb) | Core data structures, quick tour of INTENSE, DR, networks |\n"
"| Neuron analysis | [01 -- Neuron analysis](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/01_data_loading_and_neurons.ipynb) | Spike reconstruction, kinetics optimization, quality metrics, surrogates |\n"
"| Single-neuron selectivity | [02 -- INTENSE](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/02_selectivity_detection_intense.ipynb) | Detect which neurons encode which behavioral variables |\n"
"| Population geometry | [03 -- Dimensionality reduction](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb) | Extract low-dimensional manifolds from population activity |\n"
"| **Network analysis** | **04 -- this notebook** | Build and analyze interaction graphs |\n"
"| Putting it together | [05 -- Advanced](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/05_advanced_capabilities.ipynb) | Combine INTENSE + DR, leave-one-out importance, RSA, RNN analysis |\n"
"\n"
"**In this notebook** we build a functional network from\n"
"significance-tested pairwise MI between neurons, then explore its\n"
"topology:\n"
"\n"
"1. **Network construction** -- create a `Network` from a sparse\n"
"   adjacency matrix or a NetworkX graph, with preprocessing options.\n"
"2. **Structural analysis** -- degree distribution, community detection\n"
"   (Louvain), null-model comparison (degree-preserving randomization).\n"
"3. **Spectral analysis** -- eigendecomposition, IPR, complex spacing\n"
"   ratios, thermodynamic entropy, communicability, hyperbolicity,\n"
"   Laplacian Eigenmaps, and directed-network eigenvalues."
))

cells.append(code_cell(
"# TODO: revert to '!pip install -q driada' after v1.0.0 PyPI release\n"
"!pip install -q git+https://github.com/iabs-neuro/driada.git@main\n"
"%matplotlib inline\n"
"\n"
"import tempfile\n"
"\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import networkx as nx\n"
"import networkx.algorithms.community as nx_comm\n"
"import scipy.sparse as sp\n"
"from scipy.stats import gaussian_kde\n"
"\n"
"from driada.network import Network, draw_net, draw_spectrum\n"
"from driada.network.matrix_utils import turn_to_partially_directed\n"
"from driada.intense import compute_cell_cell_significance\n"
"from driada.experiment.synthetic import generate_tuned_selectivity_exp"
))

# ===== SHARED SETUP: GENERATE DATA & BUILD NETWORK ========================

cells.append(md_cell(
"## Building the network\n"
"\n"
"Create a synthetic modular population (120 neurons, 6 functional\n"
"groups) with [`generate_tuned_selectivity_exp`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.synthetic.generators.generate_tuned_selectivity_exp),\n"
"compute pairwise significance with\n"
"[`compute_cell_cell_significance`](https://driada.readthedocs.io/en/latest/api/intense/pipelines.html#driada.intense.pipelines.compute_cell_cell_significance),\n"
"and construct a [`Network`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network)\n"
"object.  This network is reused in both analysis sections below."
))

cells.append(code_cell(
'def create_modular_experiment(duration=300, seed=42):\n'
'    """\n'
'    Create synthetic experiment with hierarchical modular structure.\n'
'\n'
'    Creates 120 neurons in 6 functional groups:\n'
'    - 3 single-feature modules (30 neurons each): respond to event_0, event_1, or event_2\n'
'    - 3 dual-feature modules (10 neurons each): respond to pairs of events in OR mode\n'
'      (event_0 OR event_1, event_0 OR event_2, event_1 OR event_2)\n'
'\n'
'    This creates a realistic hierarchical network with both specialized\n'
'    and multi-selective neurons.\n'
'    """\n'
'    # Create population with mixed selectivity\n'
'    population = [\n'
'        # Single-feature modules (30 neurons each)\n'
'        {\n'
'            "name": "event_0_cells",\n'
'            "count": 30,\n'
'            "features": ["event_0"],\n'
'        },\n'
'        {\n'
'            "name": "event_1_cells",\n'
'            "count": 30,\n'
'            "features": ["event_1"],\n'
'        },\n'
'        {\n'
'            "name": "event_2_cells",\n'
'            "count": 30,\n'
'            "features": ["event_2"],\n'
'        },\n'
'        # Dual-feature modules (10 neurons each, OR combination)\n'
'        {\n'
'            "name": "event_0_or_1_cells",\n'
'            "count": 10,\n'
'            "features": ["event_0", "event_1"],\n'
'            "combination": "or",\n'
'        },\n'
'        {\n'
'            "name": "event_0_or_2_cells",\n'
'            "count": 10,\n'
'            "features": ["event_0", "event_2"],\n'
'            "combination": "or",\n'
'        },\n'
'        {\n'
'            "name": "event_1_or_2_cells",\n'
'            "count": 10,\n'
'            "features": ["event_1", "event_2"],\n'
'            "combination": "or",\n'
'        },\n'
'    ]\n'
'\n'
'    # Generate experiment with hierarchical structure\n'
'    exp = generate_tuned_selectivity_exp(\n'
'        population=population,\n'
'        n_discrete_features=3,\n'
'        duration=duration,\n'
'        fps=20.0,\n'
'        baseline_rate=0.05,\n'
'        peak_rate=2.0,\n'
'        decay_time=2.0,\n'
'        calcium_noise=0.02,\n'
'        seed=seed,\n'
'        verbose=True\n'
'    )\n'
'\n'
'    # Return info about true module structure\n'
'    n_modules = 6\n'
'    module_sizes = [30, 30, 30, 10, 10, 10]\n'
'\n'
'    return exp, n_modules, module_sizes\n'
'\n'
'\n'
'print("Creating synthetic experiment with hierarchical modular structure...")\n'
'print("  120 neurons: 30+30+30 (single-feature) + 10+10+10 (dual-feature)")\n'
'exp, n_modules_true, module_sizes_true = create_modular_experiment(duration=300)\n'
'print(f"Created {len(exp.neurons)} neurons in {n_modules_true} functional groups")'
))

cells.append(code_cell(
"# Calcium activity and event timeseries\n"
"fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5),\n"
"                                gridspec_kw={'height_ratios': [3, 1]},\n"
"                                sharex=True)\n"
"\n"
"ax1.imshow(exp.calcium.data, aspect='auto', cmap='hot', interpolation='none')\n"
"ax1.set_ylabel('Neuron')\n"
"ax1.set_title(f'Calcium traces ({exp.n_cells} neurons, {exp.n_frames} frames)')\n"
"plt.colorbar(ax1.images[0], ax=ax1, fraction=0.02)\n"
"\n"
"event_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']\n"
"for i in range(3):\n"
"    key = f'event_{i}'\n"
"    if key in exp.dynamic_features:\n"
"        ev = exp.dynamic_features[key].data\n"
"        ax2.fill_between(range(len(ev)), i, i + ev * 0.8,\n"
"                         color=event_colors[i], alpha=0.7, label=key)\n"
"ax2.set_yticks([0.4, 1.4, 2.4])\n"
"ax2.set_yticklabels(['event_0', 'event_1', 'event_2'])\n"
"ax2.set_xlabel('Frame')\n"
"ax2.set_ylim(-0.1, 3)\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(code_cell(
'print("Computing cell-cell functional connectivity")\n'
'\n'
'sim_mat, sig_mat, pval_mat, cells_list, info = compute_cell_cell_significance(\n'
'    exp,\n'
'    data_type="calcium",\n'
'    ds=5,                      # Downsample by 5x for speed (~5x faster)\n'
'    n_shuffles_stage1=100,     # Stage 1 screening\n'
'    n_shuffles_stage2=10000,   # FFT makes high shuffle counts fast!\n'
'    pval_thr=0.001,\n'
'    multicomp_correction="holm",\n'
'    verbose=True\n'
')\n'
'\n'
'print(f"\\nAnalysis complete!")\n'
'print(f"Total neuron pairs: {len(cells_list) * (len(cells_list)-1) / 2:.0f}")\n'
'print(f"Significant connections: {np.sum(sig_mat) / 2:.0f}")  # Divide by 2 for symmetry\n'
'print(f"Connection density: {np.sum(sig_mat) / (len(cells_list) * (len(cells_list)-1)):.3f}")'
))

cells.append(md_cell(
"## 1. Constructing a Network\n"
"\n"
"A [`Network`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network)\n"
"can be created from a **scipy sparse matrix** or a **NetworkX graph**.\n"
"The `preprocessing` parameter controls cleanup:\n"
"\n"
"| Option | Effect |\n"
"|---|---|\n"
"| `None` | Use the graph as-is |\n"
"| `\"remove_isolates\"` | Drop isolated nodes and self-loops |\n"
"| `\"giant_cc\"` | Extract the largest connected component (ensures connectivity for spectral methods) |"
))

cells.append(code_cell(
'sig_sparse = sp.csr_matrix(sig_mat)\n'
'\n'
'# From a sparse adjacency matrix (most common path)\n'
'net_binary = Network(\n'
'    adj=sig_sparse,\n'
'    preprocessing="giant_cc",\n'
'    name="Functional Network (Binary)"\n'
')\n'
'\n'
'# Equivalently, from a NetworkX graph\n'
'G = nx.from_scipy_sparse_array(sig_sparse)\n'
'net_from_graph = Network(\n'
'    graph=G,\n'
'    preprocessing="remove_isolates",\n'
'    name="Functional Network (from graph)"\n'
')\n'
'\n'
'# Create weighted network (similarity values for significant edges)\n'
'weighted_sparse = sp.csr_matrix(sim_mat * sig_mat)\n'
'net_weighted = Network(\n'
'    adj=weighted_sparse,\n'
'    preprocessing="giant_cc",\n'
'    name="Functional Network (Weighted)"\n'
')\n'
'\n'
'print(f"Binary  network: {net_binary.n} nodes, {net_binary.graph.number_of_edges()} edges")\n'
'print(f"From NX graph:   {net_from_graph.n} nodes, {net_from_graph.graph.number_of_edges()} edges")\n'
'print(f"Weighted network: {net_weighted.n} nodes, {net_weighted.graph.number_of_edges()} edges")'
))

cells.append(code_cell(
'# Visualize the network right away with draw_net\n'
'draw_net(net_binary)'
))

cells.append(md_cell(
"### Save and reload\n"
"\n"
"A `Network` stores its adjacency as `net.adj` (scipy sparse). Save it\n"
"with [`scipy.sparse.save_npz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html)\n"
"and reconstruct with `Network(adj=...)`."
))

cells.append(code_cell(
'import os\n'
'\n'
'with tempfile.TemporaryDirectory() as tmpdir:\n'
'    path = os.path.join(tmpdir, "network.npz")\n'
'    sp.save_npz(path, net_weighted.adj)\n'
'    print(f"Saved {os.path.getsize(path) / 1024:.1f} KB")\n'
'\n'
'    loaded_adj = sp.load_npz(path)\n'
'    net_reloaded = Network(adj=loaded_adj, preprocessing="giant_cc")\n'
'    assert net_reloaded.n == net_weighted.n\n'
'    print(f"Reloaded: {net_reloaded.n} nodes, {net_reloaded.graph.number_of_edges()} edges")'
))

# ===== SECTION 2: STRUCTURAL ANALYSIS =====================================

cells.append(md_cell(
"## 2. Structural analysis\n"
"\n"
"Degree distribution, clustering, connected components, community\n"
"detection, and null-model comparison."
))

cells.append(md_cell(
"### Network properties\n"
"\n"
"The `Network` object exposes structural attributes directly\n"
"(`net.n`, `net.directed`, `net.weighted`). For standard graph metrics\n"
"(clustering, components), we use networkx on `net.graph`."
))

cells.append(code_cell(
'net = net_weighted\n'
'print(f"Network type: {\'Directed\' if net.directed else \'Undirected\'}, "\n'
'      f"{\'Weighted\' if net.weighted else \'Binary\'}")\n'
'print(f"Nodes: {net.n}")\n'
'print(f"Edges: {net.graph.number_of_edges()}")\n'
'\n'
'degrees = [d for _, d in net.graph.degree()]\n'
'print(f"Degree: {np.mean(degrees):.1f} +/- {np.std(degrees):.1f} (max {np.max(degrees)})")\n'
'\n'
'if not net.directed:\n'
'    print(f"Clustering coefficient: {nx.average_clustering(net.graph):.3f}")\n'
'    print(f"Connected components: {nx.number_connected_components(net.graph)}")\n'
'\n'
'if net.weighted:\n'
'    weights = [d["weight"] for _, _, d in net.graph.edges(data=True)]\n'
'    print(f"Weight: {np.mean(weights):.4f} +/- {np.std(weights):.4f} "\n'
'          f"[{np.min(weights):.4f}, {np.max(weights):.4f}]")'
))

cells.append(md_cell(
"### Community detection\n"
"\n"
"Let's identify functional modules using Louvain community detection."
))

cells.append(code_cell(
'if net.weighted:\n'
'    communities = nx_comm.louvain_communities(net.graph, weight="weight", seed=42)\n'
'else:\n'
'    communities = nx_comm.louvain_communities(net.graph, seed=42)\n'
'\n'
'print(f"Found {len(communities)} modules:")\n'
'for i, community in enumerate(communities):\n'
'    print(f"  Module {i+1}: {len(community)} neurons")\n'
'\n'
'module_assignment = {}\n'
'for module_idx, community in enumerate(communities):\n'
'    for node in community:\n'
'        module_assignment[node] = module_idx'
))

cells.append(md_cell(
"### Null-model comparison\n"
"\n"
"Are these properties specific to our network, or would any graph with\n"
"the same degree sequence look similar?\n"
"[`net.randomize`](https://driada.readthedocs.io/en/latest/api/network/randomization.html)\n"
"generates degree-preserving random rewirings."
))

cells.append(code_cell(
'n_replicates = 10\n'
'\n'
'real_clustering = nx.average_clustering(net.graph)\n'
'real_modularity = nx_comm.modularity(net.graph, communities)\n'
'lap_spectrum = net.get_spectrum("lap")\n'
'real_connectivity = np.sort(np.real(lap_spectrum))[1]\n'
'\n'
'null_clustering, null_modularity, null_connectivity = [], [], []\n'
'\n'
'for i in range(n_replicates):\n'
'    rand_net = net.randomize(rmode="adj_iom")\n'
'    null_clustering.append(nx.average_clustering(rand_net.graph))\n'
'    rand_comms = nx_comm.louvain_communities(rand_net.graph, seed=i)\n'
'    null_modularity.append(nx_comm.modularity(rand_net.graph, rand_comms))\n'
'    rand_lap = rand_net.get_spectrum("lap")\n'
'    null_connectivity.append(np.sort(np.real(rand_lap))[1])\n'
'\n'
'print(f"  {\'Metric\':<25s} {\'Real\':>8s} {\'Null (mean +/- std)\':>22s}")\n'
'print(f"  {\'-\' * 57}")\n'
'for name, real_val, null_vals in [\n'
'    ("Clustering coefficient", real_clustering, null_clustering),\n'
'    ("Modularity", real_modularity, null_modularity),\n'
'    ("Algebraic connectivity", real_connectivity, null_connectivity),\n'
']:\n'
'    print(f"  {name:<25s} {real_val:>8.3f} {np.mean(null_vals):>10.3f} +/- {np.std(null_vals):.3f}")\n'
'print(f"\\n  Replicates: {n_replicates}")'
))

cells.append(code_cell(
'# Summary visualization\n'
'unique_modules = sorted(set(module_assignment.values()))\n'
'n_modules = len(unique_modules)\n'
'module_colors = plt.cm.tab10(np.linspace(0, 1, n_modules))\n'
'module_to_color = {mod: module_colors[i] for i, mod in enumerate(unique_modules)}\n'
'\n'
'fig = plt.figure(figsize=(16, 8))\n'
'\n'
'# 1. Similarity matrix with module boundaries\n'
'ax1 = plt.subplot(1, 3, 1)\n'
'im1 = ax1.imshow(sim_mat, cmap="hot", aspect="auto")\n'
'ax1.set_title("Similarity matrix (MI)")\n'
'ax1.set_xlabel("Neuron ID")\n'
'ax1.set_ylabel("Neuron ID")\n'
'plt.colorbar(im1, ax=ax1, fraction=0.046)\n'
'if module_sizes_true is not None:\n'
'    cumsum = np.cumsum([0] + module_sizes_true)\n'
'    for boundary in cumsum[1:-1]:\n'
'        ax1.axhline(boundary - 0.5, color="cyan", linewidth=2)\n'
'        ax1.axvline(boundary - 0.5, color="cyan", linewidth=2)\n'
'\n'
'# 2. Network graph colored by module\n'
'ax2 = plt.subplot(1, 3, 2)\n'
'pos = nx.spring_layout(net.graph, k=1.5/np.sqrt(len(net.graph)),\n'
'                        iterations=100, seed=42)\n'
'node_colors = [module_to_color[module_assignment[n]] for n in net.graph.nodes()]\n'
'nx.draw_networkx_nodes(net.graph, pos, node_color=node_colors,\n'
'                       node_size=20, ax=ax2, alpha=0.8)\n'
'nx.draw_networkx_edges(net.graph, pos, alpha=0.1, ax=ax2)\n'
'ax2.set_title("Functional network modules")\n'
'ax2.axis("off")\n'
'\n'
'# 3. Degree distribution\n'
'ax3 = plt.subplot(1, 3, 3)\n'
'degrees = [d for _, d in net.graph.degree()]\n'
'ax3.hist(degrees, bins=20, alpha=0.7, color="blue", edgecolor="black")\n'
'ax3.set_xlabel("Degree")\n'
'ax3.set_ylabel("Count")\n'
'ax3.set_title("Degree distribution")\n'
'ax3.grid(True, alpha=0.3)\n'
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

# ===== SECTION 3: SPECTRAL ANALYSIS =======================================

cells.append(md_cell(
"## 3. Spectral analysis\n"
"\n"
"Eigendecomposition of the adjacency and normalized Laplacian matrices\n"
"reveals global structure that is invisible to local metrics.\n"
"\n"
"> **Note:** Everything below applies to *any* `Network` -- not just\n"
"> cell-cell graphs. Graph-based DR methods (Isomap, LLE, Laplacian\n"
"> Eigenmaps) produce a `ProximityGraph` that inherits from `Network`,\n"
"> so all spectral, entropy, and community tools work on DR graphs too.\n"
"> See [Notebook 03](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb).\n"
"\n"
"**Adjacency spectrum.** The adjacency eigenvalues reflect community\n"
"structure: isolated clusters produce near-degenerate eigenvalue groups.\n"
"The spectral radius (largest $|\\lambda|$) scales with the mean degree\n"
"for random graphs.\n"
"\n"
"**Normalized Laplacian.**\n"
"$L_{\\text{norm}} = I - D^{-1/2} A D^{-1/2}$\n"
"has eigenvalues bounded in $[0, 2]$ regardless of network size or degree\n"
"distribution, making it suitable for cross-network comparison."
))

cells.append(code_cell(
'net_spectral = net_binary\n'
'\n'
'# net.get_spectrum() and net.get_eigenvectors() cache results automatically\n'
'adj_spectrum = net_spectral.get_spectrum("adj")\n'
'print(f"\\nAdjacency matrix:")\n'
'print(f"  Spectral radius (max |lambda|): {np.max(np.abs(adj_spectrum)):.3f}")\n'
'print(f"  Min eigenvalue: {np.min(np.real(adj_spectrum)):.3f}")\n'
'print(f"  Max eigenvalue: {np.max(np.real(adj_spectrum)):.3f}")\n'
'\n'
'nlap_spectrum = net_spectral.get_spectrum("nlap")\n'
'sorted_nlap = np.sort(np.real(nlap_spectrum))\n'
'\n'
'# Each zero eigenvalue corresponds to one connected component.\n'
'# Since we extracted the giant connected component, expect exactly 1.\n'
'n_zero = int(np.sum(np.abs(sorted_nlap) < 1e-6))\n'
'n_components = n_zero\n'
'\n'
'print(f"\\nNormalized Laplacian (L = I - D^(-1/2) A D^(-1/2)):")\n'
'print(f"  Eigenvalue range: [0, 2]")\n'
'print(f"  Smallest eigenvalue: {sorted_nlap[0]:.6f}")\n'
'print(f"  Zero eigenvalues: {n_zero}  (= number of connected components)")\n'
'if n_components == 1:\n'
'    print(f"  Graph is connected (single component)")\n'
'else:\n'
'    print(f"  WARNING: graph has {n_components} disconnected components")\n'
'\n'
'# The Fiedler value (first non-zero eigenvalue) is the algebraic\n'
'# connectivity of the normalized Laplacian. Larger values indicate\n'
'# a graph that is harder to disconnect by removing edges.\n'
'if len(sorted_nlap) > n_zero:\n'
'    fiedler = sorted_nlap[n_zero]\n'
'    print(f"  Fiedler value (lambda_{n_zero + 1}): {fiedler:.4f}")\n'
'    print(f"    (Algebraic connectivity -- larger = harder to disconnect)")\n'
'print(f"  Spectral gap: {sorted_nlap[n_zero] - sorted_nlap[0]:.4f}")\n'
'print(f"  Largest eigenvalue: {sorted_nlap[-1]:.4f}")\n'
'\n'
'adj_vecs = net_spectral.get_eigenvectors("adj")\n'
'print(f"\\nAdjacency eigenvectors: {adj_vecs.shape}  (N x N matrix)")'
))

cells.append(md_cell(
"### Spectral metrics\n"
"\n"
"**IPR (Inverse Participation Ratio)**\n"
"([`get_ipr`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.get_ipr))**:**\n"
"$\\text{IPR} = \\sum_i |v_i|^4$ for each eigenvector $v$.\n"
"For a vector uniformly spread over $N$ nodes, $\\text{IPR} = 1/N$.\n"
"For a vector concentrated on one node, $\\text{IPR} = 1$."
))

cells.append(code_cell(
'print("IPR analysis (eigenvector localization)")\n'
'\n'
'ipr_adj = net_spectral.get_ipr("adj")\n'
'ipr_nlap = net_spectral.get_ipr("nlap")\n'
'\n'
'delocalized_bound = 1.0 / net_spectral.n\n'
'\n'
'print(f"\\nAdjacency IPR:")\n'
'print(f"  Mean: {np.mean(ipr_adj):.4f}")\n'
'print(f"  Min:  {np.min(ipr_adj):.4f}  (delocalized bound 1/N = {delocalized_bound:.4f})")\n'
'print(f"  Max:  {np.max(ipr_adj):.4f}")\n'
'\n'
'print(f"\\nNormalized Laplacian IPR:")\n'
'print(f"  Mean: {np.mean(ipr_nlap):.4f}")\n'
'print(f"  Min:  {np.min(ipr_nlap):.4f}")\n'
'print(f"  Max:  {np.max(ipr_nlap):.4f}")'
))

cells.append(md_cell(
"**Complex spacing ratios**\n"
"([`get_z_values`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.get_z_values),\n"
"Sá, Ribeiro & Prosen, [*Phys. Rev. X* 10, 021019, 2020](https://doi.org/10.1103/PhysRevX.10.021019))**.**\n"
"For a symmetric (Hermitian) matrix all eigenvalues are real, so the\n"
"complex spacing ratios $z = (\\lambda_{\\text{nn}} - \\lambda) /\n"
"(\\lambda_{\\text{nnn}} - \\lambda)$ collapse to the real line and\n"
"$\\arg(z)$ is either $0$ or $\\pi$.\n"
"\n"
"**Localization signatures**\n"
"([`localization_signatures`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.localization_signatures))**.**\n"
"$\\langle\\cos(\\arg z)\\rangle$ measures phase coherence of spacing ratios;\n"
"$\\langle 1/|z|^2 \\rangle$ amplifies cases where nearest and next-nearest\n"
"neighbour distances differ strongly."
))

cells.append(code_cell(
'print("Spacing ratios and localization signatures")\n'
'\n'
'z_dict = net_spectral.get_z_values("nlap")\n'
'\n'
'# Discard the z-value for the zero eigenvalue (one per connected component).\n'
'# It is trivial and would distort 1/|z|^2 statistics.\n'
'z_filtered = {k: v for k, v in z_dict.items() if np.abs(k) > 1e-6}\n'
'n_discarded = len(z_dict) - len(z_filtered)\n'
'z_list = np.array(list(z_filtered.values()))\n'
'z_mags = np.abs(z_list)\n'
'\n'
'print(f"\\n  Total eigenvalues with z-values: {len(z_dict)}")\n'
'print(f"  Discarded (zero eigenvalues): {n_discarded}")\n'
'print(f"  Used for analysis: {len(z_filtered)}")\n'
'\n'
'print(f"\\nComplex spacing ratios z = (lambda_nn - lambda) / (lambda_nnn - lambda):")\n'
'print(f"  Count: {len(z_list)}")\n'
'print(f"  Mean |z|: {np.mean(z_mags):.4f}  (bounded in [0, 1])")\n'
'print(f"  Std |z|:  {np.std(z_mags):.4f}")'
))

cells.append(code_cell(
'mean_inv_r2, mean_cos_phi = net_spectral.localization_signatures("nlap")\n'
'print(f"\\nLocalization signatures (undirected, normalized Laplacian):")\n'
'print(f"  <cos(arg(z))>: {mean_cos_phi:.4f}")\n'
'print(f"  <1/|z|^2>:     {mean_inv_r2:.4f}")'
))

cells.append(md_cell(
"**Communicability**\n"
"([`calculate_estrada_communicability`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.calculate_estrada_communicability),\n"
"Estrada & Hatano, [*Phys. Rev. E* 77, 036111, 2008](https://doi.org/10.1103/PhysRevE.77.036111))**.**\n"
"$\\text{EE} = \\sum_i \\exp(\\lambda_i)$ where $\\lambda_i$ are adjacency\n"
"eigenvalues. Counts walks of all lengths, weighted by $1/k!$.\n"
"\n"
"**Bipartivity index**\n"
"([`get_estrada_bipartivity_index`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.get_estrada_bipartivity_index),\n"
"Estrada & Rodríguez-Velázquez, [*Phys. Rev. E* 72, 046105, 2005](https://doi.org/10.1103/PhysRevE.72.046105))**.**\n"
"Ratio of even-length to total weighted walks.\n"
"\n"
"**Gromov hyperbolicity**\n"
"([`calculate_gromov_hyperbolicity`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.calculate_gromov_hyperbolicity),\n"
"Chalopin et al., [arXiv:1803.06324, 2018](https://arxiv.org/abs/1803.06324))**.**\n"
"For every 4-point set, measures how far the shortest-path metric deviates\n"
"from a tree metric. $\\delta = 0$ means the network is a tree."
))

cells.append(code_cell(
'print("Communicability and network geometry")\n'
'\n'
'comm = net_spectral.calculate_estrada_communicability()\n'
'print(f"\\nEstrada communicability index: {comm:.4g}")\n'
'\n'
'bipartivity = net_spectral.get_estrada_bipartivity_index()\n'
'print(f"Estrada bipartivity index: {bipartivity:.4f}")\n'
'print(f"  (1.0 = perfectly bipartite, 0.0 = far from bipartite)")\n'
'\n'
'hyp = net_spectral.calculate_gromov_hyperbolicity(num_samples=50000)\n'
'print(f"\\nGromov hyperbolicity (mean delta): {hyp:.3f}")\n'
'print(f"  (0 = tree-like, higher = more cycle-rich)")'
))

cells.append(md_cell(
"**Thermodynamic entropy**\n"
"(De Domenico & Biamonte, [*Phys. Rev. X* 6, 041062, 2016](https://doi.org/10.1103/PhysRevX.6.041062))**.**\n"
"Temperature sweep: low $t$ probes local structure, high $t$ probes global.\n"
"\n"
"- [`calculate_thermodynamic_entropy`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.calculate_thermodynamic_entropy):\n"
"  Von Neumann entropy $S(t) = -\\sum_i p_i \\log_2 p_i$ where\n"
"  $p_i = \\exp(-\\lambda_i / t) / Z$.\n"
"- [`calculate_free_entropy`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.calculate_free_entropy):\n"
"  $F(t) = \\log_2 Z$ where $Z = \\sum_i \\exp(-\\lambda_i / t)$.\n"
"- [`calculate_q_entropy`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.calculate_q_entropy):\n"
"  Rényi $q$-entropy $S_q(t) = \\log_2(\\sum_i p_i^q) / (1 - q)$."
))

cells.append(code_cell(
'print("Thermodynamic entropy analysis")\n'
'\n'
'tlist = np.logspace(-2, 2, 50)\n'
'\n'
'vn_entropy = net_spectral.calculate_thermodynamic_entropy(tlist, norm=True)\n'
'print(f"\\nVon Neumann entropy S(t) [normalized Laplacian]:")\n'
'print(f"  At t=0.01: {vn_entropy[0]:.3f} bits")\n'
'print(f"  At t=1.00: {vn_entropy[len(tlist) // 2]:.3f} bits")\n'
'print(f"  At t=100:  {vn_entropy[-1]:.3f} bits")\n'
'print(f"  Max entropy: {np.max(vn_entropy):.3f} bits"\n'
'      f"  (upper bound = log2(N) = {np.log2(net_spectral.n):.2f})")\n'
'\n'
'free_ent = net_spectral.calculate_free_entropy(tlist, norm=True)\n'
'print(f"\\nFree entropy F(t) = log2(Z):")\n'
'print(f"  At t=0.01: {free_ent[0]:.3f}")\n'
'print(f"  At t=100:  {free_ent[-1]:.3f}")\n'
'\n'
'q_ent = net_spectral.calculate_q_entropy(q=2, tlist=tlist, norm=True)\n'
'print(f"\\nRenyi 2-entropy S_2(t):")\n'
'print(f"  At t=0.01: {q_ent[0]:.3f} bits")\n'
'print(f"  At t=100:  {q_ent[-1]:.3f} bits")'
))

cells.append(md_cell(
"### Laplacian Eigenmaps embedding\n"
"\n"
"[`construct_lem_embedding`](https://driada.readthedocs.io/en/latest/api/network/core.html#driada.network.net_base.Network.construct_lem_embedding)\n"
"uses the smallest non-zero eigenvectors of the normalized Laplacian\n"
"as embedding coordinates (Belkin & Niyogi,\n"
"[*Neural Computation* 15(6), 2003](https://doi.org/10.1162/089976603321780317)).\n"
"Nearby nodes in the graph map to nearby points in the embedding."
))

cells.append(code_cell(
'print("Laplacian Eigenmaps embedding")\n'
'\n'
'dim = 3\n'
'net_spectral.construct_lem_embedding(dim)\n'
'\n'
'# Access stored embedding (shape: dim x n_nodes)\n'
'if hasattr(net_spectral.lem_emb, "toarray"):\n'
'    emb_data = np.real(net_spectral.lem_emb.toarray())\n'
'else:\n'
'    emb_data = np.real(np.asarray(net_spectral.lem_emb))\n'
'\n'
'print(f"\\nLEM embedding ({dim}D):")\n'
'print(f"  Shape: {emb_data.shape}")\n'
'print(f"  Dim 1 range: [{emb_data[0].min():.3f}, {emb_data[0].max():.3f}]")\n'
'print(f"  Dim 2 range: [{emb_data[1].min():.3f}, {emb_data[1].max():.3f}]")\n'
'print(f"  Dim 3 range: [{emb_data[2].min():.3f}, {emb_data[2].max():.3f}]")\n'
'\n'
'# Plot LEM embedding colored by detected module\n'
'fig, ax = plt.subplots(figsize=(8, 6))\n'
'# Map nodes in the spectral network to their modules\n'
'node_list = list(net_spectral.graph.nodes())\n'
'node_module_colors = [module_to_color.get(module_assignment.get(n, 0), "gray") for n in node_list]\n'
'ax.scatter(emb_data[0], emb_data[1], s=15, alpha=0.7, c=node_module_colors, edgecolors="none")\n'
'ax.set_xlabel("LEM dim 1")\n'
'ax.set_ylabel("LEM dim 2")\n'
'ax.set_title("Laplacian Eigenmaps (colored by module)")\n'
'ax.grid(True, alpha=0.3)\n'
'plt.tight_layout()\n'
'plt.show()'
))

cells.append(md_cell(
"### Directed variant\n"
"\n"
"[`turn_to_partially_directed`](https://driada.readthedocs.io/en/latest/api/network/matrix_utils.html#driada.network.matrix_utils.turn_to_partially_directed)\n"
"randomly orients edges, breaking Hermitian symmetry and producing\n"
"complex eigenvalues whose $z$-values spread across the complex plane.\n"
"In practice, directed networks arise from causal or effective\n"
"connectivity (e.g. Granger causality, transfer entropy)."
))

cells.append(code_cell(
'dir_adj = turn_to_partially_directed(net_spectral.adj, directed=1.0)\n'
'dir_net = Network(adj=dir_adj, preprocessing=None, name="Directed variant")\n'
'\n'
'z_dict_dir = dir_net.get_z_values("adj")\n'
'z_list_dir = np.array(list(z_dict_dir.values()))\n'
'z_mags_dir = np.abs(z_list_dir)\n'
'\n'
'print(f"\\nDirected variant (randomly oriented edges):")\n'
'print(f"  Eigenvalues: {len(z_dict_dir)} (complex)")\n'
'print(f"  Mean |z|: {np.mean(z_mags_dir):.4f}")\n'
'print(f"  Std |z|:  {np.std(z_mags_dir):.4f}")\n'
'\n'
'mean_inv_r2_d, mean_cos_phi_d = dir_net.localization_signatures("adj")\n'
'print(f"\\nLocalization signatures (directed adjacency):")\n'
'print(f"  <cos(arg(z))>: {mean_cos_phi_d:.4f}")\n'
'print(f"  <1/|z|^2>:     {mean_inv_r2_d:.4f}")\n'
'\n'
'# Plot complex eigenvalues\n'
'dir_spectrum = dir_net.get_spectrum("adj")\n'
'fig, ax = plt.subplots(figsize=(8, 8))\n'
'ax.scatter(np.real(dir_spectrum), np.imag(dir_spectrum), s=20, alpha=0.6,\n'
'           c="steelblue", edgecolors="none")\n'
'theta = np.linspace(0, 2 * np.pi, 200)\n'
'r_max = np.max(np.abs(dir_spectrum))\n'
'ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), "k--", alpha=0.3, linewidth=0.8)\n'
'ax.set_xlabel("Re(lambda)")\n'
'ax.set_ylabel("Im(lambda)")\n'
'ax.set_title("Directed network eigenvalues")\n'
'ax.set_aspect("equal")\n'
'ax.grid(True, alpha=0.3)\n'
'plt.tight_layout()\n'
'plt.show()'
))

# ===== SPECTRAL SUMMARY ====================================================

cells.append(md_cell(
"### Spectral null-model comparison\n"
"\n"
"Compare spectral metrics against degree-preserving random rewirings\n"
"using [`net.randomize`](https://driada.readthedocs.io/en/latest/api/network/randomization.html)."
))

cells.append(code_cell(
'print("Spectral null model comparison (degree-preserving randomization)")\n'
'\n'
'real_ipr = np.mean(net_spectral.get_ipr("nlap"))\n'
'nlap_sorted = np.sort(np.real(net_spectral.get_spectrum("nlap")))\n'
'real_fiedler = nlap_sorted[nlap_sorted > 1e-6][0]\n'
'\n'
'n_replicates = 10\n'
'null_comm = []\n'
'null_bipart = []\n'
'null_ipr = []\n'
'null_fiedler = []\n'
'\n'
'for _ in range(n_replicates):\n'
'    rand_net = net_spectral.randomize(rmode="adj_iom")\n'
'    null_comm.append(rand_net.calculate_estrada_communicability())\n'
'    null_bipart.append(rand_net.get_estrada_bipartivity_index())\n'
'    null_ipr.append(np.mean(rand_net.get_ipr("nlap")))\n'
'    rand_nlap = np.sort(np.real(rand_net.get_spectrum("nlap")))\n'
'    nonzero = rand_nlap[rand_nlap > 1e-6]\n'
'    null_fiedler.append(nonzero[0] if len(nonzero) > 0 else 0.0)\n'
'\n'
'print(f"\\n  {\'Metric\':<28s} {\'Real\':>12s} {\'Null (mean +/- std)\':>25s}")\n'
'print(f"  {\'-\' * 67}")\n'
'\n'
'rows = [\n'
'    ("Communicability", comm, null_comm),\n'
'    ("Bipartivity", bipartivity, null_bipart),\n'
'    ("Mean nlap IPR", real_ipr, null_ipr),\n'
'    ("Fiedler value", real_fiedler, null_fiedler),\n'
']\n'
'for name, real_val, null_vals in rows:\n'
'    null_mean = np.mean(null_vals)\n'
'    null_std = np.std(null_vals)\n'
'    print(f"  {name:<28s} {real_val:>12.4g} {null_mean:>12.4g} +/- {null_std:.4g}")\n'
'\n'
'print(f"\\n  Replicates: {n_replicates}")'
))

cells.append(code_cell(
'# 2x3 spectral summary figure\n'
'fig, axes = plt.subplots(2, 3, figsize=(18, 11))\n'
'\n'
'# 1. Adjacency spectrum histogram\n'
'ax = axes[0, 0]\n'
'real_spec = np.real(adj_spectrum)\n'
'nbins = int(np.ceil(np.log2(len(real_spec)))) + 1\n'
'ax.hist(real_spec, bins=nbins, edgecolor="black", linewidth=0.5, alpha=0.8)\n'
'ax.set_xlabel("Eigenvalue")\n'
'ax.set_ylabel("Count")\n'
'ax.set_title("Adjacency spectrum")\n'
'ax.grid(True, alpha=0.3)\n'
'\n'
'# 2. Normalized Laplacian IPR (sorted by magnitude)\n'
'ax = axes[0, 1]\n'
'ax.plot(np.sort(ipr_nlap), "o-", markersize=2, linewidth=0.8)\n'
'ax.axhline(1.0 / net_spectral.n, color="r", linestyle="--", label=f"1/N = {1.0/net_spectral.n:.4f}")\n'
'ax.set_xlabel("Eigenvector index (sorted)")\n'
'ax.set_ylabel("IPR")\n'
'ax.set_title("Normalized Laplacian IPR")\n'
'ax.legend(fontsize=9)\n'
'ax.grid(True, alpha=0.3)\n'
'\n'
'# 3. Thermodynamic entropy curve\n'
'ax = axes[0, 2]\n'
'ax.semilogx(tlist, vn_entropy, "b-", linewidth=2)\n'
'ax.set_xlabel("Temperature (t)")\n'
'ax.set_ylabel("Entropy (bits)")\n'
'ax.set_title("Von Neumann entropy S(t)")\n'
'ax.grid(True, alpha=0.3)\n'
'\n'
'# 4. Complex spacing ratio density (directed network).\n'
'# Randomly orienting edges breaks the Hermitian symmetry, giving\n'
'# complex eigenvalues whose z-values spread across the unit disk.\n'
'# The density pattern is a fingerprint of the spectral universality class.\n'
'ax = axes[1, 0]\n'
'zr, zi = np.real(z_list_dir), np.imag(z_list_dir)\n'
'xy = np.vstack([zr, zi])\n'
'kde = gaussian_kde(xy, bw_method=0.25)\n'
'grid_n = 200\n'
'pad = 0.15\n'
'xmin, xmax = zr.min() - pad, zr.max() + pad\n'
'ymin, ymax = zi.min() - pad, zi.max() + pad\n'
'xg = np.linspace(xmin, xmax, grid_n)\n'
'yg = np.linspace(ymin, ymax, grid_n)\n'
'Xg, Yg = np.meshgrid(xg, yg)\n'
'Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(grid_n, grid_n)\n'
'ax.pcolormesh(Xg, Yg, Z, shading="auto", cmap="inferno")\n'
'# Unit circle for reference (|z| <= 1)\n'
'theta = np.linspace(0, 2 * np.pi, 200)\n'
'ax.plot(np.cos(theta), np.sin(theta), "w--", linewidth=0.8, alpha=0.5)\n'
'ax.set_xlabel("Re(z)")\n'
'ax.set_ylabel("Im(z)")\n'
'ax.set_title("Complex spacing ratios (directed)")\n'
'ax.set_aspect("equal")\n'
'\n'
'# 5. LEM embedding (first 2 dims)\n'
'ax = axes[1, 1]\n'
'ax.scatter(emb_data[0], emb_data[1], s=15, alpha=0.7, c="steelblue", edgecolors="none")\n'
'ax.set_xlabel("LEM dim 1")\n'
'ax.set_ylabel("LEM dim 2")\n'
'ax.set_title("Laplacian Eigenmaps")\n'
'ax.grid(True, alpha=0.3)\n'
'\n'
'# 6. Real vs null communicability\n'
'ax = axes[1, 2]\n'
'ax.hist(null_comm, bins=8, edgecolor="black", linewidth=0.5, alpha=0.6,\n'
'        color="gray", label="Null model")\n'
'ax.axvline(comm, color="red", linewidth=2, label=f"Real = {comm:.1f}")\n'
'ax.set_xlabel("Communicability")\n'
'ax.set_ylabel("Count")\n'
'ax.set_title("Real vs null model")\n'
'ax.legend(fontsize=9)\n'
'ax.grid(True, alpha=0.3)\n'
'\n'
'plt.tight_layout()\n'
'plt.show()'
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
    os.path.dirname(__file__), "..", "notebooks", "04_network_analysis.ipynb"
)
out_path = os.path.normpath(out_path)

with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
