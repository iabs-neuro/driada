# Network Analysis Module

## Overview

General-purpose graph analysis module built on NetworkX and scipy sparse matrices. Provides spectral decomposition, thermodynamic entropy, quantum-inspired measures, community detection, and visualization for any graph or network.

The `Network` class serves as the foundation for two key integrations within DRIADA:

- **INTENSE** — `compute_cell_cell_significance()` produces adjacency matrices that can be wrapped into a `Network` for functional connectivity analysis
- **Dimensionality reduction** — graph-based DR methods (Isomap, LLE, Laplacian Eigenmaps, UMAP, diffusion maps) construct a `ProximityGraph` internally, which inherits from `Network` and gains all its spectral and topological analysis capabilities

The module itself is domain-agnostic — it works equally well with structural connectomes, correlation matrices, or any other graph representation.

## Key Components

### Network Class
The core container built from sparse adjacency matrices or NetworkX graphs. Provides:
- Degree distributions, clustering coefficients, connected components
- Community detection (Louvain via NetworkX)
- Spectral decomposition (adjacency, Laplacian, normalized Laplacian, random walk)
- Inverse participation ratio (IPR) and localization signatures
- Estrada communicability and bipartivity index
- Gromov hyperbolicity estimation
- Laplacian Eigenmaps embedding
- Save/load and randomization

### ProximityGraph (in `dim_reduction.graph`)
Inherits from `Network`. Constructs k-NN, UMAP, or epsilon-ball graphs from data matrices for manifold learning. All `Network` analysis methods (spectral, entropy, etc.) are available on the resulting graph.

### Entropy Measures
- `free_entropy()` — von Neumann-type free entropy from Laplacian spectrum
- `q_entropy()` — Renyi entropy of order q
- `spectral_entropy()` — Shannon entropy of normalized eigenvalues

### Quantum-Inspired Methods
- `get_density_matrix()` — density matrix from graph Laplacian
- `renyi_divergence()` — Renyi divergence between two density matrices
- `js_divergence()` — Jensen-Shannon divergence between graphs

### Graph Utilities
- `get_giant_cc_from_graph()` / `get_giant_scc_from_graph()` — extract largest components
- `remove_selfloops_from_graph()` / `remove_isolates_from_graph()`
- `small_world_index()` — small-world coefficient

### Matrix Utilities
- `get_laplacian()` / `get_norm_laplacian()` / `get_rw_laplacian()` — Laplacian variants
- `get_trans_matrix()` — transition (random walk) matrix
- `get_symmetry_index()` — asymmetry measure for directed networks
- `get_ccs_from_adj()` / `get_sccs_from_adj()` — connected components from adjacency

### Randomization
- `randomize_graph()` — degree-preserving rewiring
- `adj_random_rewiring_iom_preserving()` — in/out/mutual degree preserving (directed)

### Visualization
- `draw_net()` — network layout plot
- `draw_degree_distr()` — degree distribution histogram
- `draw_spectrum()` — eigenvalue spectrum
- `draw_eigenvectors()` — eigenvector visualization
- `show_mat()` — adjacency/Laplacian matrix heatmap
- `plot_lem_embedding()` — Laplacian Eigenmaps 2D/3D scatter

## Example: General-Purpose Network Analysis

```python
import scipy.sparse as sp
from driada.network import Network

# Any sparse adjacency matrix — connectome, correlation, etc.
net = Network(adj=adjacency_matrix, directed=False, preprocessing='giant_cc')

# Spectral analysis
net.diagonalize(mode='lap')
spectrum = net.get_spectrum(mode='lap')
ipr = net.get_ipr(mode='lap')

# Community detection
import networkx.algorithms.community as nx_comm
communities = nx_comm.louvain_communities(net.graph, seed=42)

# Entropy at multiple temperatures
S = net.calculate_free_entropy(tlist=[0.1, 1.0, 10.0])

# Visualization
from driada.network import draw_spectrum, draw_degree_distr
draw_spectrum(net)
draw_degree_distr(net)
```

## Example: Functional Network from INTENSE

```python
from driada.intense import compute_cell_cell_significance
from driada.network import Network

# Build functional network from cell-cell MI significance
stats, significance, info, results = compute_cell_cell_significance(
    exp, n_shuffles_stage1=100, n_shuffles_stage2=1000
)

# Wrap into Network for full spectral/topological analysis
net = Network(significance['adjacency'], directed=False)
net.diagonalize(mode='lap')
```

## Example: Spectral Analysis of DR Graph

```python
from driada.dim_reduction import MVData

# Graph-based DR methods construct a ProximityGraph (inherits Network)
mvdata = MVData(neural_data, downsampling=5)
embedding = mvdata.get_embedding(method='isomap', n_components=3, n_neighbors=15)

# The proximity graph is accessible on the Embedding object
pgraph = embedding.graph  # ProximityGraph, inherits from Network
pgraph.diagonalize(mode='lap')
spectrum = pgraph.get_spectrum(mode='lap')
```

## Applications

- Structural and functional connectome analysis
- Functional connectivity networks from calcium imaging (via INTENSE)
- Spectral signatures of graph-based DR embeddings (via ProximityGraph)
- Comparing network structure across conditions, regions, or species
- Community detection and modular organization
- Graph entropy and complexity measures
