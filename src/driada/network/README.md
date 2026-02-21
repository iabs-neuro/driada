# Network Analysis Module

## Overview

This module provides tools for building and analyzing functional networks from neural data. It wraps NetworkX graphs with spectral analysis, thermodynamic entropy, quantum-inspired measures, and visualization.

## Key Components

### Network Class
The core container built from adjacency matrices or NetworkX graphs. Provides:
- Degree distributions, clustering coefficients, connected components
- Community detection (Louvain)
- Spectral decomposition (adjacency, Laplacian, normalized Laplacian, random walk)
- Inverse participation ratio (IPR) and localization signatures
- Estrada communicability and bipartivity index
- Gromov hyperbolicity estimation
- Laplacian Eigenmaps embedding
- Save/load and randomization

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

## Example Usage

```python
from driada.intense import compute_cell_cell_significance
from driada.network import Network

# Build functional network from cell-cell MI significance
stats, significance, info, results = compute_cell_cell_significance(
    exp, n_shuffles_stage1=100, n_shuffles_stage2=1000
)

# Create Network from adjacency matrix
net = Network(significance['adjacency'], directed=False)

# Spectral analysis
net.diagonalize(mode='lap')
spectrum = net.get_spectrum(mode='lap')
ipr = net.get_ipr(mode='lap')

# Community detection
communities = net.g.community_louvain()  # via NetworkX/community

# Entropy
from driada.network import free_entropy
S = free_entropy(net.get_spectrum(mode='lap'), tlist=[0.1, 1.0, 10.0])

# Visualization
from driada.network import draw_spectrum, draw_degree_distr
draw_spectrum(net)
draw_degree_distr(net)
```

## Applications

- Functional connectivity networks from calcium imaging
- Spectral signatures of network organization
- Comparing network structure across conditions or brain regions
- Detecting localization and community structure in neural populations
