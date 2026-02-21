# DRIADA interactive tutorials

Five self-contained Colab/Jupyter notebooks that walk through the full DRIADA
toolkit -- from loading data to building functional networks and running
cross-method analyses. Every notebook generates its own synthetic data, so no
external files are needed.

## Notebooks

### 1. [Data loading & neuron analysis](01_data_loading_and_neurons.ipynb)

Loading data into DRIADA, inspecting `Experiment` objects, and working with
individual neurons.

- Creating an `Experiment` from aligned arrays (`load_exp_from_aligned_data`)
- Feature types: continuous, circular, categorical, aggregated
- `TimeSeries` / `MultiTimeSeries` data representations
- Single-neuron spike reconstruction (wavelet vs threshold), kinetics, quality metrics
- Calcium and spike surrogates
- Save / reload experiments with pickle

### 2. [Detecting neuron selectivity with INTENSE](02_selectivity_detection_intense.ipynb)

Information-theoretic selectivity testing -- the core DRIADA workflow.

- Information theory basics: `get_mi`, `get_sim`, `get_tdmi`, `conditional_mi`, `interaction_information`
- GCMI vs KSG estimators
- `compute_cell_feat_significance` -- two-stage shuffle pipeline with ground-truth validation
- Optimal temporal delays
- Feature-feature significance (`compute_feat_feat_significance`)
- Mixed selectivity and disentanglement analysis
- Saving / loading INTENSE results

### 3. [Population geometry & dimensionality reduction](03_population_geometry_dr.ipynb)

Embedding neural populations and measuring manifold quality.

- `MVData` / `get_embedding` API -- PCA, Isomap, LLE, LE, t-SNE, UMAP, MDS
- Sequential DR (`dr_sequence`), custom distance metrics, sparse data
- Method comparison: speed benchmark, quality metrics (kNN preservation, trustworthiness, continuity, stress)
- Autoencoder-based DR (`flexible_ae`, continue learning)
- Circular manifold analysis and alignment metrics
- Intrinsic dimensionality estimation (PCA, correlation, geodesic, effective)
- INTENSE-guided DR: selectivity on embeddings

### 4. [Functional networks](04_functional_networks.ipynb)

Cell-cell significance networks and spectral graph analysis.

- `compute_cell_cell_significance` -- building binary and weighted networks
- `Network` object: degree distributions, clustering, communities (Louvain)
- Spectral analysis: eigenvalues, eigenvectors, IPR, complex spacing ratios
- Estrada communicability, bipartivity, Gromov hyperbolicity
- Thermodynamic, free, and Renyi entropy
- Laplacian Eigenmaps embedding
- Directed networks and localization signatures
- Network save / load and randomization

### 5. [Advanced capabilities](05_advanced_capabilities.ipynb)

Cross-method analyses and non-calcium applications.

- **Embedding selectivity** -- running INTENSE on DR components (`compute_embedding_selectivity`, `get_functional_organization`, `compare_embeddings`)
- **Leave-one-out neuron importance** -- measuring each neuron's contribution to manifold reconstruction
- **RSA** -- `compute_rdm_unified`, `rsa_compare`, cross-region and cross-session RDM comparison, bootstrap confidence intervals, MVData integration
- **Beyond calcium** -- applying the full DRIADA pipeline (INTENSE + networks + DR) to RNN activations

## Running the notebooks

All notebooks are designed to run in Google Colab (install cell included) or
locally with a standard Jupyter setup:

```bash
pip install driada jupyter
jupyter notebook
```

Notebooks are independent -- you can start with any one -- but they are ordered
from foundational concepts to advanced analyses.

## Further resources

- [INTENSE mathematical framework](https://driada.readthedocs.io/en/latest/intense_mathematical_framework.html) -- mathematical framework behind INTENSE
- [examples/](../examples/) -- standalone Python scripts for production workflows
- [GitHub Issues](https://github.com/iabs-neuro/driada/issues) -- bug reports and questions