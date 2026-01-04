<!-- IMPORTANT: When updating authors, sync from AUTHORS.yaml -->
---
title: 'DRIADA: A Unified Framework Bridging Single-Neuron Selectivity and Population Dynamics'
tags:
  - Python
  - neuroscience
  - calcium imaging
  - dimensionality reduction
  - mutual information
  - neural selectivity
authors:
  - name: Nikita A. Pospelov
    orcid: 0000-0000-0000-0000  # UPDATE with actual ORCID
    affiliation: 1
affiliations:
 - name: Faculty of Physics, Lomonosov Moscow State University, Moscow, Russia
   index: 1
date: 3 January 2026
bibliography: paper.bib
---

# Summary

DRIADA (Dimensionality Reduction for Integrated Activity Data) is a comprehensive Python framework that uniquely bridges single-neuron selectivity analysis with population-level dimensionality reduction. While traditional neuroscience analysis focuses either on individual neurons or population dynamics in isolation, DRIADA provides the first integrated workflow connecting these scales. The framework combines the INTENSE (Information-Theoretic Evaluation of Neuronal Selectivity) module for rigorous single-cell analysis with manifold learning techniques, enabling researchers to understand how individual neural selectivity gives rise to collective population representations.

# Statement of Need

Understanding neural computation requires analyzing both individual neurons and population dynamics. Current tools address these problems separately: packages like CaImAn [@giovannucci2019caiman] focus on calcium imaging preprocessing, PyEntropy [@ince2009python] analyzes single-neuron information content, and dimensionality reduction libraries implement population-level methods. However, no existing framework connects these analyses, leaving a critical gap: *How do individual neural selectivities contribute to population-level representations?*

DRIADA fills this gap by providing:

1. **INTENSE Module**: Rigorous single-neuron selectivity analysis using mutual information with novel two-stage statistical testing, achieving 100× computational efficiency while maintaining statistical rigor through Holm-Bonferroni correction [@holm1979simple]. Unlike correlation-based methods, INTENSE detects both linear and nonlinear relationships using Gaussian Copula Mutual Information [@ince2017statistical], handles temporal delays through optimal shift detection, and disentangles mixed selectivity when neurons respond to multiple correlated variables.

2. **Population Analysis**: Comprehensive dimensionality estimation and reduction toolkit implementing both classical (PCA, Factor Analysis) and modern manifold learning methods (Isomap, UMAP [@mcinnes2018umap], Diffusion Maps), with specialized neural network architectures for extracting latent variables from population activity.

3. **Integration Analysis**: Unique capability to map single-cell selectivity onto population manifolds, revealing how individual neurons contribute to collective representations—a workflow not available in any existing package.

4. **Validation Tools**: Synthetic data generators creating populations with known ground truth (head direction cells, place cells, mixed-selectivity neurons) enabling algorithm validation before application to experimental data.

DRIADA's integrated approach is particularly valuable for cognitive neuroscience (identifying task-relevant neural subspaces), systems neuroscience (bridging cellular and population descriptions), and AI interpretability (understanding representations in artificial neural networks).

# Key Features and Implementation

The software is designed with modularity and extensibility in mind:

- **Experiment Class**: Unified data container managing neural recordings, behavioral variables, and analysis results
- **INTENSE Statistical Engine**: Implements Gaussian Copula MI [@ince2017statistical] with dual-criterion significance testing (rank-based non-parametric + parametric gamma distribution fitting)
- **Spike Reconstruction**: Wavelet-based calcium transient detection with GPU acceleration support
- **Dimensionality Estimation**: Linear (PCA-based dimension, effective rank) and nonlinear (k-NN dimension, correlation dimension) methods
- **Manifold Learning**: Graph-based proximity methods with heat kernel affinities and giant component preprocessing
- **Publication Framework**: Built-in tools for generating publication-ready multi-panel figures with precise physical sizing

Performance is optimized through Numba JIT compilation for computational kernels, parallel processing support via joblib, and efficient sparse matrix operations. The codebase maintains 90% test coverage with comprehensive CI/CD workflows including unit tests, doctests, and documentation consistency checks.

# Comparison to Existing Tools

| Feature | DRIADA | CaImAn | PyEntropy | dPCA | scikit-learn |
|---------|--------|--------|-----------|------|--------------|
| Calcium spike extraction | ✓ | ✓ | — | — | — |
| Single-neuron selectivity | ✓ | — | ✓* | — | — |
| Rigorous MI statistics | ✓ | — | ✓* | — | — |
| Population dimensionality reduction | ✓ | — | — | ✓ | ✓ |
| **Single-cell ↔ Population integration** | **✓** | **—** | **—** | **—** | **—** |
| Synthetic validation data | ✓ | — | — | — | — |
| GPU acceleration | ✓ | ✓ | — | — | — |

*PyEntropy provides basic MI estimation but lacks DRIADA's two-stage testing, multiple comparison correction, optimal delay detection, and mixed selectivity analysis.

Notable differences: CaImAn excels at motion correction and source extraction from raw imaging movies but does not analyze neural selectivity or population structure. Demixed PCA (dPCA) [@kobak2016demixed] performs targeted dimensionality reduction for task variables but cannot identify which individual neurons encode those variables. DRIADA uniquely enables the complete workflow: preprocessing → single-neuron analysis → population analysis → integration.

# Research Applications

DRIADA has been developed through application to hippocampal and cortical calcium imaging data, enabling analysis of place cells, head-direction cells, and mixed-selectivity populations. The framework is currently used in ongoing research at Moscow State University investigating neural representations during spatial navigation tasks.

The software's synthetic data generators have proven valuable for algorithm validation, allowing researchers to test analysis pipelines on populations with known ground truth before applying them to experimental recordings. This "ground-truth-first" approach reduces analysis errors and increases confidence in findings.

# Acknowledgements

We acknowledge contributions from the neuroscience community for feedback on the INTENSE methodology and from users who tested early versions of the framework.

# References
