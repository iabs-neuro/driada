---
title: 'DRIADA: A Substrate-Agnostic Framework Bridging Single-Neuron Selectivity and Population Dynamics'
tags:
  - Python
  - neuroscience
  - calcium imaging
  - dimensionality reduction
  - mutual information
  - neural selectivity
  - artificial neural networks
  - interpretability
authors:
  - name: Nikita A. Pospelov
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Faculty of Physics, Lomonosov Moscow State University, Moscow, Russia
   index: 1
date: 3 January 2026
bibliography: paper.bib
---

# Summary

DRIADA (Dimensionality Reduction for Integrated Activity Data) is a Python framework that connects single-neuron selectivity analysis with population-level dimensionality reduction. The framework operates on neural activity data regardless of its source—calcium imaging recordings, electrophysiology, or artificial neural network activations. While neuroscience has increasingly adopted population-level approaches [@saxena2019towards], understanding how individual neuron properties contribute to collective representations remains methodologically challenging [@spalletti2022single]. DRIADA provides an integrated workflow linking information-theoretic selectivity testing at the single-cell level with manifold learning at the population level.

# Statement of Need

Analyzing neural computation requires tools that span both individual neurons and population dynamics, yet existing software addresses these scales separately. @quirogapanzeri2009 noted that "the complementary knowledge offered by decoding and information theory has not been exploited enough in neuroscience." This integration gap persists: @chung2021neural demonstrate that population manifold geometry "depends on the tuning curves of all neurons," explicitly linking single-neuron selectivity to population structure. No existing software provides a complete workflow to operationalize this connection.

**Information-theoretic toolboxes** provide mutual information estimation but do not integrate with dimensionality reduction. @climer2021information identify that traditional spike-based information metrics "were not designed for the slow timescales and variable amplitudes typical of functional fluorescence recordings," motivating the need for continuous estimators like Gaussian Copula MI [@ince2017statistical]. NIT [@maffulli2022nit] focuses on spike trains and local field potentials with Poisson-based estimators, while MINT [@lorenz2025mint] addresses information flow across brain areas at the population level without single-neuron selectivity testing. FRITES [@combrisson2022frites] implements information-based functional connectivity for EEG/MEG/sEEG data but targets different recording modalities than calcium imaging. None provide integrated workflows connecting single-cell information content to population manifold structure.

**Population dimensionality reduction tools** extract latent representations but lack single-neuron selectivity statistics. CEBRA [@schneider2023learnable] produces consistent embeddings across modalities but operates purely at the population level. CILDS [@koh2023dimensionality] performs joint deconvolution and dimensionality reduction for calcium imaging without selectivity analysis. Demixed PCA [@kobak2016demixed] provides neuron contribution weights but lacks formal statistical tests for individual selectivity and is limited to categorical variables.

**DRIADA addresses this gap through:**

1. **Information Module**: Built on Gaussian Copula Mutual Information [@ince2017statistical], this module provides single-neuron selectivity analysis with two-stage statistical testing and Holm-Bonferroni correction [@holm1979simple]. The implementation supports interaction information and other multivariate measures through an efficient GCMI-based framework. Unlike correlation methods, it detects nonlinear relationships, handles temporal delays, and disentangles mixed selectivity when neurons respond to multiple correlated variables [@rigotti2013importance; @fusi2016why].

2. **Dimensionality Reduction Module**: Implements both classical methods (PCA, Factor Analysis) and manifold learning approaches (Isomap, UMAP [@mcinnes2018umap], Diffusion Maps). Includes a comprehensive autoencoder system with configurable architectures for neural network-based dimensionality reduction. Dimensionality estimation methods include PCA-based dimension, effective rank, k-NN dimension, and correlation dimension.

3. **Network Analysis Module**: Tools for analyzing functional connectivity structure in neural populations using graph-theoretic methods, including heat kernel affinities and giant component analysis.

4. **Signal Processing**: Calcium transient detection using synchrosqueezing wavelet transforms [@muradeli2020ssqueezepy] with GPU acceleration support.

5. **Substrate-Agnostic Design**: The framework analyzes activity from biological recordings and artificial neural networks identically. This follows @mante2013context, who applied identical analyses to prefrontal cortex and RNNs. Cross-domain tools such as RSA [@kriegeskorte2008representational] and CKA [@kornblith2019similarity] operate at the population level; DRIADA extends this to single-neuron selectivity testing across substrates.

6. **Validation Tools**: Synthetic data generators produce populations with known ground truth (head direction cells, place cells, mixed-selectivity neurons) for algorithm validation.

# Key Features and Implementation

DRIADA employs a modular architecture centered on three core data structures: the `Experiment` class manages multi-neuron recordings and behavioral variables, individual `Neuron` objects handle spike-calcium deconvolution and event detection, and `TimeSeries`/`MultiTimeSeries` objects represent neural and behavioral variables with automatic type detection (discrete vs. continuous). The analysis pipeline integrates single-neuron and population-level methods through a unified interface.

**Information-theoretic analysis** leverages multiple mutual information estimators [@ince2017statistical] automatically selected based on data type: Gaussian Copula MI for continuous data, k-nearest neighbor (KSG) for non-parametric estimation, and discrete MI for categorical variables. Two-stage significance testing (100 permutations for screening, 10,000 for validation) with Holm-Bonferroni correction [@holm1979simple] ensures statistical rigor while maintaining computational efficiency. The framework supports conditional MI, interaction information, and redundancy/synergy decomposition for multivariate analysis.

**Dimensionality reduction** operates through `MVData` containers into a unified interface supporting 12+ methods [@cunningham2014dimensionality; @vyas2020computation]: linear (PCA, MDS), manifold (Isomap, LLE, UMAP [@mcinnes2018umap]), spectral (Laplacian Eigenmaps, Diffusion Maps), and neural network-based (autoencoders, VAEs with flexible loss composition). Each method includes quality metrics (reconstruction error, embedding stress, neighborhood preservation) for validation [@jazayeri2021interpreting].

**Integration capabilities** uniquely map single-cell selectivity onto population manifolds, enabling researchers to identify which neurons encode task variables and how their tuning properties shape collective geometry—addressing the methodological gap identified by @spalletti2022single between single-neuron and population approaches.

Performance optimization employs conditional Numba JIT compilation (27 functions across information theory and signal processing), joblib-based parallelization with automatic backend selection, and optional PyTorch GPU acceleration. The codebase maintains >90% test coverage for core computational modules (information theory, INTENSE, network analysis) with comprehensive CI/CD workflows across Linux, macOS, and Windows.

# Research Applications

DRIADA formalizes and extends analysis methods developed and refined over several years of neuroscience research. The framework has been applied to calcium imaging analysis of hippocampal place cells, revealing fast tuning dynamics during free exploration [@Sotskov2022]. Dimensionality estimation methods from DRIADA have demonstrated behavioral correlates of neural population activity [@Pospelov2024] and enabled analysis of structural connectome architecture [@Bobyleva2025]. The framework's dimensionality reduction toolkit has been applied to fMRI resting-state analysis [@Pospelov2021] and functional connectome characterization [@Pospelov2022].

Demonstrating substrate-agnostic applicability, DRIADA has been used to analyze recurrent neural networks, revealing hybrid attractor architectures in reinforcement learning agents [@Kononov2024]. This cross-domain capability positions DRIADA as both a neuroscience analysis tool and an artificial neural network interpretability framework.

A comprehensive list of publications using DRIADA is maintained at https://github.com/iabs-neuro/driada/blob/main/PUBLICATIONS.md.

# Acknowledgements

We acknowledge feedback from the neuroscience community on the INTENSE methodology and from users who tested early versions of the framework.

# References
