---
title: 'DRIADA: A Python Framework Bridging Single-Neuron Selectivity and Population Dynamics'
tags:
  - Python
  - neuroscience
  - calcium imaging
  - dimensionality reduction
  - mutual information
  - neural selectivity
  - neural networks
authors:
  - name: Nikita Pospelov
    orcid: 0000-0001-6637-2120
    corresponding: true
    affiliation: 1
  - name: Victor Plusnin
    orcid: 0000-0001-9784-8283
    affiliation: 1
  - name: Olga Rogozhnikova
    orcid: 0000-0002-2540-4402
    affiliation: 1
  - name: Anna Ivanova
    orcid: 0000-0002-2454-1288
    affiliation: 1
  - name: Vladimir Sotskov
    orcid: 0000-0002-1729-5744
    affiliation: 3
  - name: Margarita Orobets
    orcid: 0009-0002-4231-5329
    affiliation: 1
  - name: Ksenia Toropova
    orcid: 0000-0003-3081-2133
    affiliation: 1
  - name: Olga Ivashkina
    orcid: 0000-0002-1540-7677
    affiliation: 1
  - name: Vladik Avetisov
    orcid: 0000-0002-2516-8868
    affiliation: 2
  - name: Konstantin Anokhin
    orcid: 0000-0003-4437-6002
    affiliation: 1
affiliations:
  - name: Institute for Advanced Brain Studies, Lomonosov Moscow State University, Moscow, Russia
    index: 1
  - name: Semenov Institute of Chemical Physics, Russian Academy of Sciences, Moscow, Russia
    index: 2
  - name: Center for Interdisciplinary Research in Biology, Collège de France, Paris, France
    index: 3
date: 1 March 2026
bibliography: paper.bib
---

# Summary

DRIADA (Dimensionality Reduction for Integrated Activity DAta) is a Python
framework for integrated analysis of neural activity across three scales:
single-neuron selectivity, population manifold structure, and functional
network organization. The framework supports calcium imaging,
electrophysiology, and artificial neural network activations, alongside
time-aligned behavioral variables, and provides a unified path from
information-theoretic selectivity testing to manifold extraction and network
analysis. DRIADA automatically detects variable types (continuous, discrete,
circular, and multivariate) and selects appropriate information-theoretic
estimators, making these analyses accessible to users without specialized
expertise in estimator choice. By combining analyses that are often performed
in isolation [@saxena2019towards], DRIADA helps users test how neuron-level
tuning relates to the collective geometry and network structure of population
activity.

# Statement of Need

Neural data analysis increasingly requires linking multiple scales of
organization: population manifold geometry is shaped by single-neuron tuning
properties [@chung2021neural], yet existing software often treats
single-neuron statistics and population dynamics as separate tasks
[@spalletti2022single], forcing researchers to assemble ad hoc pipelines. As
experiments increase in neuron counts and behavioral complexity
[@rigotti2013importance; @fusi2016why; @tye2024mixed], the practical need for
a unified framework connecting these scales becomes increasingly pressing.

DRIADA is intended for experimental researchers who need a single software
workflow to identify which neurons encode which variables, extract
low-dimensional population structure, relate single-cell properties to
manifold geometry, and analyze functional networks. By reducing tool
switching and repeated data reformatting, DRIADA supports cross-scale
analyses that are difficult to implement reproducibly with fragmented
toolchains. This need is especially strong for calcium imaging data, where
fluorescence signals are continuous and shaped by slow indicator kinetics,
making spike-oriented information measures not directly applicable
[@climer2021information]. DRIADA addresses this by providing estimators for
continuous activity signals and permutation schemes that preserve temporal
autocorrelation.

# State of the Field

Several packages address subsets of the neural analysis workflow. Table 1
summarizes key differences.

| Feature | DRIADA | NIT | MINT | FRITES | CEBRA | dPCA |
|---------|--------|-----|------|--------|-------|------|
| Role of individual neurons | $\checkmark$ | $\checkmark$ | — | — | — | $\checkmark$ |
| Autocorrelation-aware p-values | $\checkmark$ | — | — | — | — | — |
| Diverse variable types | $\checkmark$ | $\checkmark$ | $\checkmark$ | — | $\checkmark$ | — |
| Population coding analysis | $\checkmark$ | — | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Network analysis | $\checkmark$ | — | — | $\checkmark$ | — | — |

: Comparison of DRIADA with related tools. NIT [@maffulli2024nit] provides MI
estimation; MINT [@lorenz2025mint] analyzes population-level information
transmission; FRITES [@combrisson2022frites] targets neurophysiological
functional connectivity; CEBRA [@schneider2023learnable] learns population
embeddings; dPCA [@kobak2016demixed] decomposes activity by task variable.

**Information-theoretic toolboxes.** NIT [@maffulli2024nit] provides mutual
information estimation including GCMI for continuous signals and extensive
bias corrections, but lacks autocorrelation-aware permutation testing,
automatic variable type detection, or integration with dimensionality
reduction and network analysis. MINT [@lorenz2025mint] analyzes
population-level information transmission without single-neuron selectivity.
FRITES [@combrisson2022frites] targets EEG/MEG functional connectivity. IDTxl
[@wollstadt2019idtxl] and HOI [@neri2024hoi] focus on information dynamics
and higher-order interactions. None integrate single-neuron information with
population manifold structure.

**Population-level tools.** CEBRA [@schneider2023learnable] produces
consistent embeddings but operates purely at the population level. CILDS
[@koh2023dimensionality] jointly deconvolves and reduces dimensionality
without selectivity analysis. Demixed PCA [@kobak2016demixed] decomposes
activity by task variable and provides encoder weights quantifying neuron
contributions, but lacks formal statistical tests (p-values with multiple
comparison correction) for individual selectivity and handles only
categorical variables.

**Calcium imaging pipelines.** CaImAn [@giovannucci2019caiman] handles
upstream preprocessing and is complementary to DRIADA, which begins where
preprocessing ends.

DRIADA fills the gap by connecting information-theoretic selectivity testing
with dimensionality reduction and network analysis under a unified data model
spanning all three scales. We built DRIADA rather than extending an existing
toolbox because this integration requires a shared data model spanning both
analysis scales—something that cannot be added as a plugin to packages
designed around different abstractions.

# Software Design

DRIADA is organized around a shared data model that enables seamless
transitions between analysis scales.

**Unified data representation.** The central data structure holds neural
activity matrices, time-aligned behavioral variables with automatic type
inference (continuous, discrete, circular, multivariate), and per-neuron
metadata. Precomputed representations (copula transforms, discretized values,
valid shuffle boundaries) allow all downstream modules to operate without
format conversion—researchers load data once and access any analysis.

**Modular analysis pipeline.** Five modules operate on the shared data: (1)
single-neuron selectivity testing using information-theoretic measures with
GCMI [@ince2017statistical] and KSG [@kraskov2004estimating] estimators;
(2) dimensionality reduction [@cunningham2014dimensionality] with 15
methods spanning linear (PCA), manifold (UMAP [@mcinnes2018umap], diffusion
maps), and autoencoder approaches, plus intrinsic dimensionality estimators
[@jazayeri2021interpreting]; (3) network analysis providing functional
connectivity, spectral decomposition, and analysis tools; (4)
representational similarity analysis [@kriegeskorte2008representational]; and
(5) scale integration functions.

**FFT-accelerated permutation testing.** Circular time-shift
permutations—rather than naive shuffles—preserve the temporal
autocorrelation of calcium and behavioral signals, avoiding inflated
false-positive rates. Because circular shifts correspond to circular
cross-correlations, the convolution theorem computes MI at all $n$ possible
shifts in $O(n \log n)$ rather than $O(n_{\text{sh}} \cdot n)$ for
$n_{\text{sh}}$ individual shuffles, making large-scale permutation testing
tractable. GCMI is the default estimator because its closed-form expression
integrates directly into this FFT pipeline, enabling real-scale analysis.
KSG captures arbitrary nonlinear dependencies at higher computational
cost and is data-demanding, while GCMI captures monotonic nonlinearities only, but
is very efficient in production.

**Cross-module integration.** The shared data model creates natural
connections between modules. Graph-based dimensionality reduction methods
return proximity graphs that inherit the full network analysis toolkit, so
population structure obtained via manifold learning can be analyzed as a
network without conversion. Multiple latent representations can be computed,
compared, and mapped back to individual neuron roles within a single
experiment. The same MI-based significance engine underlies cell--variable,
cell--cell, and variable--variable analysis, providing a uniform statistical
framework for both neural and behavioral analyses.

**Scale bridging.** The integration module connects single-neuron properties
to population structure: selectivity analysis can target embedding components
to identify which neurons drive each manifold dimension, and leave-one-out
procedures quantify individual contributions to collective geometry. This
workflow—from neuron-level encoding through population manifolds to
network organization—addresses a gap in existing tools, which typically
handle these scales separately.

**Substrate generality.** The same architecture supports biological recordings
(calcium imaging, electrophysiology) and artificial systems (RNN activations),
following the cross-domain approach of @mante2013context and demonstrated on
recurrent networks by @Kononov2025.

# Research Impact Statement

DRIADA formalizes methods developed over several years of research: fast
tuning dynamics of hippocampal place cells [@Sotskov2022], dimensionality
estimation with behavioral correlates [@Pospelov2024], fMRI resting-state
analysis [@Pospelov2021], structural connectome characterization
[@Bobyleva2025], and spectral entropy of functional networks [@Pospelov2022].
Demonstrating substrate-agnostic applicability, @Kononov2025 analyzed recurrent
neural network activations.

The package includes six tutorial notebooks (Google Colab compatible), 23
example scripts, and synthetic data generators for validation. The test suite
contains 1,880+ tests across Python 3.9--3.13. Documentation is at
[driada.readthedocs.io](https://driada.readthedocs.io).

# AI Usage Disclosure

Core algorithms were written manually over DRIADA's multi-year development.
From mid-2025, generative AI (Anthropic Claude Opus 4 and Sonnet 4 family) assisted
with test generation, documentation, and development acceleration. All
AI-assisted outputs were reviewed, tested, and validated by the authors.
Scientific methodology, algorithmic design, and research applications are the
authors' original contribution. This paper was drafted with partial AI
assistance and revised by the authors.

# Acknowledgements

This work was supported by the Non-Commercial Foundation for Support of
Science and Education "INTELLECT".

<!-- CRediT Author Contributions (not rendered in JOSS paper — for review records)

N.P.: Conceptualization, Methodology, Software, Validation, Formal Analysis,
      Data Curation, Writing – Original Draft, Writing – Review & Editing,
      Visualization, Project Administration.
V.P.: Methodology, Software, Validation, Investigation, Data Curation.
O.R.: Validation, Investigation, Resources, Data Curation.
A.I.: Validation, Investigation, Resources, Data Curation.
V.S.: Methodology, Software, Investigation.
M.O.: Software, Validation, Investigation.
K.T.: Conceptualization, Investigation, Resources, Data Curation,
      Writing – Review & Editing.
O.I.: Conceptualization, Investigation, Resources, Data Curation,
      Writing – Review & Editing, Project Administration.
V.A.: Conceptualization, Methodology, Supervision.
K.A.: Conceptualization, Supervision, Funding Acquisition.
-->

# References
