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
    affiliation: 1
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
date: 1 March 2026
bibliography: paper.bib
---

# Summary

DRIADA (Dimensionality Reduction for Integrated Activity Data) is a Python
framework that links single-neuron selectivity analysis with population-level
dimensionality reduction for neural activity data. The framework accepts any
(n_units, n_frames) activity matrix -- calcium imaging, electrophysiology, or
artificial neural network activations -- alongside time-aligned behavioral
variables, and provides an integrated workflow from information-theoretic
significance testing of individual neurons through manifold extraction and
network analysis of the full population. DRIADA is designed for neuroscientists
who need to move beyond analyzing scales in isolation
[@saxena2019towards], enabling them to ask:
which neurons encode which behavioral variables, and how do those tuning
properties shape the population's collective geometry?

# Statement of Need

Neurons encode information through selective responses to stimuli, behaviors,
and cognitive states, but deciphering these codes is complicated by two
challenges. First, neurons often exhibit mixed selectivity -- responding to
combinations of variables rather than single features
[@rigotti2013importance; @tye2024mixed]. When behavioral variables covary, as
they typically do in naturalistic settings, it becomes difficult to determine
whether a neuron truly encodes a given variable or merely correlates with it
through a confound [@fusi2016why]. Modern automated behavior analysis tools
now extract hundreds of features from video recordings, producing rich yet
highly correlated datasets that exacerbate this identifiability problem.
Second, population manifold geometry depends on the tuning curves of
individual neurons [@chung2021neural], yet existing software addresses
single-neuron statistics and population dynamics separately
[@spalletti2022single], forcing researchers to assemble ad hoc pipelines with
no shared data model.

DRIADA targets experimentalists working with calcium imaging or
electrophysiology who need to (1) detect which neurons encode which variables
with rigorous statistical testing, (2) disentangle genuine mixed selectivity
from behavioral covariance, (3) extract low-dimensional population structure,
and (4) map single-cell properties onto population manifolds. Calcium imaging
presents particular analytical challenges: fluorescence signals are continuous
with slow indicator kinetics that create temporal smearing, and traditional
spike-based information measures are not directly applicable
[@climer2021information]. DRIADA operates on raw calcium traces without
requiring spike deconvolution, using estimators designed for continuous data.

# State of the Field

Several open-source packages address subsets of the neural analysis workflow.

**Information-theoretic toolboxes.** NIT [@maffulli2024nit] provides mutual
information estimation for spike trains and local field potentials but assumes
Poisson statistics, which are unsuitable for the continuous signals of calcium
imaging. MINT [@lorenz2025mint] analyzes information transmission at the
population level without single-neuron selectivity testing. FRITES
[@combrisson2022frites] implements information-based functional connectivity
for EEG/MEG data. IDTxl [@wollstadt2019idtxl] and HOI [@neri2024hoi] focus
on information dynamics and higher-order interactions in networks. The
Gaussian Copula MI framework [@ince2017statistical] enables efficient MI
estimation for continuous data but has not been adapted for calcium-specific
challenges: temporal delay optimization for indicator kinetics, circular-shift
permutations for autocorrelated signals, or mixed-selectivity
disentanglement. None of these toolboxes integrate single-neuron information
content with population manifold structure.

**Population-level tools.** CEBRA [@schneider2023learnable] produces
consistent embeddings across modalities but operates purely at the population
level without single-neuron statistics. CILDS [@koh2023dimensionality] jointly
deconvolves and reduces dimensionality for calcium imaging without selectivity
analysis. Demixed PCA [@kobak2016demixed] decomposes population activity by
task variable but lacks formal statistical tests for individual neuron
selectivity and is limited to categorical variables.

**Calcium imaging pipelines.** CaImAn [@giovannucci2019caiman] handles
upstream preprocessing (motion correction, source extraction, deconvolution)
and is complementary to DRIADA, which begins where preprocessing ends.

DRIADA fills the gap between these tools by connecting Gaussian Copula MI
with dimensionality reduction and network analysis under a shared data model.
We built DRIADA rather than extending an existing toolbox because the
integration requires a unified data model (`Experiment`, `TimeSeries`,
`MultiTimeSeries`) spanning both analysis scales -- something that cannot be
added as a plugin to packages designed around different abstractions.

# Software Design

DRIADA's architecture reflects four design decisions motivated by the
neuroscience workflow.

**Unified data model.** The `Experiment` class holds neural activity as a
`MultiTimeSeries` (n_units $\times$ n_frames), behavioral variables as
`TimeSeries` objects, and per-neuron `Neuron` objects that handle spike
reconstruction via synchrosqueezing wavelet transforms
[@muradeli2020ssqueezepy], kinetics optimization, and surrogate generation.
Each `TimeSeries` carries both the raw data and precomputed representations
required by downstream modules: a copula-normal transform for Gaussian Copula
MI, integer-coded values for discrete estimators, a boolean shuffle mask
defining valid circular-shift boundaries, and rich type metadata produced by
an automatic detection pipeline that classifies variables as continuous
(linear or circular), discrete (binary, categorical, count, or timeline), or
multivariate based on uniqueness ratio, gap statistics, and Von Mises
goodness-of-fit. Circular variables are additionally encoded as cos/sin pairs
on $\mathbb{R}^2$ to avoid wraparound artifacts in correlation-based
measures. This shared representation allows downstream modules -- INTENSE
selectivity testing, dimensionality reduction, network analysis, RSA
[@kriegeskorte2008representational] -- to operate on the same data without
format conversion.

**FFT-accelerated permutation testing.** The INTENSE module uses circular
time-shift permutations to assess whether each neuron's mutual information
with each behavioral variable exceeds chance. Circular shifts preserve the
temporal autocorrelation structure of both calcium and behavioral signals --
a critical requirement, since naive permutation tests produce overwhelming
false positive rates on autocorrelated data. The computational bottleneck is
addressed by precomputing per-signal FFTs and combining them via
cross-correlation, reducing redundant computation from O(n_neurons $\times$
n_features) to O(n_neurons + n_features) unique FFTs. A two-stage design
(100 permutations for screening, 10,000 for validation) with Holm-Bonferroni
correction [@holm1979simple] maintains statistical rigor while keeping
computation tractable. The MI estimator is Gaussian Copula MI
[@ince2017statistical], which captures nonlinear dependencies that
correlation-based methods miss -- in hippocampal data, MI-based testing
detected 2.2$\times$ more selective neurons than correlation applied within
the same pipeline.

**Mixed-selectivity disentanglement.** When a neuron appears selective to
multiple correlated variables, INTENSE uses conditional mutual information to
test whether selectivity to each variable persists after controlling for the
others, and interaction information to classify feature pairs as redundant or
synergistic [@rigotti2013importance]. In hippocampal recordings, this
procedure attributed roughly one-third of apparent multi-variable associations
to behavioral covariance rather than genuine mixed encoding.

**Modularity across analysis scales.** Five modules operate on the shared
`Experiment` object through pipeline functions. The dimensionality reduction
module wraps the activity matrix in an `MVData` object that provides a
uniform interface to 15 embedding methods (PCA, Isomap, UMAP
[@mcinnes2018umap], diffusion maps, autoencoders, and others)
[@cunningham2014dimensionality] with quality metrics (k-NN preservation,
trustworthiness, continuity, stress) for method comparison, plus intrinsic
dimensionality estimators for characterizing manifold complexity
[@jazayeri2021interpreting]. Graph-based
dimensionality reduction methods return a `ProximityGraph` that inherits from `Network`, so the
full spectral and community-detection toolkit of the network module applies
directly to DR-derived graphs. The network module itself constructs
functional connectivity graphs from pairwise MI
(`compute_cell_cell_significance`) and provides structural analysis (degree,
clustering, Louvain communities), spectral analysis (eigendecomposition,
inverse participation ratio, communicability, Von Neumann entropy), and
null-model comparison via degree-preserving randomization. An integration
module bridges scales: `compute_embedding_selectivity` runs INTENSE on
embedding components to identify which neurons drive each manifold dimension,
and leave-one-out analysis quantifies each neuron's contribution to manifold
structure, enabling direct comparison with single-neuron MI. A
representational similarity analysis (RSA) module computes representational
dissimilarity matrices from neural data or embeddings, supporting
cross-region and cross-session comparisons with bootstrap significance
testing. The same architecture supports artificial neural networks -- RNN
hidden-unit activations load identically to calcium traces -- following the
cross-domain approach of @mante2013context.

# Research Impact Statement

DRIADA formalizes analysis methods developed over several years of
neuroscience research. The framework has been applied to hippocampal calcium
imaging, revealing fast tuning dynamics of place cells during free exploration
[@Sotskov2022], and to dimensionality estimation of hippocampal population
activity, demonstrating behavioral correlates [@Pospelov2024]. The
dimensionality reduction module has been used for fMRI resting-state analysis
[@Pospelov2021], the network module for structural connectome
characterization [@Bobyleva2025] and spectral entropy analysis of functional
brain networks [@Pospelov2022]. Demonstrating substrate-agnostic
applicability, @Kononov2025 used DRIADA to analyze recurrent neural network
activations, revealing hybrid attractor dynamics in reinforcement learning
agents.

The package includes six tutorial notebooks executable on Google Colab,
23 standalone example scripts, and a synthetic data generator producing
populations with known ground truth for validating analysis pipelines.
Validation on synthetic datasets with known selectivity demonstrates robust
detection across a wide range of signal-to-noise ratios and response
reliability conditions. The test suite contains 1,880+ tests running on
Linux, macOS, and Windows across Python 3.9--3.13. Full API documentation is
hosted at [driada.readthedocs.io](https://driada.readthedocs.io).

# AI Usage Disclosure

The core algorithms and library code were written manually over the course
of DRIADA's multi-year development. Starting from mid-2025, generative AI
tools (Anthropic Claude, Opus 4 and Sonnet 4 families) were used for
partial assistance with test generation, documentation writing, and
development acceleration. All AI-assisted outputs were reviewed, tested,
and validated by the authors. The scientific methodology, algorithmic
design, and research applications are the original intellectual
contribution of the authors. This paper was drafted with partial AI
assistance and will be revised by the authors.

# Acknowledgements

This work was supported by the Non-Commercial Foundation for Support of
Science and Education "INTELLECT". We acknowledge feedback from the
neuroscience community on the INTENSE methodology and from users who tested
early versions of the framework.

<!-- CRediT Author Contributions (not rendered in JOSS paper — for internal records)

N.P.: Conceptualization, Methodology, Software, Writing – Original Draft.
V.P.:
O.R.:
A.I.:
V.S.:
M.O.:
K.T.:
O.I.:
V.A.:
K.A.:

Roles: Conceptualization, Methodology, Software, Validation,
Formal Analysis, Investigation, Data Curation, Writing – Original Draft,
Writing – Review & Editing, Visualization, Supervision,
Project Administration, Funding Acquisition
-->

# References
