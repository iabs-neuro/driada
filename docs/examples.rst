Examples
========

Standalone scripts covering every major DRIADA capability.
Each example generates synthetic data internally — no external files needed.
Run any script directly: ``python examples/<folder>/<script>.py``

Source code: `examples/ on GitHub <https://github.com/iabs-neuro/driada/tree/main/examples>`__

.. contents:: On this page
   :local:
   :depth: 2


INTENSE — Selectivity Detection
--------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `intense_basic_usage <https://github.com/iabs-neuro/driada/tree/main/examples/intense_basic_usage>`__
     - Minimal INTENSE workflow: generate a synthetic population, run two-stage
       significance testing, extract per-neuron results, and visualize tuning.
   * - `full_intense_pipeline <https://github.com/iabs-neuro/driada/tree/main/examples/full_intense_pipeline>`__
     - Complete pipeline across all feature types — circular (head direction),
       spatial (position), linear (speed), and discrete (events) — with
       ground-truth validation and detection metrics.
   * - `mixed_selectivity <https://github.com/iabs-neuro/driada/tree/main/examples/mixed_selectivity>`__
     - Disentangle mixed selectivity patterns using multivariate features,
       synergy/redundancy decomposition, and interaction information.


Dimensionality Reduction
-------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `compare_dr_methods <https://github.com/iabs-neuro/driada/tree/main/examples/compare_dr_methods>`__
     - Systematic comparison of dimensionality reduction methods (PCA, Isomap, UMAP, t-SNE, etc.)
       on synthetic datasets with quality metrics (k-NN preservation,
       trustworthiness, continuity, stress) and timing benchmarks.
   * - `dr_simplified_api <https://github.com/iabs-neuro/driada/tree/main/examples/dr_simplified_api>`__
     - Quick-start guide for DRIADA's dimensionality reduction API: ``MVData``, automatic parameter
       handling, custom metrics, and method-specific configurations.
   * - `dr_sequence <https://github.com/iabs-neuro/driada/tree/main/examples/dr_sequence>`__
     - Compare direct UMAP vs PCA-then-UMAP sequential dimensionality reduction
       on synthetic neural manifolds with preservation metrics.
   * - `autoencoder_dr <https://github.com/iabs-neuro/driada/tree/main/examples/autoencoder_dr>`__
     - Neural-network-based dimensionality reduction on circular manifold data: standard autoencoder
       with ``continue_learning``, Beta-VAE, and PCA comparison.


Dimensionality Estimation
--------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `circular_manifold <https://github.com/iabs-neuro/driada/tree/main/examples/circular_manifold>`__
     - Extract 1D circular structure from head direction cells using multiple
       dimensionality reduction methods and intrinsic dimensionality estimators (correlation dimension,
       geodesic dimension, participation ratio).


Integration — INTENSE + Dimensionality Reduction
--------------------------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `intense_dr_pipeline <https://github.com/iabs-neuro/driada/tree/main/examples/intense_dr_pipeline>`__
     - Use INTENSE to identify spatially selective neurons, then compare dimensionality reduction
       quality (decoding R², distance correlation) using all neurons vs the
       INTENSE-selected subset.
   * - `loo_dr_analysis <https://github.com/iabs-neuro/driada/tree/main/examples/loo_dr_analysis>`__
     - Leave-one-out neuron importance analysis: remove each neuron, measure
       manifold degradation, and compare structural importance with INTENSE
       selectivity scores.
   * - `functional_organization <https://github.com/iabs-neuro/driada/tree/main/examples/functional_organization>`__
     - Reverse the usual direction: treat embedding components as features,
       run INTENSE on them, and discover which neurons drive which manifold
       dimensions.


Network Analysis
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `network_analysis <https://github.com/iabs-neuro/driada/tree/main/examples/network_analysis>`__
     - Compute cell-cell functional connectivity via pairwise MI significance,
       build a ``Network`` object, and analyze graph structure (degree
       distribution, communities, spectral properties).
   * - `network_spectrum <https://github.com/iabs-neuro/driada/tree/main/examples/network_spectrum>`__
     - Spectral analysis toolkit: eigendecomposition of adjacency and Laplacian
       matrices, inverse participation ratio, spectral entropy, communicability,
       and Gromov hyperbolicity.


RSA — Representational Similarity
-----------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `rsa <https://github.com/iabs-neuro/driada/tree/main/examples/rsa>`__
     - Compute RDMs from stimulus-selective populations, compare representations
       across regions and sessions with multiple distance metrics, and run
       bootstrap significance testing.


Synthetic Data
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `rnn_activations <https://github.com/iabs-neuro/driada/tree/main/examples/rnn_activations>`__
     - Full DRIADA pipeline on simulated RNN activations: generate behavioral
       inputs, simulate a driven RNN, load into an ``Experiment``, and run
       INTENSE + dimensionality reduction + network analysis. Demonstrates that DRIADA works with
       any ``(n_units, n_frames)`` data, not just calcium imaging.


Neuron — Spike Reconstruction & Quality
-----------------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `neuron_basic_usage <https://github.com/iabs-neuro/driada/tree/main/examples/neuron_basic_usage>`__
     - Core ``Neuron`` class: generate synthetic calcium signals, reconstruct
       spikes (wavelet method), optimize rise/decay kinetics, and compute
       signal quality metrics.
   * - `spike_reconstruction <https://github.com/iabs-neuro/driada/tree/main/examples/spike_reconstruction>`__
     - Compare wavelet vs threshold spike reconstruction on calcium traces with
       overlapping events; analyze detection accuracy differences.
   * - `threshold_vs_wavelet_optimization <https://github.com/iabs-neuro/driada/tree/main/examples/spike_reconstruction>`__
     - Benchmark reconstruction modes: default kinetics, optimized kinetics,
       and iterative detection (n_iter=2, 3) with performance tradeoffs.


Utilities & Data Loading
-------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - `data_loading <https://github.com/iabs-neuro/driada/tree/main/examples/data_loading>`__
     - Load real recording data into DRIADA: numpy arrays and feature
       annotations → ``Experiment`` object with all downstream analysis
       enabled.
   * - `signal_association <https://github.com/iabs-neuro/driada/tree/main/examples/signal_association>`__
     - Information-theory primitives: pairwise MI (GCMI vs KSG estimators),
       time-delayed MI, conditional MI, and interaction information
       (synergy/redundancy).
   * - `behavior_relations <https://github.com/iabs-neuro/driada/tree/main/examples/behavior_relations>`__
     - Feature-feature significance testing with FFT-based circular shuffling
       for behavioral variable correlations, independent of neural data.
   * - `spatial_analysis <https://github.com/iabs-neuro/driada/tree/main/examples/spatial_analysis>`__
     - Spatial data visualization: trajectory maps, occupancy maps, rate maps,
       and calcium traces for place cell analysis.
   * - `visual_utils <https://github.com/iabs-neuro/driada/tree/main/examples/visual_utils>`__
     - Publication-ready figure generation using DRIADA's visual utilities:
       embedding comparison plots, selectivity summaries, and consistent
       styling.
