INTENSE Module
==============

.. automodule:: driada.intense
   :no-members:
   :noindex:

Information-Theoretic Evaluation of Neuronal Selectivity (INTENSE) provides tools for analyzing
how individual neurons encode behavioral and task variables using mutual information.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   intense/pipelines
   intense/stats
   intense/visual
   intense/disentanglement
   intense/delay
   intense/validation
   intense/fft
   intense/correction
   intense/base
   ../intense_mathematical_framework

Quick Links
-----------

**Main Analysis Pipelines**
   * :doc:`intense/pipelines` - High-level functions for significance testing
   * :func:`~driada.intense.pipelines.compute_cell_feat_significance` - Neuron-feature analysis
   * :func:`~driada.intense.pipelines.compute_feat_feat_significance` - Feature-feature dependencies
   * :func:`~driada.intense.pipelines.compute_cell_cell_significance` - Neuron-neuron connectivity
   * :func:`~driada.intense.pipelines.compute_embedding_selectivity` - Embedding dimension selectivity

**Statistical Tools**
   * :doc:`intense/stats` - Distribution fitting, testing, and p-value computation
   * :doc:`intense/correction` - Multiple comparison corrections

**Visualization**
   * :doc:`intense/visual` - Selectivity heatmaps, summaries, and neuron-feature pair plots

**Advanced Analysis**
   * :doc:`intense/disentanglement` - Mixed selectivity disentanglement and feature correlation analysis

**Delay Optimization**
   * :doc:`intense/delay` - Temporal delay optimization between time series

**Input Validation**
   * :doc:`intense/validation` - Time series and parameter validation

**FFT Infrastructure**
   * :doc:`intense/fft` - FFT type dispatch and MI caching

**Multiple Comparison Correction**
   * :doc:`intense/correction` - P-value threshold calculation (Holm, FDR, Bonferroni)

**Core Implementation**
   * :doc:`intense/base` - Low-level MI computation functions