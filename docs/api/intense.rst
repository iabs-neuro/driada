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
   intense/base

Quick Links
-----------

**Main Analysis Pipelines**
   * :doc:`intense/pipelines` - High-level functions for significance testing
   * :func:`~driada.intense.pipelines.compute_cell_feat_significance` - Neuron-feature analysis
   * :func:`~driada.intense.pipelines.compute_feat_feat_significance` - Feature-feature dependencies
   * :func:`~driada.intense.pipelines.compute_cell_cell_significance` - Neuron-neuron connectivity
   * :func:`~driada.intense.pipelines.compute_embedding_selectivity` - Embedding dimension selectivity

**Statistical Tools**
   * :doc:`intense/stats` - Statistical testing and p-value computation
   * Distribution fitting and testing
   * Multiple comparison corrections

**Visualization**
   * :doc:`intense/visual` - Plotting functions for INTENSE results
   * Selectivity heatmaps and summaries
   * Neuron-feature pair visualization

**Advanced Analysis**
   * :doc:`intense/disentanglement` - Mixed selectivity disentanglement
   * Feature correlation analysis
   * Selectivity decomposition

**Core Implementation**
   * :doc:`intense/base` - Low-level computation functions
   * Mutual information computation
   * Delay optimization algorithms