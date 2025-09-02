RSA Module
==========

.. automodule:: driada.rsa
   :no-members:
   :noindex:

Representational Similarity Analysis (RSA) for comparing neural representations
across conditions, time points, or brain regions.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   rsa/core
   rsa/integration
   rsa/visualization

Quick Links
-----------

**Core RDM Computation**
   * :func:`~driada.rsa.compute_rdm` - Basic RDM computation
   * :func:`~driada.rsa.compute_rdm_unified` - Unified API for RDM computation
   * :func:`~driada.rsa.rsa_compare` - Simplified comparison API
   * :doc:`rsa/core` - All core RSA functions

**RDM Comparison**
   * :func:`~driada.rsa.compare_rdms` - Compare two RDMs
   * :func:`~driada.rsa.bootstrap_rdm_comparison` - Statistical comparison
   * :doc:`rsa/core` - Comparison methods

**Integration with DRIADA**
   * :func:`~driada.rsa.compute_experiment_rdm` - RDM from Experiment objects
   * :func:`~driada.rsa.rsa_between_experiments` - Cross-experiment comparison
   * :doc:`rsa/integration` - Integration functions

**Visualization**
   * :func:`~driada.rsa.plot_rdm` - Plot single RDM
   * :func:`~driada.rsa.plot_rdm_comparison` - Plot RDM comparisons
   * :doc:`rsa/visualization` - Plotting utilities

Usage Example
-------------

.. code-block:: python

   from driada.rsa import compute_rdm, compare_rdms, plot_rdm
   
   # Compute RDM from neural data
   # Data shape: (n_neurons, n_conditions, n_timepoints)
   rdm = compute_rdm(neural_data, method='correlation')
   
   # Compare with model RDM
   similarity = compare_rdms(neural_rdm, model_rdm, method='spearman')
   
   # Visualize
   plot_rdm(rdm, title="Neural RDM")
   
   # For experiments
   from driada.rsa import compute_experiment_rdm
   exp_rdm = compute_experiment_rdm(
       exp,
       condition_labels=trial_types,
       time_window=(0, 500)
   )