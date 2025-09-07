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
   * :func:`~driada.rsa.core.compute_rdm` - Basic RDM computation
   * :func:`~driada.rsa.compute_rdm_unified` - Unified API for RDM computation
   * :func:`~driada.rsa.rsa_compare` - Simplified comparison API
   * :doc:`rsa/core` - All core RSA functions

**RDM Comparison**
   * :func:`~driada.rsa.core.compare_rdms` - Compare two RDMs
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
   import numpy as np
   
   # Generate example neural data
   # Shape: (n_neurons, n_conditions, n_timepoints)
   n_neurons, n_conditions, n_timepoints = 50, 10, 100
   neural_data = np.random.randn(n_neurons, n_conditions, n_timepoints)
   
   # Prepare neural patterns: average activity per condition
   # If you have 3D data (n_neurons, n_conditions, n_timepoints), average over time
   # patterns shape: (n_conditions, n_neurons)
   patterns = np.mean(neural_data, axis=2)  # Average over time
   
   # Compute RDM from neural patterns
   rdm = compute_rdm(patterns, metric='correlation')
   
   # Compare with model RDM (must be same size)
   model_rdm = np.random.rand(patterns.shape[0], patterns.shape[0])
   np.fill_diagonal(model_rdm, 0)  # Diagonal should be 0
   similarity = compare_rdms(rdm, model_rdm, method='spearman')
   
   # Visualize
   plot_rdm(rdm, title="Neural RDM")
   
   # For experiments
   from driada.rsa import compute_experiment_rdm
   from driada.experiment import load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Method 1: Use behavioral variable as conditions
   # Use trial_type which has discrete values
   rdm, item_names = compute_experiment_rdm(
       exp,
       items='trial_type',  # Use behavioral variable name
       data_type='calcium',
       metric='correlation'
   )