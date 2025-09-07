Dimensionality Estimation Module
================================

.. automodule:: driada.dimensionality
   :no-members:
   :noindex:

Methods for estimating the intrinsic dimensionality of high-dimensional datasets,
including linear methods (PCA-based), effective dimensionality, and nonlinear approaches.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   dimensionality/linear
   dimensionality/effective
   dimensionality/intrinsic

Quick Links
-----------

**Linear Methods**
   * :func:`~driada.dimensionality.linear.pca_dimension` - PCA variance threshold
   * :func:`~driada.dimensionality.linear.pca_dimension_profile` - Full PCA profile
   * :func:`~driada.dimensionality.linear.effective_rank` - Matrix effective rank
   * :doc:`dimensionality/linear` - All linear methods

**Effective Dimensionality**
   * :func:`~driada.dimensionality.effective.eff_dim` - Participation ratio method
   * :doc:`dimensionality/effective` - Rényi entropy-based estimation

**Intrinsic Dimensionality**
   * :func:`~driada.dimensionality.intrinsic.nn_dimension` - k-NN based estimation
   * :func:`~driada.dimensionality.intrinsic.correlation_dimension` - Correlation dimension
   * :func:`~driada.dimensionality.intrinsic.geodesic_dimension` - Geodesic-based estimation
   * :doc:`dimensionality/intrinsic` - Nonlinear methods

Usage Example
-------------

.. code-block:: python

   from driada.dimensionality import (
       pca_dimension, eff_dim, nn_dimension
   )
   from driada.experiment import load_demo_experiment

   # Load sample experiment
   exp = load_demo_experiment()

   # Prepare data (n_samples, n_features)
   neural_data = exp.calcium.scdata.T
   
   # Add small noise to avoid duplicate points for nn_dimension
   import numpy as np
   neural_data_noisy = neural_data + 1e-6 * np.random.randn(*neural_data.shape)

   # Linear methods
   pca_dim_90 = pca_dimension(neural_data, threshold=0.90)
   pca_dim_95 = pca_dimension(neural_data, threshold=0.95)

   # Effective dimension (participation ratio)
   eff_d = eff_dim(neural_data, enable_correction=True)

   # Nonlinear intrinsic dimension (use noisy data to avoid duplicates)
   nn_dim = nn_dimension(neural_data_noisy, k=5)

   print(f"PCA 90%: {pca_dim_90}, PCA 95%: {pca_dim_95}")
   print(f"Effective dim: {eff_d:.2f}")
   print(f"k-NN dimension: {nn_dim:.2f}")