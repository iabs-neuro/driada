Linear Dimensionality Methods
=============================

.. currentmodule:: driada.dimensionality.linear

This module contains linear methods for dimensionality estimation, primarily based on Principal Component Analysis (PCA).

Functions
---------

.. autofunction:: driada.dimensionality.linear.pca_dimension
.. autofunction:: driada.dimensionality.linear.pca_dimension_profile
.. autofunction:: driada.dimensionality.linear.effective_rank

Usage Examples
--------------

PCA-based Dimensionality Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.dimensionality import pca_dimension, pca_dimension_profile
   import numpy as np
   
   # Generate sample data
   data = np.random.randn(1000, 50)  # 1000 samples, 50 features
   
   # Find dimension explaining 90% variance
   dim_90 = pca_dimension(data, threshold=0.90)
   print(f"Dimensions for 90% variance: {dim_90}")
   
   # Get full PCA profile
   profile = pca_dimension_profile(data)
   print(f"Cumulative variance: {profile['cumulative_variance'][:10]}")

Effective Rank
^^^^^^^^^^^^^^

.. code-block:: python

   from driada.dimensionality import effective_rank
   
   # Compute effective rank of covariance matrix
   cov_matrix = np.cov(data.T)
   eff_rank = effective_rank(cov_matrix)
   print(f"Effective rank: {eff_rank:.2f}")

Implementation Details
----------------------

The linear dimensionality methods are based on eigenvalue decomposition of the data covariance matrix. The key idea is that the eigenvalue spectrum reveals the intrinsic dimensionality:

- **PCA dimension**: Number of components needed to explain a threshold percentage of variance
- **Effective rank**: Entropy-based measure using the eigenvalue distribution
- **PCA profile**: Complete characterization of the eigenvalue spectrum

These methods are computationally efficient and provide interpretable results, making them suitable for initial dimensionality assessment.