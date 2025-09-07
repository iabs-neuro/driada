Effective Dimensionality Estimation
===================================

.. currentmodule:: driada.dimensionality.effective

This module implements effective dimensionality estimation based on participation ratio and Rényi entropy methods.

Functions
---------

.. autofunction:: driada.dimensionality.effective.eff_dim

Usage Examples
--------------

Participation Ratio Method
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.dimensionality import eff_dim
   import numpy as np
   
   # Neural population data
   neural_data = np.random.randn(1000, 100)  # 1000 timepoints, 100 neurons
   
   # Basic effective dimension (no correction)
   d_eff = eff_dim(neural_data, enable_correction=False)
   print(f"Effective dimensionality: {d_eff:.2f}")
   
   # With bias correction (recommended for finite samples)
   d_eff_corrected = eff_dim(neural_data, enable_correction=True)
   print(f"Corrected effective dim: {d_eff_corrected:.2f}")


Theory
------

Effective dimensionality quantifies how many dimensions are "effectively" used by the data:

**Participation Ratio**:

.. math::

   D_{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}

where :math:`\lambda_i` are the eigenvalues of the covariance matrix.

**Rényi Entropy**:

.. math::

   D_{\alpha} = \frac{1}{1-\alpha} \log \sum_i p_i^{\alpha}

The participation ratio is a special case of Rényi entropy dimension with :math:`\alpha = 2`.

Interpretation
--------------

- **Low effective dimension** (e.g., 2-5): Data lies on a low-dimensional manifold
- **Medium dimension** (e.g., 10-20): Moderate complexity, typical for many neural recordings
- **High dimension** (approaching number of features): Data spans the full space, possibly noise-dominated

The corrected version accounts for finite sample bias, providing more accurate estimates for small datasets.