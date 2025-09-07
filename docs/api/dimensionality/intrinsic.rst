Intrinsic Dimensionality Estimation
====================================

.. currentmodule:: driada.dimensionality.intrinsic

This module provides nonlinear methods for estimating the intrinsic dimensionality of data manifolds.

Functions
---------

.. autofunction:: driada.dimensionality.intrinsic.nn_dimension
.. autofunction:: driada.dimensionality.intrinsic.correlation_dimension
.. autofunction:: driada.dimensionality.intrinsic.geodesic_dimension

Usage Examples
--------------

k-Nearest Neighbors Method
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.dimensionality import nn_dimension
   import numpy as np
   
   # Data on a nonlinear manifold
   theta = np.random.uniform(0, 2*np.pi, 1000)
   phi = np.random.uniform(0, np.pi, 1000)
   # Points on a sphere (2D manifold in 3D)
   data = np.column_stack([
       np.sin(phi) * np.cos(theta),
       np.sin(phi) * np.sin(theta),
       np.cos(phi)
   ])
   
   # Estimate intrinsic dimension
   dim_knn = nn_dimension(data, k=5)
   print(f"k-NN dimension estimate: {dim_knn:.2f}")  # Should be ~2

Correlation Dimension
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.dimensionality import correlation_dimension
   
   # Estimate correlation dimension
   dim_corr = correlation_dimension(data, r_min=0.01, r_max=1.0, n_bins=20)
   print(f"Correlation dimension: {dim_corr:.2f}")


Theory
------

Intrinsic dimensionality methods estimate the dimension of the manifold on which data lies:

**k-NN Method**: Based on the growth rate of nearest neighbor distances:

.. math::

   \hat{d} = \left[ \frac{1}{n} \sum_{i=1}^n \log \frac{r_{k}(i)}{r_{j}(i)} \right]^{-1} \log \frac{k}{j}

**Correlation Dimension**: Measures scaling of correlation integral:

.. math::

   D_c = \lim_{r \to 0} \frac{\log C(r)}{\log r}

where :math:`C(r)` is the fraction of point pairs within distance :math:`r`.

**MLE Method**: Maximum likelihood estimation assuming locally uniform distribution.

Choosing a Method
-----------------

- **nn_dimension**: Good general-purpose method, robust to noise
- **correlation_dimension**: Classic method, requires careful scale selection
- **geodesic_dimension**: Accounts for manifold curvature, slower

For neural data, `nn_dimension` with k=5-10 often provides reliable estimates.