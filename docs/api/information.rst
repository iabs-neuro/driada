Information Theory Module
=========================

.. automodule:: driada.information
   :no-members:
   :noindex:

Information-theoretic measures for analyzing neural data, including entropy estimation,
mutual information computation, and advanced corrections for finite data.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   information/core
   information/entropy
   information/mutual_information
   information/estimators
   information/utilities

Quick Links
-----------

**Core Data Structures**
   * :class:`~driada.information.info_base.TimeSeries` - Single time series container
   * :class:`~driada.information.info_base.MultiTimeSeries` - Multi-dimensional time series
   * :doc:`information/core` - Base classes and main MI functions

**Mutual Information**
   * :func:`~driada.information.info_base.get_mi` - Main MI computation function
   * :func:`~driada.information.info_base.conditional_mi` - Conditional MI
   * :func:`~driada.information.info_base.interaction_information` - Multi-variable interactions
   * :doc:`information/mutual_information` - All MI-related functions

**Entropy Estimation**
   * :doc:`information/entropy` - Discrete and continuous entropy
   * JIT-optimized implementations for performance

**Estimators**
   * :doc:`information/estimators` - Different MI estimation methods
   * GCMI (Gaussian Copula) - Fast parametric estimation
   * KSG (k-nearest neighbors) - Non-parametric estimation

Usage Example
-------------

.. code-block:: python

   from driada.information import TimeSeries, get_mi
   import numpy as np
   
   # Create time series with matching lengths
   n_samples = 1000
   
   # Generate example data
   spike_counts = np.random.poisson(3, n_samples)  # Discrete spike counts
   position = np.cumsum(np.random.randn(n_samples) * 0.1)  # Continuous position
   
   # Ensure data has no extreme outliers for GCMI
   position = np.clip(position, np.percentile(position, 1), np.percentile(position, 99))
   
   neural_data = TimeSeries(spike_counts, discrete=True)
   behavior = TimeSeries(position, discrete=False)
   
   # Compute mutual information  
   mi_value = get_mi(neural_data, behavior, estimator='gcmi')
   
   # Conditional MI (X must be continuous)
   from driada.information import conditional_mi
   speed = np.abs(np.diff(position, prepend=position[0]))  # Speed from position
   speed_ts = TimeSeries(speed[:n_samples], discrete=False)
   cmi = conditional_mi(behavior, neural_data, speed_ts)  # I(position; spikes | speed)