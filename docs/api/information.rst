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
   * :class:`~driada.information.TimeSeries` - Single time series container
   * :class:`~driada.information.MultiTimeSeries` - Multi-dimensional time series
   * :doc:`information/core` - Base classes and main MI functions

**Mutual Information**
   * :func:`~driada.information.get_mi` - Main MI computation function
   * :func:`~driada.information.conditional_mi` - Conditional MI
   * :func:`~driada.information.interaction_information` - Multi-variable interactions
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
   
   # Create time series
   neural_data = TimeSeries(spike_counts, discrete=True)
   behavior = TimeSeries(position, discrete=False)
   
   # Compute mutual information
   mi_value = get_mi(neural_data, behavior, estimator='gcmi')
   
   # Conditional MI
   from driada.information import conditional_mi
   cmi = conditional_mi(neural_data, behavior, condition=speed)