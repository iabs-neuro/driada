Core Information Theory Classes
================================

.. currentmodule:: driada.information

This module contains the core data structures and main functions for information-theoretic analysis.

Classes
-------

.. autoclass:: driada.information.info_base.TimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: driada.information.info_base.MultiTimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: driada.information.info_base.get_mi
.. autofunction:: driada.information.info_base.get_1d_mi
.. autofunction:: driada.information.info_base.get_multi_mi
.. autofunction:: driada.information.info_base.get_tdmi
.. autofunction:: driada.information.info_base.conditional_mi
.. autofunction:: driada.information.info_base.interaction_information

Usage Examples
--------------

TimeSeries Objects
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import TimeSeries
   import numpy as np
   
   # Discrete time series (e.g., spike counts)
   spike_counts = np.random.poisson(2, 1000)
   ts_discrete = TimeSeries(spike_counts, discrete=True)
   
   # Continuous time series (e.g., LFP)
   lfp_signal = np.random.randn(1000)
   ts_continuous = TimeSeries(lfp_signal, discrete=False)
   
   # Access properties
   print(f"Is discrete: {ts_discrete.discrete}")
   print(f"Data shape: {ts_discrete.data.shape}")
   print(f"Data length: {ts_discrete.data.shape[0]}")

MultiTimeSeries Objects
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import MultiTimeSeries
   
   # Multiple neural recordings
   neural_data = np.random.randn(50, 10000)  # 50 neurons, 10000 timepoints
   mts = MultiTimeSeries(neural_data, discrete=False)
   
   # Properties
   print(f"Number of series: {mts.n_dim}")  # n_dim = number of rows/series
   print(f"Number of timepoints: {mts.n_points}")  # n_points = number of columns/timepoints
   print(f"Data shape: {mts.data.shape}")  # Access numpy array shape directly
   
   # Access individual series data
   neuron_5_data = mts.data[5, :]  # Returns numpy array for neuron 5

Basic Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import get_mi
   
   # MI between two continuous variables
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   
   mi_gcmi = get_mi(x, y, estimator='gcmi')
   print(f"MI (GCMI): {mi_gcmi:.3f} bits")
   
   # MI between discrete variables
   x_discrete = TimeSeries(np.random.randint(0, 5, 1000), discrete=True)
   y_discrete = TimeSeries(np.random.randint(0, 3, 1000), discrete=True)
   
   mi_discrete = get_mi(x_discrete, y_discrete)  # Default estimator handles discrete data
   print(f"MI (discrete): {mi_discrete:.3f} bits")

Conditional Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import conditional_mi
   
   # CMI: I(X;Y|Z)
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   z = TimeSeries(np.random.randn(1000), discrete=False)
   
   cmi = conditional_mi(x, y, z)
   print(f"I(X;Y|Z) = {cmi:.3f} bits")

Interaction Information
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import interaction_information
   
   # Three-way interaction
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   z = TimeSeries(np.random.randn(1000), discrete=False)
   
   ii = interaction_information(x, y, z)
   print(f"Interaction information: {ii:.3f} bits")
   
   # Interpretation:
   # ii > 0: Synergy (variables together provide more info)
   # ii < 0: Redundancy (variables share information)

Advanced Usage
--------------

Time-lagged MI
^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import get_tdmi
   
   # Compute time-delayed mutual information
   signal = np.random.randn(1000)
   
   # Calculate TDMI for lags 1 to 50
   tdmi_values = get_tdmi(signal, min_shift=1, max_shift=50)
   
   # Find optimal embedding delay (first local minimum)
   from scipy.signal import argrelmin
   minima = argrelmin(np.array(tdmi_values))[0]
   if len(minima) > 0:
       optimal_delay = minima[0] + 1  # +1 because min_shift=1
   
   # Plot TDMI curve
   import matplotlib.pyplot as plt
   plt.plot(range(1, 50), tdmi_values)
   plt.xlabel('Time lag')
   plt.ylabel('TDMI (bits)')

Multivariate MI
^^^^^^^^^^^^^^^

.. code-block:: python

   # MI between groups of variables
   group1 = MultiTimeSeries(np.random.randn(5, 1000), discrete=False)
   group2 = MultiTimeSeries(np.random.randn(3, 1000), discrete=False)
   
   mi_groups = get_mi(group1, group2, estimator='gcmi')
   print(f"MI between groups: {mi_groups:.3f} bits")

Estimator Selection
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compare different estimators
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   
   estimators = ['gcmi', 'ksg']  # Both work for continuous data
   
   for est in estimators:
       mi = get_mi(x, y, estimator=est)
       print(f"{est}: {mi:.3f} bits")

Best Practices
--------------

1. **Data Preparation**:
   - Ensure time series are aligned
   - Remove artifacts before analysis
   - Consider normalization for continuous data

2. **Estimator Choice**:
   - ``gcmi``: Fast, good for continuous Gaussian-like data
   - ``ksg``: Non-parametric, works for any continuous distribution
   - ``discrete``: For categorical/integer data

3. **Sample Size**:
   - MI estimation requires sufficient data
   - Rule of thumb: >1000 samples for reliable estimates
   - More data needed for higher dimensions

4. **Bias Correction**:
   - All estimators have finite sample bias
   - Consider shuffle controls for significance testing

.. code-block:: python

   # Example: Significance testing
   def mi_significance(x, y, n_shuffles=100):
       true_mi = get_mi(x, y)
       
       shuffle_mi = []
       for _ in range(n_shuffles):
           y_shuffled = TimeSeries(
               np.random.permutation(y.data),
               discrete=y.discrete
           )
           shuffle_mi.append(get_mi(x, y_shuffled))
       
       p_value = np.mean(shuffle_mi >= true_mi)
       return true_mi, p_value