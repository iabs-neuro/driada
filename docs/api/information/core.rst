Core Information Theory Classes
================================

.. currentmodule:: driada.information

This module contains the core data structures and main functions for information-theoretic analysis.

Classes
-------

.. autoclass:: driada.information.TimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: driada.information.MultiTimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: driada.information.get_mi
.. autofunction:: driada.information.conditional_mi
.. autofunction:: driada.information.interaction_information

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
   print(f"Length: {len(ts_discrete)}")
   print(f"Is discrete: {ts_discrete.discrete}")
   print(f"Data shape: {ts_discrete.data.shape}")

MultiTimeSeries Objects
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import MultiTimeSeries
   
   # Multiple neural recordings
   neural_data = np.random.randn(50, 10000)  # 50 neurons, 10000 timepoints
   mts = MultiTimeSeries(neural_data, discrete=False)
   
   # Access individual time series
   neuron_5 = mts[5]  # Returns TimeSeries for neuron 5
   
   # Slice multiple neurons
   subset = mts[10:20]  # Returns MultiTimeSeries with neurons 10-19
   
   # Properties
   print(f"Number of series: {mts.n_series}")
   print(f"Length: {mts.n_timepoints}")
   print(f"Shape: {mts.shape}")

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
   
   mi_discrete = get_mi(x_discrete, y_discrete, estimator='discrete')
   print(f"MI (discrete): {mi_discrete:.3f} bits")

Conditional Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import conditional_mi
   
   # CMI: I(X;Y|Z)
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   z = TimeSeries(np.random.randn(1000), discrete=False)
   
   cmi = conditional_mi(x, y, z, estimator='gcmi')
   print(f"I(X;Y|Z) = {cmi:.3f} bits")
   
   # Multiple conditioning variables
   z_multi = MultiTimeSeries(np.random.randn(3, 1000), discrete=False)
   cmi_multi = conditional_mi(x, y, z_multi, estimator='gcmi')

Interaction Information
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import interaction_information
   
   # Three-way interaction
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   z = TimeSeries(np.random.randn(1000), discrete=False)
   
   ii = interaction_information([x, y, z], estimator='gcmi')
   print(f"Interaction information: {ii:.3f} bits")
   
   # Interpretation:
   # ii > 0: Synergy (variables together provide more info)
   # ii < 0: Redundancy (variables share information)

Advanced Usage
--------------

Time-lagged MI
^^^^^^^^^^^^^^

.. code-block:: python

   from driada.information import get_mi
   
   # Compute MI at different lags
   x = TimeSeries(np.random.randn(1000), discrete=False)
   y = TimeSeries(np.random.randn(1000), discrete=False)
   
   lags = range(-10, 11)
   mi_values = []
   
   for lag in lags:
       if lag > 0:
           mi = get_mi(x[:-lag], y[lag:])
       elif lag < 0:
           mi = get_mi(x[-lag:], y[:lag])
       else:
           mi = get_mi(x, y)
       mi_values.append(mi)
   
   # Find optimal lag
   optimal_lag = lags[np.argmax(mi_values)]

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
   estimators = ['gcmi', 'ksg', 'discrete']
   
   for est in estimators:
       try:
           mi = get_mi(x, y, estimator=est)
           print(f"{est}: {mi:.3f} bits")
       except ValueError as e:
           print(f"{est}: Not applicable - {e}")

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