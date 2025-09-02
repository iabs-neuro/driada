Core Experiment Classes
=======================

.. currentmodule:: driada.experiment

This module contains the core data structures for managing neural experiments.

Classes
-------

.. autoclass:: driada.experiment.Experiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: driada.experiment.Neuron
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Creating an Experiment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import Experiment
   from driada.information import MultiTimeSeries
   import numpy as np
   
   # Create calcium imaging data
   n_neurons = 50
   n_timepoints = 10000
   calcium_data = np.random.randn(n_neurons, n_timepoints)
   
   # Note: Experiment requires multiple parameters
   # exp = Experiment(signature, calcium, spikes, exp_identificators, static_features, dynamic_features)
   # For this example, assume exp is already created
   exp.fps = 30.0  # 30 Hz sampling rate
   
   # Add behavior data
   position = np.random.randn(2, n_timepoints)  # x, y coordinates
   exp.behavior = {'position': position}
   
   # Add metadata
   exp.info = {
       'mouse_id': 'M001',
       'session': 'day1',
       'brain_region': 'CA1'
   }

Working with Neurons
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import Neuron
   
   # Create neuron analyzer for first cell
   neuron = Neuron(exp, cell_id=0)
   
   # Get neuron's activity
   activity = neuron.get_activity()
   
   # Compute firing rate
   firing_rate = neuron.compute_firing_rate()
   
   # Get selectivity profile (if INTENSE analysis done)
   if hasattr(neuron, 'selectivity'):
       selectivity = neuron.get_selectivity()

Experiment Properties
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Assume exp is an Experiment object already created
   # exp = Experiment(...) # See Experiment docs for full parameters

   # Core data access
   calcium_matrix = exp.calcium.data  # (n_neurons, n_time)
   sampling_rate = exp.fps

   # Duration and time
   duration = exp.duration  # in seconds
   time_vector = exp.get_time_vector()

   # Neuron count
   n_neurons = exp.n_neurons

   # Check for spike data
   if exp.has_spikes:
       spikes = exp.spikes.data

Data Organization
-----------------

The Experiment class organizes different data modalities:

- **calcium**: MultiTimeSeries of calcium imaging data
- **spikes**: MultiTimeSeries of spike data (optional)
- **behavior**: Dictionary of behavioral variables
- **info**: Metadata dictionary
- **fps**: Sampling rate in Hz

Time alignment is automatic - all time series are assumed to be synchronously sampled at the specified frame rate.