Core Experiment Classes
=======================

.. currentmodule:: driada.experiment

This module contains the core data structures for managing neural experiments.

Classes
-------

.. autoclass:: driada.experiment.exp_base.Experiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: driada.experiment.neuron.Neuron
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Creating an Experiment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import Experiment, load_demo_experiment
   from driada.information import MultiTimeSeries
   import numpy as np
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Access and modify experiment properties
   print(f"Sampling rate: {exp.fps} Hz")
   
   # Add behavior data as dynamic features
   n_timepoints = exp.n_frames  # Get number of timepoints from experiment
   position = np.random.randn(2, n_timepoints)  # x, y coordinates
   exp.dynamic_features['position'] = position
   
   # Access metadata
   # Note: exp_identificators is set during experiment creation
   # and contains experiment parameters like mouse_id, session, etc.

Working with Neurons
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import Neuron, load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Create neuron analyzer for first cell
   # Neuron takes calcium data and spike data directly
   neuron = Neuron(
       cell_id=0,
       ca=exp.calcium.scdata[0],  # calcium trace for neuron 0
       sp=None,  # spike data (optional)
       fps=exp.fps
   )
   
   # Access neuron's data
   ca_trace = neuron.ca  # Raw calcium data
   sp_trace = neuron.sp  # Spike data (if provided)

Experiment Properties
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()

   # Core data access
   calcium_matrix = exp.calcium.data  # (n_neurons, n_time)
   sampling_rate = exp.fps

   # Duration and time
   duration = exp.n_frames / exp.fps  # compute duration in seconds
   time_vector = np.arange(exp.n_frames) / exp.fps  # compute time vector

   # Neuron count
   n_neurons = exp.n_cells

   # Check for spike data
   if exp.spikes is not None:
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