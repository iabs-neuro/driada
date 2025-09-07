Data Loading and Saving
=======================

.. currentmodule:: driada.experiment

This module provides functions for loading experimental data from various formats and saving processed experiments.

Functions
---------

.. autofunction:: driada.experiment.exp_build.load_experiment
.. autofunction:: driada.experiment.exp_build.load_exp_from_aligned_data
.. autofunction:: driada.experiment.exp_build.save_exp_to_pickle
.. autofunction:: driada.experiment.exp_build.load_exp_from_pickle

Usage Examples
--------------

Loading from IABS Data
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import load_experiment
   
   # Define experiment parameters for IABS data
   exp_params = {
       'track': 'STFP',
       'animal_id': 'M123',
       'session': '1'
   }
   
   # Load from Google Drive (IABS data)
   # Note: Requires config.py with IABS_ROUTER_URL set
   # exp_gdrive = load_experiment('IABS', exp_params, via_pydrive=True)

Loading Pre-aligned Data
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import load_exp_from_aligned_data
   import numpy as np
   
   # Prepare aligned data dictionary
   # Note: Dynamic features should be 1D arrays
   data = {
       'calcium': np.random.randn(50, 10000),  # 50 neurons, 10000 timepoints
       'position_x': np.random.randn(10000),   # x coordinates
       'position_y': np.random.randn(10000),   # y coordinates
       'velocity_x': np.random.randn(10000),   # x velocity
       'velocity_y': np.random.randn(10000),   # y velocity
   }
   
   # Create experiment from IABS-style data
   exp = load_exp_from_aligned_data(
       data_source='IABS',
       exp_params={
           'track': 'STFP',
           'animal_id': 'M001', 
           'session': '1'
       },
       data=data,
       static_features={'fps': 30.0}  # Pass fps as static feature
   )

Loading from Generic Lab Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import load_experiment
   
   # Load from NPZ file for non-IABS labs
   # Using example data file from the repository
   exp, _ = load_experiment(
       'MyLab',
       {'name': 'spatial_navigation_task'},
       data_path='examples/example_data/sample_recording.npz',
       reconstruct_spikes=False,  # Disable for speed
       verbose=False
   )
   
   # Load with custom naming based on parameters
   exp2, _ = load_experiment(
       'NeuroLab',
       {'subject': 'rat42', 'session': 'day3', 'experiment': 'maze'},
       data_path='examples/example_data/sample_recording.npz',
       reconstruct_spikes=False,  # Disable for speed
       save_to_pickle=False,
       verbose=False
   )
   
   # NPZ files automatically handle multidimensional features
   # 2D arrays become MultiTimeSeries objects
   # Scalar and non-numeric values are ignored with warnings

Handling Multidimensional Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 2D features (e.g., position) are automatically handled
   data = {
       'calcium': np.random.randn(50, 10000),
       'position': np.random.randn(2, 10000),  # 2D trajectory -> MultiTimeSeries
       'speed': np.random.randn(10000),        # 1D -> TimeSeries
   }
   
   # Create experiment - 2D arrays automatically become MultiTimeSeries
   exp = load_exp_from_aligned_data(
       data_source='MyLab',
       exp_params={'name': 'test_exp'},
       data=data
   )
   
   # Access multidimensional features
   print(exp.position.n_dim)  # 2
   print(exp.position.data.shape)  # (2, 10000)

Saving and Loading Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import save_exp_to_pickle, load_exp_from_pickle
   
   # First create/load an experiment
   import numpy as np
   from driada.experiment import load_exp_from_aligned_data
   data = {
       'calcium': np.random.rand(10, 1000),
       'position': np.random.rand(1000)
   }
   exp = load_exp_from_aligned_data('MyLab', {'name': 'test'}, data, verbose=False)
   
   # Save experiment
   save_exp_to_pickle(exp, 'processed_experiment.pkl')
   
   # Load later
   exp_loaded = load_exp_from_pickle('processed_experiment.pkl')
   
   # Verify data integrity
   assert np.allclose(exp.calcium.data, exp_loaded.calcium.data)

MATLAB File Format
^^^^^^^^^^^^^^^^^^

Expected structure for MATLAB files:

.. code-block:: matlab

   % Required fields
   data.calcium = [n_neurons x n_timepoints];  % Calcium imaging data
   data.fps = 30;                              % Sampling rate
   
   % Optional fields
   data.spikes = [n_neurons x n_timepoints];   % Spike data
   data.behavior.position = [2 x n_timepoints]; % Position data
   data.behavior.velocity = [2 x n_timepoints]; % Velocity data
   data.info.mouse_id = 'M001';                % Metadata


Custom Loaders
^^^^^^^^^^^^^^

.. code-block:: python

   # For custom formats, use load_exp_from_aligned_data
   def load_custom_format(filename):
       # Load your custom format
       data = custom_loader(filename)
       
       # Extract components
       calcium = data['neural_activity']
       behavior = {
           'stimulus': data['stimulus_trace'],
           'response': data['behavioral_response']
       }
       
       # Create experiment
       return load_exp_from_aligned_data(
           calcium=calcium,
           behavior=behavior,
           fps=data['sampling_rate']
       )

Error Handling
^^^^^^^^^^^^^^

.. code-block:: python

   # Missing data_path for non-IABS sources
   try:
       exp, _ = load_experiment('MyLab', {'name': 'test'})
   except ValueError as e:
       print(e)  # "For data source 'MyLab', you must provide the 'data_path' parameter"
   
   # Missing calcium data
   try:
       data = {'position': np.random.randn(1000)}  # No calcium!
       exp = load_exp_from_aligned_data('MyLab', {}, data)
   except ValueError as e:
       print(e)  # "No calcium data found!"
   
   # Warnings for invalid data types
   data = {
       'calcium': np.random.randn(50, 1000),
       'fps': 30.0,  # Scalar - will be ignored with warning
       'labels': np.array(['A', 'B', 'C'] * 333 + ['A'])  # Non-numeric - ignored
   }
   exp = load_exp_from_aligned_data('MyLab', {}, data)
   # Warning: Ignoring scalar value 'fps' found in NPZ file
   # Warning: Ignoring non-numeric feature 'labels' with dtype <U1

Best Practices
--------------

1. **Data Organization**: Keep calcium and behavior data time-aligned
2. **Metadata**: Include as much metadata as possible in exp_params
3. **File Formats**: Use NPZ for raw data, pickle for processed experiments
4. **Compression**: Pickle files are compressed by default
5. **Large Files**: Consider HDF5 for very large datasets
6. **Static Features**: Pass constants like fps via static_features parameter
7. **Multidimensional Data**: Store as 2D arrays (components x time) in NPZ files