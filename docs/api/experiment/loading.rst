Data Loading and Saving
=======================

.. currentmodule:: driada.experiment

This module provides functions for loading experimental data from various formats and saving processed experiments.

Functions
---------

.. autofunction:: driada.experiment.load_experiment
.. autofunction:: driada.experiment.load_exp_from_aligned_data
.. autofunction:: driada.experiment.save_exp_to_pickle
.. autofunction:: driada.experiment.load_exp_from_pickle

Usage Examples
--------------

Loading from Different Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import load_experiment
   
   # Load from MATLAB file
   exp_mat = load_experiment('data/experiment.mat')
   
   # Load from NWB file
   exp_nwb = load_experiment('data/experiment.nwb')
   
   # Load from pickle
   exp_pkl = load_experiment('data/experiment.pkl')
   
   # Auto-detect format based on extension
   exp = load_experiment('path/to/data.mat')

Loading Pre-aligned Data
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import load_exp_from_aligned_data
   import numpy as np
   
   # Prepare aligned data
   calcium = np.random.randn(50, 10000)  # 50 neurons, 10000 timepoints
   behavior = {
       'position': np.random.randn(2, 10000),
       'velocity': np.random.randn(2, 10000)
   }
   
   # Create experiment
   exp = load_exp_from_aligned_data(
       calcium=calcium,
       behavior=behavior,
       fps=30.0,
       info={'mouse': 'M001', 'session': 'day1'}
   )

Saving and Loading Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import save_exp_to_pickle, load_exp_from_pickle
   
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

NWB File Support
^^^^^^^^^^^^^^^^

.. code-block:: python

   # NWB files are automatically parsed
   exp = load_experiment('recording.nwb')
   
   # Access standard NWB data
   # - Calcium imaging from ophys processing module
   # - Behavior from behavior processing module
   # - Metadata from file metadata

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

Best Practices
--------------

1. **Data Organization**: Keep calcium and behavior data time-aligned
2. **Metadata**: Include as much metadata as possible in the info dictionary
3. **File Formats**: Use pickle for processed data, original formats for raw data
4. **Compression**: Pickle files are compressed by default
5. **Large Files**: Consider HDF5 for very large datasets