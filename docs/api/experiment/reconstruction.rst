Spike Reconstruction Methods
============================

.. currentmodule:: driada.experiment

This module provides methods for reconstructing spike trains from calcium imaging data.

Functions
---------

.. autofunction:: driada.experiment.spike_reconstruction.reconstruct_spikes
.. autofunction:: driada.experiment.spike_reconstruction.wavelet_reconstruction
.. autofunction:: driada.experiment.spike_reconstruction.threshold_reconstruction

Usage Examples
--------------

Basic Spike Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import reconstruct_spikes, load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Reconstruct spikes using default method (wavelet)
   spikes = reconstruct_spikes(
       exp.calcium,
       fps=exp.fps,
       method='wavelet'
   )
   
   # Add to experiment
   exp.spikes = spikes

Wavelet-based Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import wavelet_reconstruction, load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Wavelet reconstruction with custom parameters
   params = {
       'sigma': 2,            # Smoothing parameter
       'eps': 3,              # Min spacing between events
       'scale_length_thr': 3, # Min scales for ridge
       'max_scale_thr': 5     # Scale with max intensity
   }
   
   spikes, metadata = wavelet_reconstruction(
       exp.calcium,
       fps=exp.fps,
       params=params
   )
   
   # Access reconstruction info
   print(f"Detected {len(metadata['start_events'])} spike events")

Threshold-based Method
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import threshold_reconstruction, load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Simple threshold-based detection
   params = {
       'threshold': 3.0,      # threshold in standard deviations
       'min_width': 2         # minimum spike width
   }
   
   spikes, metadata = threshold_reconstruction(
       exp.calcium,
       fps=exp.fps,
       params=params
   )

Validation and Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import reconstruct_spikes, load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Compare different methods
   methods = ['wavelet', 'threshold']
   reconstructions = {}

   for method in methods:
       spikes, metadata = reconstruct_spikes(
           exp.calcium, 
           method=method,
           fps=exp.fps
       )
       reconstructions[method] = spikes

   # Compare reconstructions using correlation
   from scipy.stats import pearsonr

   # Assume exp is an Experiment object already created
   # exp = Experiment(...) # See Experiment docs for full parameters
   corr, _ = pearsonr(
       reconstructions['wavelet'].data.flatten(),
       reconstructions['threshold'].data.flatten()
   )
   print(f"Method correlation: {corr:.3f}")

Method Selection Guide
----------------------

**Wavelet Method** (default):
- Pros: Robust to noise, captures event timing well
- Cons: May miss very small events
- Best for: Standard calcium imaging data

**Threshold Method**:
- Pros: Simple, fast, interpretable
- Cons: Sensitive to baseline fluctuations
- Best for: High SNR data with stable baseline


Parameter Guidelines
--------------------

**Wavelet parameters**:

- ``wavelet``: 'morse' (default) or 'morlet'
- ``threshold_factor``: 2.5-3.5 (higher = fewer false positives)

**Threshold parameters**:

- ``threshold``: 'adaptive' or fixed value (e.g., 2.0)
- ``sigma``: 2.5-4.0 standard deviations
- ``min_spike_width``: 1-3 frames


Output Format
-------------

All methods return a MultiTimeSeries object with:

- Binary spike indicators (0 or 1)
- Same dimensions as input calcium data
- Preserved neuron ordering
- Time alignment with calcium data