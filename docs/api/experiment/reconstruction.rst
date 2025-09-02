Spike Reconstruction Methods
============================

.. currentmodule:: driada.experiment

This module provides methods for reconstructing spike trains from calcium imaging data.

Functions
---------

.. autofunction:: driada.experiment.reconstruct_spikes
.. autofunction:: driada.experiment.wavelet_reconstruction
.. autofunction:: driada.experiment.threshold_reconstruction

Usage Examples
--------------

Basic Spike Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import reconstruct_spikes
   
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

   from driada.experiment import wavelet_reconstruction
   
   # Detailed wavelet reconstruction with custom parameters
   spikes, metadata = wavelet_reconstruction(
       exp.calcium,
       fps=exp.fps,
       wavelet='morse',        # Morse wavelet
       scales=None,           # Auto-determine scales
       threshold_factor=3.0,  # 3 sigma threshold
       return_metadata=True
   )
   
   # Access reconstruction info
   print(f"Detected {metadata['n_events']} spike events")
   print(f"Mean firing rate: {metadata['mean_rate']:.2f} Hz")

Threshold-based Method
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import threshold_reconstruction
   
   # Simple threshold-based detection
   spikes = threshold_reconstruction(
       exp.calcium,
       fps=exp.fps,
       threshold='adaptive',  # or fixed value
       sigma=3.0,            # threshold in standard deviations
       min_spike_width=2     # minimum frames between spikes
   )

Validation and Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compare different methods
   methods = ['wavelet', 'threshold']
   reconstructions = {}

   for method in methods:
       spikes = reconstruct_spikes(exp.calcium, exp.fps, method=method)
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