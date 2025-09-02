Wavelet Event Detection
=======================

.. currentmodule:: driada.experiment

This module provides wavelet-based methods for detecting events in neural time series data.

Functions
---------

.. autofunction:: driada.experiment.extract_wvt_events
.. autofunction:: driada.experiment.get_cwt_ridges

Usage Examples
--------------

Basic Event Detection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import extract_wvt_events
   
   # Extract events from calcium traces
   events = extract_wvt_events(
       exp.calcium,
       fps=exp.fps,
       wavelet='morse',
       threshold='otsu'
   )
   
   # events is a list of event dictionaries per neuron
   neuron_0_events = events[0]
   print(f"Neuron 0: {len(neuron_0_events)} events detected")

Continuous Wavelet Transform Ridges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import get_cwt_ridges
   import numpy as np
   
   # Get single neuron trace
   trace = exp.calcium.data[0, :]
   
   # Find CWT ridges
   ridges, cwt_matrix = get_cwt_ridges(
       trace,
       fps=exp.fps,
       wavelet='morlet',
       scales=np.arange(1, 20),
       return_cwt=True
   )
   
   # Plot wavelet transform with ridges
   import matplotlib.pyplot as plt
   plt.imshow(np.abs(cwt_matrix), aspect='auto', cmap='hot')
   for ridge in ridges:
       plt.plot(ridge, 'b-', linewidth=2)
   plt.xlabel('Time (frames)')
   plt.ylabel('Scale')


Advanced Usage
--------------

Custom Wavelet Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Full control over wavelet transform
   from scipy import signal

   # Assume exp is an Experiment object already created
   # exp = Experiment(...) # See Experiment docs for full parameters

   # Create custom wavelet
   wavelet = signal.morlet2
   scales = np.logspace(0, 1.5, 30)

   # Apply CWT
   cwt_matrix = signal.cwt(trace, wavelet, scales)

   # Use with ridge detection
   ridges = get_cwt_ridges(
       trace,
       fps=exp.fps,
       precomputed_cwt=cwt_matrix,
       scales=scales
   )

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Process all neurons in parallel
   from concurrent.futures import ProcessPoolExecutor

   # Assume exp is an Experiment object already created
   # exp = Experiment(...) # See Experiment docs for full parameters

   def process_neuron(args):
       trace, fps = args
       return extract_wvt_events(trace, fps=fps)

   # Parallel processing
   with ProcessPoolExecutor() as executor:
       args = [(exp.calcium.data[i, :], exp.fps) 
               for i in range(exp.n_neurons)]
       all_events = list(executor.map(process_neuron, args))

Theory
------

The wavelet transform provides time-frequency localization of events:

**Continuous Wavelet Transform (CWT)**:

.. math::

   CWT(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt

where :math:`\psi` is the wavelet function, :math:`a` is scale, and :math:`b` is translation.

**Ridge Detection**: Ridges in the CWT correspond to dominant frequency components that persist over time, indicating events.

**Event Detection Pipeline**:

1. Compute CWT of the signal
2. Find ridges in scale-time space
3. Identify peaks along ridges
4. Threshold based on SNR or statistical criteria
5. Extract event times and properties

Wavelet Selection
-----------------

**Morse Wavelet** (default):
- Analytic wavelet with good time-frequency localization
- Flexible shape parameter
- Best for: General purpose event detection

**Morlet Wavelet**:
- Gaussian-windowed complex sinusoid
- Good frequency resolution
- Best for: Oscillatory events

**Mexican Hat (Ricker)**:
- Second derivative of Gaussian
- Simple, real-valued
- Best for: Sharp, transient events