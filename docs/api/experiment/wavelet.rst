Wavelet Event Detection
=======================

.. currentmodule:: driada.experiment

This module provides wavelet-based methods for detecting events in neural time series data.

Functions
---------

.. autofunction:: driada.experiment.wavelet_event_detection.extract_wvt_events
.. autofunction:: driada.experiment.wavelet_event_detection.get_cwt_ridges

Usage Examples
--------------

Basic Event Detection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import extract_wvt_events, load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Extract events from calcium traces
   wvt_kwargs = {
       'fps': exp.fps,
       'sigma': 8,  # smoothing parameter (frames)
       'eps': 10    # minimum spacing between events (frames)
   }
   st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(
       exp.calcium.scdata,  # scaled data as numpy array
       wvt_kwargs
   )
   
   # Results are lists of start/end indices per neuron
   neuron_0_events = st_ev_inds[0]
   print(f"Neuron 0: {len(neuron_0_events)} events detected")

Single Neuron Event Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import events_from_trace, load_demo_experiment
   from ssqueezepy import Wavelet
   import numpy as np
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Get single neuron trace (scaled data)
   trace = exp.calcium.scdata[0, :]
   
   # Setup wavelet and scales for calcium imaging
   wavelet = Wavelet(
       ("gmw", {"gamma": 3, "beta": 2, "centered_scale": True}), 
       N=8196
   )
   manual_scales = np.logspace(2.5, 5.5, 50, base=2)  # calcium-appropriate scales
   
   # Precompute time resolutions
   from ssqueezepy import time_resolution
   rel_wvt_times = [
       time_resolution(wavelet, scale=sc, nondim=False, min_decay=200)
       for sc in manual_scales
   ]
   
   # Detect events
   all_ridges, st_ev, end_ev = events_from_trace(
       trace, wavelet, manual_scales, rel_wvt_times, 
       fps=exp.fps, sigma=8, eps=10
   )
   
   print(f"Found {len(st_ev)} events")
   for i, (start, end) in enumerate(zip(st_ev[:3], end_ev[:3])):
       duration_ms = (end - start) / exp.fps * 1000
       print(f"Event {i}: frames {start}-{end} ({duration_ms:.0f} ms)")


Advanced Usage
--------------

Custom Event Detection Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Customize event detection parameters
   from driada.experiment import extract_wvt_events, load_demo_experiment
   from driada.experiment.wavelet_event_detection import WVT_EVENT_DETECTION_PARAMS
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Customize parameters for more/less sensitive detection
   custom_params = WVT_EVENT_DETECTION_PARAMS.copy()
   custom_params.update({
       'fps': exp.fps,
       'sigma': 4,             # less smoothing for sharper events
       'eps': 20,              # require more spacing between events
       'max_ampl_thr': 0.1,    # higher threshold, fewer events
       'scale_length_thr': 30  # events must persist across 30+ scales
   })
   
   # Extract events with custom parameters
   st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(
       exp.calcium.scdata,
       custom_params
   )
   
   # Compare default vs custom
   default_params = {'fps': exp.fps}
   st_def, _, _ = extract_wvt_events(exp.calcium.scdata, default_params)
   
   print(f"Default params: {sum(len(s) for s in st_def)} total events")
   print(f"Custom params: {sum(len(s) for s in st_ev_inds)} total events")

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Process all neurons
   from driada.experiment import extract_wvt_events, load_demo_experiment

   # Load sample experiment
   exp = load_demo_experiment()

   # Extract events for all neurons
   wvt_kwargs = {
       'fps': exp.fps,
       'sigma': 8,   # default smoothing
       'eps': 10     # default spacing
   }
   st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(
       exp.calcium.scdata,  # pass all neurons at once
       wvt_kwargs
   )
   
   # Analyze results
   n_events_per_neuron = [len(events) for events in st_ev_inds]
   print(f"Average events per neuron: {np.mean(n_events_per_neuron):.1f}")
   print(f"Total events detected: {sum(n_events_per_neuron)}")

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