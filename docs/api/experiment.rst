Experiment Module
=================

.. automodule:: driada.experiment
   :no-members:
   :noindex:

Core data structures and utilities for managing neural experiments, including data loading/saving,
spike reconstruction, wavelet analysis, and synthetic data generation.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   experiment/core
   experiment/loading
   experiment/reconstruction
   experiment/wavelet
   experiment/synthetic

Quick Links
-----------

**Core Classes**
   * :class:`~driada.experiment.Experiment` - Main experiment container
   * :class:`~driada.experiment.Neuron` - Single neuron analysis
   * :doc:`experiment/core` - Experiment management

**Loading and Saving**
   * :func:`~driada.experiment.load_experiment` - Load from various formats
   * :func:`~driada.experiment.load_exp_from_aligned_data` - Load pre-aligned data
   * :func:`~driada.experiment.save_exp_to_pickle` - Save experiments
   * :func:`~driada.experiment.load_exp_from_pickle` - Load pickled experiments
   * :doc:`experiment/loading` - All I/O functions

**Spike Reconstruction**
   * :func:`~driada.experiment.reconstruct_spikes` - Main reconstruction function
   * :func:`~driada.experiment.wavelet_reconstruction` - Wavelet-based method
   * :func:`~driada.experiment.threshold_reconstruction` - Threshold-based method
   * :doc:`experiment/reconstruction` - Spike deconvolution methods

**Wavelet Analysis**
   * :func:`~driada.experiment.extract_wvt_events` - Extract wavelet events
   * :func:`~driada.experiment.get_cwt_ridges` - Find CWT ridges
   * :doc:`experiment/wavelet` - Wavelet event detection

**Synthetic Data Generation**
   * :func:`~driada.experiment.generate_synthetic_exp` - Basic synthetic data
   * :func:`~driada.experiment.generate_circular_manifold_exp` - Head direction cells
   * :func:`~driada.experiment.generate_2d_manifold_exp` - 2D place cells
   * :func:`~driada.experiment.generate_3d_manifold_exp` - 3D spatial cells
   * :func:`~driada.experiment.generate_mixed_population_exp` - Mixed populations
   * :doc:`experiment/synthetic` - All synthetic data generators

Usage Example
-------------

.. code-block:: python

   from driada.experiment import Experiment, load_experiment
   
   # Load experimental data
   exp = load_experiment('path/to/data.mat')
   
   # Or create synthetic data
   from driada.experiment import generate_circular_manifold_exp
   exp = generate_circular_manifold_exp(
       n_neurons=50,
       duration=600,
       noise_std=0.1
   )
   
   # Access data
   calcium_data = exp.calcium  # MultiTimeSeries
   
   # Reconstruct spikes
   from driada.experiment import reconstruct_spikes
   spikes = reconstruct_spikes(exp.calcium, fps=exp.fps, method='wavelet')