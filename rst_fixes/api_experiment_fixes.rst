# Fixes for api/experiment.rst
# Add these to the appropriate sections

.. autoclass:: driada.experiment.exp_base.Experiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: driada.experiment.exp_build.load_exp_from_aligned_data

.. autofunction:: driada.experiment.exp_build.load_exp_from_pickle

.. autofunction:: driada.experiment.exp_build.load_experiment

.. autofunction:: driada.experiment.exp_build.save_exp_to_pickle

.. autofunction:: driada.experiment.synthetic.manifold_spatial_2d.generate_2d_manifold_exp

.. autofunction:: driada.experiment.synthetic.manifold_spatial_3d.generate_3d_manifold_exp

.. autofunction:: driada.experiment.synthetic.manifold_circular.generate_circular_manifold_exp

.. autofunction:: driada.experiment.synthetic.experiment_generators.generate_mixed_population_exp

.. autofunction:: driada.experiment.synthetic.experiment_generators.generate_synthetic_exp

.. autoclass:: driada.experiment.neuron.Neuron
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: driada.experiment.spike_reconstruction.reconstruct_spikes

.. autofunction:: driada.experiment.spike_reconstruction.threshold_reconstruction

.. autofunction:: driada.experiment.spike_reconstruction.wavelet_reconstruction

.. autofunction:: driada.experiment.wavelet_event_detection.extract_wvt_events

.. autofunction:: driada.experiment.wavelet_event_detection.get_cwt_ridges
