Synthetic Data Generation
=========================

.. currentmodule:: driada.experiment.synthetic

This module provides functions for generating synthetic neural data with known ground truth properties.

Functions
---------

.. autofunction:: driada.experiment.generate_synthetic_exp
.. autofunction:: driada.experiment.generate_circular_manifold_exp
.. autofunction:: driada.experiment.generate_2d_manifold_exp
.. autofunction:: driada.experiment.generate_3d_manifold_exp
.. autofunction:: driada.experiment.generate_mixed_population_exp

Usage Examples
--------------

Basic Synthetic Data
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_synthetic_exp
   
   # Generate basic synthetic experiment
   exp = generate_synthetic_exp(
       nneurons=100,      # number of neurons
       n_dfeats=10,       # discrete features
       n_cfeats=10,       # continuous features
       duration=600,      # 10 minutes
       fps=30,           # 30 Hz sampling
       seed=42           # For reproducibility
   )
   
   print(f"Generated {exp.n_cells} neurons, {exp.n_frames/exp.fps}s recording")

Head Direction Cells (Circular Manifold)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_circular_manifold_exp
   
   # Generate head direction cell population
   exp = generate_circular_manifold_exp(
       n_neurons=50,
       duration=600,
       fps=30,
       kappa=4.0,           # concentration parameter
       noise_std=0.1,
       step_std=0.1         # angular velocity noise
   )
   
   # Access ground truth data if return_info=True
   # exp, info = generate_circular_manifold_exp(..., return_info=True)
   # true_angle = info['true_angle']
   # preferred_directions = info['preferred_directions']

2D Place Cells
^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_2d_manifold_exp
   
   # Generate place cell population
   exp = generate_2d_manifold_exp(
       n_neurons=50,
       duration=60,          # 1 minute (for demo)
       fps=20,
       field_sigma=0.1,      # receptive field size
       step_size=0.02,       # movement step
       momentum=0.8,         # movement momentum
       noise_std=0.15
   )
   
   # Access data if return_info=True
   # exp, info = generate_2d_manifold_exp(..., return_info=True)
   # position = info['position']  # (2, n_frames)
   # place_fields = info['place_fields']  # neuron receptive fields

3D Grid Cells
^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_3d_manifold_exp
   
   # Generate 3D grid cell population
   exp = generate_3d_manifold_exp(
       n_neurons=50,
       duration=60,          # 1 minute (for demo)
       fps=20,
       field_sigma=0.1,      # receptive field size
       step_size=0.02,       # movement step
       momentum=0.8,         # movement momentum
       noise_std=0.2
   )
   
   # Access data if return_info=True
   # exp, info = generate_3d_manifold_exp(..., return_info=True)
   # position_3d = info['position']  # (3, n_frames)

Mixed Selectivity Population
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_mixed_population_exp
   
   # Generate population with mixed selectivity
   exp = generate_mixed_population_exp(
       n_neurons=100,
       manifold_fraction=0.6,    # 60% manifold cells
       manifold_type="2d_spatial",
       n_discrete_features=3,
       n_continuous_features=2,
       duration=1200,
       fps=30
   )
   
   # Access info if return_info=True
   # exp, info = generate_mixed_population_exp(..., return_info=True)
   # manifold_neurons = info['manifold_neurons']
   # feature_neurons = info['feature_neurons']


Ground Truth Information
------------------------

When using `return_info=True` parameter, synthetic experiments return ground truth data:

.. code-block:: python

   # Get ground truth information
   exp, info = generate_circular_manifold_exp(
       n_neurons=50,
       duration=600,
       return_info=True
   )
   
   # Access ground truth data from info dict
   # The exact contents depend on the generator used

Best Practices
--------------

1. **Reproducibility**: Always set random seed for reproducible results
2. **Realistic Parameters**: Use physiologically plausible parameters
3. **Validation**: Use ground truth to validate analysis methods
4. **Noise Levels**: Start with low noise, increase to test robustness
5. **Duration**: Ensure sufficient data for statistical analyses