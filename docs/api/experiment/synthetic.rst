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
       n_neurons=100,
       duration=600,      # 10 minutes
       fps=30,           # 30 Hz sampling
       noise_std=0.1,    # Noise level
       seed=42           # For reproducibility
   )
   
   print(f"Generated {exp.n_neurons} neurons, {exp.duration}s recording")

Head Direction Cells (Circular Manifold)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_circular_manifold_exp
   
   # Generate head direction cell population
   exp = generate_circular_manifold_exp(
       n_neurons=50,
       duration=600,
       fps=30,
       tuning_width=30,      # degrees
       noise_std=0.1,
       rotation_speed=10     # deg/s
   )
   
   # Access ground truth
   true_angle = exp.behavior['head_direction']
   preferred_directions = exp.info['preferred_directions']
   
   # Verify tuning
   import numpy as np
   from scipy.stats import circmean
   
   # Compute tuning curves
   angles = np.linspace(0, 360, 36)
   tuning_curves = compute_tuning_curves(exp, angles)

2D Place Cells
^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_2d_manifold_exp
   
   # Generate place cell population
   exp = generate_2d_manifold_exp(
       n_neurons=100,
       duration=1200,        # 20 minutes
       fps=30,
       field_size=20,        # cm
       arena_size=(100, 100), # cm
       movement_speed=10,    # cm/s
       noise_std=0.15
   )
   
   # Access position and place fields
   position = exp.behavior['position']  # (2, n_frames)
   place_fields = exp.info['place_fields']  # List of (x, y, size) tuples
   
   # Compute rate maps
   from driada.utils import compute_rate_map
   rate_maps = []
   for neuron_id in range(exp.n_neurons):
       rate_map = compute_rate_map(
           exp.calcium.data[neuron_id, :],
           position,
           bin_size=5  # cm
       )
       rate_maps.append(rate_map)

3D Grid Cells
^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_3d_manifold_exp
   
   # Generate 3D grid cell population
   exp = generate_3d_manifold_exp(
       n_neurons=150,
       duration=1800,        # 30 minutes
       fps=30,
       grid_scale=50,        # cm
       arena_size=(200, 200, 100),  # cm
       flight_speed=20,      # cm/s
       noise_std=0.2
   )
   
   # Access 3D trajectory
   position_3d = exp.behavior['position']  # (3, n_frames)
   grid_phases = exp.info['grid_phases']   # Grid cell phase offsets

Mixed Selectivity Population
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from driada.experiment import generate_mixed_population_exp
   
   # Generate population with mixed selectivity
   exp = generate_mixed_population_exp(
       population_spec={
           'place_cells': 40,
           'head_direction': 30,
           'speed_cells': 20,
           'conjunctive': 10  # place × head direction
       },
       duration=1200,
       fps=30,
       arena_size=(100, 100),
       noise_std=0.1
   )
   
   # Access cell type labels
   cell_types = exp.info['cell_types']
   
   # Analyze different populations
   place_cell_ids = [i for i, t in enumerate(cell_types) if t == 'place_cells']
   hd_cell_ids = [i for i, t in enumerate(cell_types) if t == 'head_direction']


Advanced Options
----------------

Custom Tuning Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Define custom tuning function
   def custom_tuning(stimulus, preferred_stim, **params):
       width = params.get('width', 1.0)
       gain = params.get('gain', 1.0)
       baseline = params.get('baseline', 0.1)
       
       response = gain * np.exp(-(stimulus - preferred_stim)**2 / (2 * width**2))
       return response + baseline
   
   # Use in generation
   exp = generate_synthetic_exp(
       n_neurons=50,
       duration=600,
       tuning_function=custom_tuning,
       tuning_params={'width': 0.5, 'gain': 2.0}
   )

Structured Connectivity
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Generate with specific connectivity
   from driada.experiment.synthetic import generate_connected_population
   
   exp = generate_connected_population(
       n_neurons=100,
       duration=600,
       connectivity_type='small_world',
       connection_probability=0.1,
       connection_strength=0.3
   )
   
   # Access connectivity matrix
   W = exp.info['connectivity_matrix']

Ground Truth Validation
^^^^^^^^^^^^^^^^^^^^^^^

All synthetic experiments include ground truth information:

- **True spike times**: Available in `exp.info['true_spikes']`
- **Tuning parameters**: Stored in `exp.info['tuning_params']`
- **Generative parameters**: All parameters used for generation
- **Latent variables**: True latent states/positions

This enables validation of analysis methods:

.. code-block:: python

   # Validate dimensionality reduction
   true_manifold = exp.info['true_manifold']
   embedding = exp.calcium.get_embedding(method='umap', n_components=2)

   # Compare with ground truth
   from scipy.stats import spearmanr

   # Assume exp is an Experiment object already created
   # exp = Experiment(...) # See Experiment docs for full parameters
   correlation = spearmanr(true_manifold.flat, embedding.data.flat)[0]
   print(f"Embedding correlation with truth: {correlation:.3f}")

Best Practices
--------------

1. **Reproducibility**: Always set random seed for reproducible results
2. **Realistic Parameters**: Use physiologically plausible parameters
3. **Validation**: Use ground truth to validate analysis methods
4. **Noise Levels**: Start with low noise, increase to test robustness
5. **Duration**: Ensure sufficient data for statistical analyses