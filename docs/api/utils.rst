Utilities Module
================

.. automodule:: driada.utils
   :no-members:
   :noindex:

General utility functions for data manipulation, visualization, signal processing,
and other common operations in neural data analysis.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   utils/data
   utils/visualization
   utils/signals
   utils/spatial
   utils/matrix
   utils/misc

Quick Links
-----------

**Data Manipulation**
   * :doc:`utils/data` - Data structures and I/O utilities
   * :func:`~driada.utils.data.rescale` - Rescale data to range
   * :func:`~driada.utils.data.get_hash` - Generate data hashes
   * :func:`~driada.utils.data.write_dict_to_hdf5` - Save to HDF5
   * :func:`~driada.utils.data.read_hdf5_to_dict` - Load from HDF5

**Visualization**
   * :doc:`utils/visualization` - Advanced plotting functions
   * :func:`~driada.utils.visual.plot_embedding_comparison` - Compare embeddings
   * :func:`~driada.utils.plot.make_beautiful` - Style matplotlib plots
   * :func:`~driada.utils.plot.create_default_figure` - Standard figure setup

**Signal Processing**
   * :doc:`utils/signals` - Signal analysis and generation
   * :func:`~driada.utils.signals.brownian` - Generate Brownian motion
   * :func:`~driada.utils.signals.approximate_entropy` - Compute ApEn
   * :func:`~driada.utils.signals.filter_signals` - Apply filters

**Spatial Analysis**
   * :doc:`utils/spatial` - Place field and spatial coding
   * :func:`~driada.utils.spatial.compute_rate_map` - Firing rate maps
   * :func:`~driada.utils.spatial.extract_place_fields` - Find place fields
   * :func:`~driada.utils.spatial.compute_spatial_information` - Spatial info

**Matrix Operations**
   * :doc:`utils/matrix` - Matrix utilities
   * :func:`~driada.utils.matrix.nearestPD` - Nearest positive definite matrix
   * :func:`~driada.utils.matrix.is_positive_definite` - Check PD property

Usage Example
-------------

.. code-block:: python

   from driada.utils import (
       rescale, make_beautiful, 
       compute_rate_map, brownian
   )
   
   # Data manipulation - rescales 1D data to [0, 1]
   import numpy as np
   data = np.random.randn(1000)
   normalized_data = rescale(data)
   
   # Visualization
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots()
   make_beautiful(ax)
   
   # Spatial analysis
   from driada.utils import compute_occupancy_map, compute_rate_map
   
   # First compute occupancy
   positions = np.random.randn(1000, 2)  # x,y positions
   neural_signal = np.random.randn(1000)  # neural activity
   occupancy, x_edges, y_edges = compute_occupancy_map(
       positions, 
       bin_size=0.1,
       fps=30.0
   )
   
   # Then compute rate map
   rate_map = compute_rate_map(
       neural_signal,
       positions,
       occupancy,
       x_edges,
       y_edges,
       fps=30.0
   )
   
   # Signal generation
   random_walk = brownian(x0=0.0, n=1000, dt=0.1)