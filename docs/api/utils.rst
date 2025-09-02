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
   * :func:`~driada.utils.rescale` - Rescale data to range
   * :func:`~driada.utils.get_hash` - Generate data hashes
   * :func:`~driada.utils.write_dict_to_hdf5` - Save to HDF5
   * :func:`~driada.utils.read_hdf5_to_dict` - Load from HDF5

**Visualization**
   * :doc:`utils/visualization` - Advanced plotting functions
   * :func:`~driada.utils.plot_embedding_comparison` - Compare embeddings
   * :func:`~driada.utils.make_beautiful` - Style matplotlib plots
   * :func:`~driada.utils.create_default_figure` - Standard figure setup

**Signal Processing**
   * :doc:`utils/signals` - Signal analysis and generation
   * :func:`~driada.utils.brownian` - Generate Brownian motion
   * :func:`~driada.utils.approximate_entropy` - Compute ApEn
   * :func:`~driada.utils.filter_signals` - Apply filters

**Spatial Analysis**
   * :doc:`utils/spatial` - Place field and spatial coding
   * :func:`~driada.utils.compute_rate_map` - Firing rate maps
   * :func:`~driada.utils.extract_place_fields` - Find place fields
   * :func:`~driada.utils.compute_spatial_information` - Spatial info

**Matrix Operations**
   * :doc:`utils/matrix` - Matrix utilities
   * :func:`~driada.utils.nearestPD` - Nearest positive definite matrix
   * :func:`~driada.utils.is_positive_definite` - Check PD property

Usage Example
-------------

.. code-block:: python

   from driada.utils import (
       rescale, make_beautiful, 
       compute_rate_map, brownian
   )
   
   # Data manipulation
   normalized_data = rescale(data, 0, 1)
   
   # Visualization
   fig, ax = plt.subplots()
   make_beautiful(ax)
   
   # Spatial analysis
   rate_map = compute_rate_map(
       spike_times, 
       positions, 
       bin_size=5
   )
   
   # Signal generation
   random_walk = brownian(n_steps=1000, dt=0.1)