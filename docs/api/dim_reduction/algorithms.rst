Dimensionality Reduction Algorithms
===================================

Core algorithms and method registry for dimensionality reduction.

Method Registry
---------------

.. autodata:: driada.dim_reduction.METHODS_DICT
   :annotation: = {'pca': DRMethod(...), 'le': DRMethod(...), ...}

   Dictionary mapping method names to DRMethod instances. Available methods include:
   
   * ``pca`` - Principal Component Analysis
   * ``le`` - Laplacian Eigenmaps
   * ``auto_le`` - Auto-tuned Laplacian Eigenmaps
   * ``dmaps`` - Diffusion Maps
   * ``isomap`` - Isometric Feature Mapping
   * ``tsne`` - t-Distributed Stochastic Neighbor Embedding
   * ``umap`` - Uniform Manifold Approximation and Projection
   * (and others - see source for complete list)

Base Classes
------------

.. autoclass:: driada.dim_reduction.DRMethod
   :members:
   :special-members: __init__

Sequential Processing
---------------------

.. autofunction:: driada.dim_reduction.dr_sequence

Helper Functions
----------------

.. autofunction:: driada.dim_reduction.merge_params_with_defaults
.. autofunction:: driada.dim_reduction.e_param_filter
.. autofunction:: driada.dim_reduction.g_param_filter
.. autofunction:: driada.dim_reduction.m_param_filter