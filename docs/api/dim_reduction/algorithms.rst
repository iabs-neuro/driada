Dimensionality Reduction Algorithms
===================================

Core algorithms and method registry for dimensionality reduction.

Method Registry
---------------

.. data:: driada.dim_reduction.dr_base.METHODS_DICT
   :annotation: = dict

   Dictionary mapping method names to their DRMethod configurations.
   
   Available methods:
   
   * ``pca`` - Principal Component Analysis
   * ``mds`` - Multi-dimensional Scaling  
   * ``isomap`` - Isometric Feature Mapping
   * ``lle`` - Locally Linear Embedding
   * ``hlle`` - Hessian Locally Linear Embedding
   * ``le`` - Laplacian Eigenmaps
   * ``dmaps`` - Diffusion Maps
   * ``mvu`` - Maximum Variance Unfolding
   * ``tsne`` - t-Distributed Stochastic Neighbor Embedding
   * ``umap`` - Uniform Manifold Approximation and Projection
   * ``ae`` - Autoencoder
   * ``vae`` - Variational Autoencoder

Base Classes
------------

.. autoclass:: driada.dim_reduction.dr_base.DRMethod
   :members:
   :special-members: __init__

Sequential Processing
---------------------

.. autofunction:: driada.dim_reduction.sequences.dr_sequence

Helper Functions
----------------

.. autofunction:: driada.dim_reduction.dr_base.merge_params_with_defaults
.. autofunction:: driada.dim_reduction.dr_base.e_param_filter
.. autofunction:: driada.dim_reduction.dr_base.g_param_filter
.. autofunction:: driada.dim_reduction.dr_base.m_param_filter