Dimensionality Reduction Module
================================

.. automodule:: driada.dim_reduction
   :no-members:
   :noindex:

Comprehensive dimensionality reduction tools for analyzing high-dimensional neural data,
including classical methods (PCA, FA), manifold learning (Isomap, UMAP), and neural network
approaches (autoencoders).

Graph-based DR methods (Isomap, LLE, Laplacian Eigenmaps, UMAP, diffusion maps) construct
a :class:`~driada.dim_reduction.graph.ProximityGraph` internally, which inherits from
:class:`~driada.network.net_base.Network`. This means the proximity graph powering your
embedding has full spectral analysis, entropy, community detection, and visualization
capabilities. Access it via ``embedding.graph`` after running a graph-based method.

Module Components
-----------------

.. toctree::
   :maxdepth: 1

   dim_reduction/data_structures
   dim_reduction/algorithms
   dim_reduction/manifold_metrics
   dim_reduction/neural_methods
   dim_reduction/utilities

Quick Links
-----------

**Core Classes**
   * :class:`~driada.dim_reduction.data.MVData` - Multivariate data container
   * :class:`~driada.dim_reduction.embedding.Embedding` - Embedding results
   * :class:`~driada.dim_reduction.graph.ProximityGraph` - Graph-based methods (inherits from :class:`~driada.network.net_base.Network`)
   * :class:`~driada.dim_reduction.dr_base.DRMethod` - Method base class

**Main Functions**
   * :func:`~driada.dim_reduction.sequences.dr_sequence` - Sequential DR pipeline
   * See :data:`~driada.dim_reduction.dr_base.METHODS_DICT` for available methods

**Manifold Quality Metrics**
   * :doc:`dim_reduction/manifold_metrics` - Preservation, trustworthiness, continuity, and reconstruction metrics

**Neural Network Methods**
   * :doc:`dim_reduction/neural_methods` - Autoencoders with flexible architecture and custom losses

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from driada.dim_reduction import MVData
   
   # Generate example neural data
   # 100 neurons, 1000 time points
   neural_data = np.random.randn(100, 1000)
   
   # Create data container
   mvdata = MVData(neural_data, downsampling=5)
   
   # Apply dimensionality reduction
   embedding = mvdata.get_embedding(method='umap', dim=3)
   
   # Validate quality
   from driada.dim_reduction import knn_preservation_rate
   quality = knn_preservation_rate(mvdata.data.T, embedding.coords.T, k=10)