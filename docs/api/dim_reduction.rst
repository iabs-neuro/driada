Dimensionality Reduction Module
================================

.. automodule:: driada.dim_reduction
   :no-members:
   :noindex:

Comprehensive dimensionality reduction tools for analyzing high-dimensional neural data,
including classical methods (PCA, FA), manifold learning (Isomap, UMAP), and neural network
approaches (autoencoders).

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
   * :class:`~driada.dim_reduction.MVData` - Multivariate data container
   * :class:`~driada.dim_reduction.Embedding` - Embedding results
   * :class:`~driada.dim_reduction.ProximityGraph` - Graph-based methods
   * :class:`~driada.dim_reduction.DRMethod` - Method base class

**Main Functions**
   * :func:`~driada.dim_reduction.dr_sequence` - Sequential DR pipeline
   * See :data:`~driada.dim_reduction.METHODS_DICT` for available methods

**Manifold Quality Metrics**
   * :doc:`dim_reduction/manifold_metrics` - Comprehensive validation tools
   * Preservation metrics (k-NN, trustworthiness, continuity)
   * Reconstruction accuracy metrics
   * Structure preservation measures

**Neural Network Methods**
   * :doc:`dim_reduction/neural_methods` - Autoencoder implementations
   * Flexible architecture design
   * Custom loss functions

Usage Example
-------------

.. code-block:: python

   from driada.dim_reduction import MVData
   
   # Create data container
   mvdata = MVData(neural_data, downsampling=5)
   
   # Apply dimensionality reduction
   embedding = mvdata.get_embedding(method='umap', n_components=3)
   
   # Validate quality
   from driada.dim_reduction import knn_preservation_rate
   quality = knn_preservation_rate(neural_data, embedding.coords.T, k=10)