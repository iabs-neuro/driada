Dimensionality Reduction Module
================================

.. automodule:: driada.dim_reduction
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

.. autofunction:: driada.dim_reduction.dr_sequence

Core Classes
------------

.. autoclass:: driada.dim_reduction.MVData
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.ProximityGraph
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.Embedding
   :members:
   :special-members: __init__

.. autoclass:: driada.dim_reduction.DRMethod
   :members:

Constants
---------

.. autodata:: driada.dim_reduction.METHODS_DICT

Manifold Metrics
----------------

.. autofunction:: driada.dim_reduction.compute_distance_matrix
.. autofunction:: driada.dim_reduction.knn_preservation_rate
.. autofunction:: driada.dim_reduction.trustworthiness
.. autofunction:: driada.dim_reduction.continuity
.. autofunction:: driada.dim_reduction.geodesic_distance_correlation
.. autofunction:: driada.dim_reduction.stress
.. autofunction:: driada.dim_reduction.circular_structure_preservation
.. autofunction:: driada.dim_reduction.procrustes_analysis
.. autofunction:: driada.dim_reduction.manifold_preservation_score

Reconstruction Validation
-------------------------

.. autofunction:: driada.dim_reduction.circular_distance
.. autofunction:: driada.dim_reduction.extract_angles_from_embedding
.. autofunction:: driada.dim_reduction.compute_reconstruction_error
.. autofunction:: driada.dim_reduction.compute_embedding_alignment_metrics
.. autofunction:: driada.dim_reduction.compute_decoding_accuracy
.. autofunction:: driada.dim_reduction.manifold_reconstruction_score

Submodules
----------

Data Structures
^^^^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.data
   :members:
   :undoc-members:

Base Classes
^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.dr_base
   :members:
   :undoc-members:

Graph-Based Methods
^^^^^^^^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.graph
   :members:
   :undoc-members:

.. automodule:: driada.dim_reduction.embedding
   :members:
   :undoc-members:

Neural Network Methods
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.neural
   :members:
   :undoc-members:

.. automodule:: driada.dim_reduction.flexible_ae
   :members:
   :undoc-members:

.. automodule:: driada.dim_reduction.losses
   :members:
   :undoc-members:

Optimization-Based Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.mvu
   :members:
   :undoc-members:

Sequence Processing
^^^^^^^^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.sequences
   :members:
   :undoc-members:

Manifold Metrics
^^^^^^^^^^^^^^^^

.. automodule:: driada.dim_reduction.manifold_metrics
   :members:
   :undoc-members: