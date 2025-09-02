Manifold Metrics
================

.. automodule:: driada.dim_reduction.manifold_metrics

Tools for evaluating the quality of dimensionality reduction embeddings.

Distance and Structure Metrics
------------------------------

.. autofunction:: compute_distance_matrix
.. autofunction:: circular_distance
.. autofunction:: circular_diff

Preservation Metrics
--------------------

.. autofunction:: knn_preservation_rate
.. autofunction:: trustworthiness
.. autofunction:: continuity
.. autofunction:: geodesic_distance_correlation
.. autofunction:: stress
.. autofunction:: manifold_preservation_score

Circular Manifold Analysis
--------------------------

.. autofunction:: circular_structure_preservation
.. autofunction:: extract_angles_from_embedding
.. autofunction:: find_optimal_circular_alignment
.. autofunction:: compute_circular_correlation

Reconstruction and Alignment
----------------------------

.. autofunction:: compute_reconstruction_error
.. autofunction:: compute_embedding_alignment_metrics
.. autofunction:: procrustes_analysis
.. autofunction:: manifold_reconstruction_score

Decoding and Quality Assessment
-------------------------------

.. autofunction:: compute_decoding_accuracy
.. autofunction:: compute_embedding_quality
.. autofunction:: train_simple_decoder