INTENSE Pipelines
=================

.. automodule:: driada.intense.pipelines

High-level analysis pipelines for computing statistical significance of neural selectivity.

Main Functions
--------------

.. autofunction:: compute_cell_feat_significance
.. autofunction:: compute_feat_feat_significance
.. autofunction:: compute_cell_cell_significance
.. autofunction:: compute_embedding_selectivity

Usage Example
-------------

.. code-block:: python

   from driada.intense import compute_cell_feat_significance
   
   # Analyze neuron-feature selectivity
   stats, significance, info, results = compute_cell_feat_significance(
       exp,
       n_shuffles_stage1=100,
       n_shuffles_stage2=1000,
       ds=5  # Downsample for speed
   )