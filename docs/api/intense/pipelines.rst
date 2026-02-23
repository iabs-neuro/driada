INTENSE Pipelines
=================

.. automodule:: driada.intense.pipelines

High-level analysis pipelines for computing statistical significance of neural selectivity.

All pipeline functions are importable from the top-level ``driada.intense`` namespace:

.. doctest::

   >>> from driada.intense import compute_cell_feat_significance
   >>> from driada.intense import compute_cell_cell_significance
   >>> from driada.intense import compute_embedding_selectivity
   >>> import inspect
   >>> 'n_shuffles_stage1' in inspect.signature(compute_cell_feat_significance).parameters
   True
   >>> 'n_shuffles_stage2' in inspect.signature(compute_cell_cell_significance).parameters
   True

``compute_cell_cell_significance`` produces pairwise similarity and significance matrices.
The significance matrix can be wrapped in a :class:`~driada.network.net_base.Network` for
spectral and topological analysis:

.. code-block::

   sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
       exp, n_shuffles_stage1=100, n_shuffles_stage2=1000, ds=5
   )

   import scipy.sparse as sp
   from driada.network import Network

   net = Network(adj=sp.csr_matrix(sig_mat), preprocessing='giant_cc')
   net.diagonalize(mode='nlap')
   spectrum = net.get_spectrum('nlap')

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
   from driada.experiment import load_demo_experiment

   exp = load_demo_experiment()

   stats, significance, info, results = compute_cell_feat_significance(
       exp,
       n_shuffles_stage1=100,
       n_shuffles_stage2=1000,
       ds=5,
       find_optimal_delays=False,
   )
