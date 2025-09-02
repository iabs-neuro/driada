Integration Module
==================

.. automodule:: driada.integration
   :no-members:
   :noindex:

Bridge between single-neuron selectivity analysis (INTENSE) and population-level
dimensionality reduction, enabling integrated analysis of neural data.

Main Functions
--------------

.. autofunction:: driada.integration.get_functional_organization
.. autofunction:: driada.integration.compare_embeddings

Usage Example
-------------

.. code-block:: python

   from driada.integration import get_functional_organization, compare_embeddings
   from driada.intense import compute_cell_feat_significance

   # Assume exp is an Experiment object already created
   # exp = Experiment(...) # See Experiment docs for full parameters

   # First, run INTENSE analysis
   stats, sig, info, intense_results = compute_cell_feat_significance(exp)

   # Then create embeddings
   pca_emb = exp.calcium.get_embedding(method='pca', dim=3)
   umap_emb = exp.calcium.get_embedding(method='umap', n_components=2)

   # Analyze functional organization in PCA space
   pca_org = get_functional_organization(
       exp, 
       'pca',
       intense_results=intense_results
   )

   # Compare multiple embeddings
   comparison = compare_embeddings(
       exp,
       ['pca', 'umap'],
       intense_results_dict={
           'pca': intense_results,
           'umap': intense_results
       }
   )

   print(f"PCA captures {pca_org['variance_explained']:.1%} of selectivity")
   print(f"UMAP similarity to PCA: {comparison['similarity_matrix'][0,1]:.3f}")