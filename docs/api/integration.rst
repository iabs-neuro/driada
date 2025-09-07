Integration Module
==================

.. automodule:: driada.integration
   :no-members:
   :noindex:

Bridge between single-neuron selectivity analysis (INTENSE) and population-level
dimensionality reduction, enabling integrated analysis of neural data.

Main Functions
--------------

.. autofunction:: driada.integration.manifold_analysis.get_functional_organization
.. autofunction:: driada.integration.manifold_analysis.compare_embeddings

Usage Example
-------------

.. code-block:: python

   from driada.integration import get_functional_organization, compare_embeddings
   from driada.intense import compute_cell_feat_significance
   from driada.experiment import load_demo_experiment

   # Load sample experiment
   exp = load_demo_experiment()

   # First, run INTENSE analysis
   stats, sig, info, intense_results = compute_cell_feat_significance(
       exp,
       allow_mixed_dimensions=True,
       find_optimal_delays=False  # Skip temporal alignment for demo
   )

   # Then create and store embeddings
   pca_array = exp.create_embedding('pca', n_components=3)
   umap_array = exp.create_embedding('umap', n_components=2)

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

   print(f"Component importance: {pca_org['component_importance']}")
   print(f"PCA vs UMAP overlap: {comparison['participation_overlap']['pca_vs_umap']:.3f}")