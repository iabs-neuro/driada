Quick Start Guide
=================

This guide will help you get started with DRIADA in just a few minutes.

Installation
------------

.. code-block:: bash

   # Basic installation
   pip install driada

   # With GPU support (recommended for large datasets)
   pip install driada[gpu]

Getting Started with DRIADA
---------------------------

1. Generate Synthetic Data for Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start by generating synthetic data to understand DRIADA's capabilities:

.. code-block:: python

   import driada
   from driada.experiment import generate_circular_manifold_exp
   import numpy as np

   # Generate a population with head direction cells
   exp = generate_circular_manifold_exp(
       n_neurons=50,           # 50 head direction cells
       duration=60,            # 1 minute of recording (for demo)
       noise_std=0.1,          # 10% noise (std deviation)
       seed=42
   )

   # Or generate place cells in 2D environment
   from driada.experiment import generate_2d_manifold_exp
   
   exp = generate_2d_manifold_exp(
       n_neurons=64,           # 8x8 grid of place cells
       duration=900,           # 15 minutes of exploration
       fps=20.0,               # Frame rate
       seed=42
   )

   # Or create mixed populations
   from driada.experiment import generate_mixed_population_exp
   
   exp = generate_mixed_population_exp(
       n_neurons=100,
       manifold_type='circular',
       manifold_fraction=0.4,  # 40% manifold cells, 60% feature-selective
       duration=600,
       seed=42
   )

2. Analyze Single-Neuron Selectivity (INTENSE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Discover which neurons encode which variables:

.. code-block:: python

   # Discover which neurons encode which variables
   from driada.intense import compute_cell_feat_significance
   from driada.experiment import load_demo_experiment

   exp = load_demo_experiment()  # Load sample experiment
   stats, significance, info, results = compute_cell_feat_significance(
       exp,
       n_shuffles_stage1=100,    # Quick screening
       n_shuffles_stage2=1000,   # Rigorous validation
       ds=5,                     # Downsample by factor of 5 for speed
       verbose=True,
       allow_mixed_dimensions=True,
       find_optimal_delays=False # Skip temporal alignment for demo
   )

   # View results
   significant_neurons = exp.get_significant_neurons()
   print(f"Found {len(significant_neurons)} selective neurons")

   # Visualize selectivity
   if significant_neurons:
       from driada.intense.visual import plot_neuron_feature_pair
       
       neuron_id = list(significant_neurons.keys())[0]
       feature_name = significant_neurons[neuron_id][0]
       plot_neuron_feature_pair(exp, neuron_id, feature_name)

3. Estimate Intrinsic Dimensionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before applying dimensionality reduction, estimate the intrinsic dimensionality:

.. code-block:: python

   # Multiple methods for dimensionality estimation
   from driada.dimensionality import (
       eff_dim, pca_dimension, nn_dimension, correlation_dimension
   )
   from driada.experiment import load_demo_experiment

   # Load sample experiment
   exp = load_demo_experiment()

   # Get neural activity data (n_samples, n_features)
   neural_data = exp.calcium.scdata.T  # Transpose to standard format

   # Linear methods
   pca_90 = pca_dimension(neural_data, threshold=0.90)
   pca_95 = pca_dimension(neural_data, threshold=0.95)

   # Effective dimension (participation ratio)
   eff_d = eff_dim(neural_data, enable_correction=True, q=2)

   # Nonlinear methods - add small noise to avoid duplicate points
   neural_data_noisy = neural_data + 1e-8 * np.random.randn(*neural_data.shape)
   nn_dim = nn_dimension(neural_data_noisy, k=5)
   corr_dim = correlation_dimension(neural_data_noisy)

   print(f"PCA 90%: {pca_90} dims, PCA 95%: {pca_95} dims")
   print(f"Effective dim: {eff_d:.2f}")
   print(f"k-NN dimension: {nn_dim:.2f}")
   print(f"Correlation dimension: {corr_dim:.2f}")

4. Apply Dimensionality Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract low-dimensional representations of population activity:

.. code-block:: python

   from driada.experiment import load_demo_experiment
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Two approaches for dimensionality reduction:
   # 1. Direct method: exp.calcium.get_embedding() returns Embedding objects
   # 2. Experiment method: exp.create_embedding() returns numpy arrays and stores them
   
   # Approach 1: Direct dimensionality reduction on calcium data
   # exp.calcium is a MultiTimeSeries, which inherits from MVData
   # So it directly supports all dimensionality reduction methods!
   
   # Apply different DR methods directly on calcium data
   # PCA - captures linear variance
   pca_emb = exp.calcium.get_embedding(method='pca', dim=3)
   
   # Isomap - preserves geodesic distances
   iso_emb = exp.calcium.get_embedding(method='isomap', dim=2, n_neighbors=30)
   
   # UMAP - preserves local and global structure
   umap_emb = exp.calcium.get_embedding(method='umap', dim=2, 
                                       n_neighbors=50, min_dist=0.1)
   
   # t-SNE - emphasizes local structure
   tsne_emb = exp.calcium.get_embedding(method='tsne', dim=2, perplexity=30)
   
   # Access the coordinates from the Embedding object
   coords = pca_emb.coords.T  # (n_samples, n_dims)
   
   # For custom downsampling, create new MVData
   from driada.dim_reduction import MVData
   mvdata_ds = MVData(exp.calcium.data, downsampling=5)
   pca_ds = mvdata_ds.get_embedding(method='pca', dim=3)
   
   # Approach 2: Use experiment's create_embedding() to store embeddings
   # This is required for INTENSE analysis (compute_embedding_selectivity)
   # Returns numpy arrays instead of Embedding objects
   pca_array = exp.create_embedding('pca', n_components=3)
   umap_array = exp.create_embedding('umap', n_components=2)
   
   # These are now stored in exp.embeddings and can be retrieved:
   stored_pca = exp.get_embedding('pca')  # Returns dict with 'data' key
   print(f"Stored PCA shape: {stored_pca['data'].shape}")

5. Validate Manifold Quality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assess how well the embedding preserves the original structure:

.. code-block:: python

   from driada.dim_reduction import (
       knn_preservation_rate, trustworthiness, continuity
   )
   from driada.experiment import load_demo_experiment
   
   # Load sample experiment and create embedding
   exp = load_demo_experiment()
   pca_emb = exp.calcium.get_embedding(method='pca', dim=3)
   
   # Compare high-D and low-D representations
   # Need to transpose calcium data to get (n_samples, n_features)
   high_d = exp.calcium.scdata.T  # Original high-dimensional data
   low_d = pca_emb.coords.T        # Low-dimensional embedding from above
   
   # k-NN preservation: how many neighbors stay the same
   knn_score = knn_preservation_rate(high_d, low_d, k=10)
   
   # Trustworthiness: are close points in low-D truly close in high-D?
   trust = trustworthiness(high_d, low_d, k=10)
   
   # Continuity: are close points in high-D still close in low-D?
   cont = continuity(high_d, low_d, k=10)
   
   print(f"k-NN preservation: {knn_score:.3f}")
   print(f"Trustworthiness: {trust:.3f}")
   print(f"Continuity: {cont:.3f}")

6. Integrate Single-Cell and Population Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze how single neurons contribute to population embeddings:

.. code-block:: python

   # First, compute INTENSE selectivity for embedding components
   from driada.intense import compute_embedding_selectivity
   from driada.experiment import load_demo_experiment

   # Load sample experiment
   exp = load_demo_experiment()
   
   # First, store embeddings using create_embedding() if not already done
   # This returns numpy arrays and stores them for INTENSE analysis
   if 'pca' not in exp.embeddings['calcium']:
       exp.create_embedding('pca', n_components=3)
   if 'umap' not in exp.embeddings['calcium']:
       exp.create_embedding('umap', n_components=2)

   # Then analyze how neurons contribute to embedding components
   emb_results = compute_embedding_selectivity(
       exp, 
       embedding_methods=['pca', 'umap'],
       n_shuffles_stage1=100,
       n_shuffles_stage2=1000,
       find_optimal_delays=False  # Skip temporal alignment for demo
   )

   # Extract INTENSE results for functional organization analysis
   from driada.integration import get_functional_organization

   # Analyze PCA functional organization
   pca_org = get_functional_organization(
       exp, 
       'pca',
       intense_results=emb_results['pca']['intense_results']
   )

   print(f"Component importance: {pca_org['component_importance']}")
   print(f"Neurons participating: {pca_org['n_participating_neurons']}")

   # Compare multiple embeddings
   from driada.integration import compare_embeddings

   intense_dict = {
       'pca': emb_results['pca']['intense_results'],
       'umap': emb_results['umap']['intense_results']
   }

   comparison = compare_embeddings(
       exp, 
       ['pca', 'umap'],
       intense_results_dict=intense_dict
   )

   # Visualize embeddings with features
   from driada.utils.visual import plot_embedding_comparison

   # Load experiment and create embeddings if needed
   exp = load_demo_experiment()
   
   # First create the embeddings if not already done
   if 'pca' not in exp.embeddings['calcium']:
       exp.create_embedding('pca', n_components=3)
   if 'umap' not in exp.embeddings['calcium']:
       exp.create_embedding('umap', n_components=2)

   # Get the stored numpy arrays from the experiment
   embeddings = {
       'PCA': exp.get_embedding('pca')['data'],  # Get stored numpy array
       'UMAP': exp.get_embedding('umap')['data']  # Get stored numpy array
   }

   # Color by a behavioral feature (ensure lengths match)
   features = {}
   if 'position_2d' in exp.dynamic_features:
       pos = exp.dynamic_features['position_2d'].data
       angle = np.arctan2(pos[1] - 0.5, pos[0] - 0.5)
       # Handle downsampling if embeddings were downsampled
       if hasattr(exp.calcium, 'downsampling'):
           ds = exp.calcium.downsampling
           features['angle'] = angle[::ds]
       else:
           features['angle'] = angle

   fig = plot_embedding_comparison(
       embeddings=embeddings,
       features=features,
       compute_metrics=True,
       figsize=(12, 5)
   )

7. Network Analysis: Cell-Cell Functional Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identify functional networks by analyzing pairwise neural correlations:

.. code-block:: python

   from driada.intense import compute_cell_cell_significance
   from driada.network import Network
   from driada.experiment import load_demo_experiment
   import scipy.sparse as sp
   
   # Load sample experiment
   exp = load_demo_experiment()
   
   # Compute pairwise functional connectivity
   # Uses mutual information to measure dependencies
   results = compute_cell_cell_significance(
       exp,
       n_shuffles_stage1=100,    # Quick screening
       n_shuffles_stage2=1000,   # Rigorous validation  
       ds=5,                     # Downsample for speed
       verbose=True
   )
   
   sim_mat, sig_mat, pval_mat, cells, info = results
   
   # sig_mat is binary: 1 = significant correlation, 0 = not significant
   n_connections = np.sum(sig_mat)
   print(f"Found {n_connections} significant connections")
   print(f"Network density: {n_connections / (len(cells)**2 - len(cells)):.3f}")
   
   # Create network from significant connections
   sig_sparse = sp.csr_matrix(sig_mat)
   net = Network(adj=sig_sparse, preprocessing='giant_cc')
   
   # Analyze network properties
   print(f"Network has {net.n} nodes in giant component")
   print(f"Average degree: {net.deg.mean():.2f}")
   
   # Detect functional modules
   from sklearn.cluster import SpectralClustering
   
   if net.n > 10:
       # Use spectral clustering on the network
       clustering = SpectralClustering(
           n_clusters=3, 
           affinity='precomputed',
           random_state=42
       )
       modules = clustering.fit_predict(net.adj.toarray())
       
       print(f"Detected {len(np.unique(modules))} functional modules")
   
   # Visualize network (for smaller networks)
   if net.n < 50:
       import networkx as nx
       import matplotlib.pyplot as plt
       
       G = nx.from_scipy_sparse_array(net.adj)
       pos = nx.spring_layout(G, seed=42)
       
       plt.figure(figsize=(10, 8))
       nx.draw_networkx_nodes(G, pos, node_size=300, 
                              node_color='lightblue', alpha=0.7)
       nx.draw_networkx_edges(G, pos, alpha=0.5)
       nx.draw_networkx_labels(G, pos, font_size=8)
       plt.title(f"Functional Network ({net.n} neurons)")
       plt.axis('off')
       plt.tight_layout()

8. Working with Real Data
^^^^^^^^^^^^^^^^^^^^^^^^^

Load and analyze your own neural recordings:

.. code-block:: python

   import numpy as np
   from driada import load_exp_from_aligned_data
   
   # Load data from NPZ file (recommended format)
   data = dict(np.load(sample_npz_path))
   # Expected structure in sample_recording.npz:
   # - data['calcium']: (50, 10000) - neural activity, REQUIRED
   # - data['position']: (2, 10000) - 2D trajectory (x,y coordinates)
   # - data['x_pos']: (10000,) - x coordinate
   # - data['y_pos']: (10000,) - y coordinate  
   # - data['speed']: (10000,) - movement speed
   # - data['trial_type']: (10000,) - discrete labels ('A', 'B', 'C', 'D')
   # - data['head_direction']: (10000,) - circular variable (radians)
   # - data['fps']: scalar - frame rate (30.0)
   
   # Create experiment with automatic feature detection
   exp = load_exp_from_aligned_data(
       data_source='MyLab',  # Can be any lab identifier (e.g., 'MyLab', 'NeuroLab')
       exp_params={'name': 'my_experiment'},  # For custom labs, use 'name' key
       data=data,
       static_features={'fps': 30.0},  # Recording frame rate
       force_continuous=['trial_type'],  # Override auto-detection if needed
       bad_frames=[100, 101, 102],  # Mark corrupted frames
       reconstruct_spikes='wavelet'  # Automatic spike deconvolution
   )
   
   # For HDF5 files (example with your own file)
   # from driada.utils.data import read_hdf5_to_dict
   # data = read_hdf5_to_dict('path/to/your/recording.h5')
   # exp = load_exp_from_aligned_data(
   #     data_source='my_lab', 
   #     exp_params={'name': 'my_experiment'},
   #     data=data
   # )
   
   # For multi-dimensional features (e.g., 2D position)
   from driada.information.info_base import MultiTimeSeries
   
   # Combine x,y coordinates into single feature
   spatial_data = np.stack([data['x_pos'], data['y_pos']])
   spatial_feature = MultiTimeSeries(
       spatial_data,
       discrete=False
   )
   
   # Add to data dictionary
   data['position_2d'] = spatial_feature
   exp = load_exp_from_aligned_data(
       data_source='my_lab',
       exp_params={'name': 'my_experiment'},
       data=data
   )

9. Advanced Analysis Workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Leverage DRIADA's advanced capabilities:

.. code-block:: python

   # Sequential dimensionality reduction pipeline
   from driada.dim_reduction import dr_sequence
   from driada.experiment import load_demo_experiment
   
   exp = load_demo_experiment()
   
   # Chain multiple DR methods for optimal results
   embedding = dr_sequence(
       exp.calcium,
       steps=[
           ('pca', {'dim': 20}),     # Initial denoising
           ('umap', {'dim': 3, 'n_neighbors': 15})  # Final embedding
       ],
       keep_intermediate=True  # Access results from each step
   )
   
   # Access intermediate results (returns tuple when keep_intermediate=True)
   final_embedding, intermediate_results = embedding
   pca_result = intermediate_results[0]
   umap_result = intermediate_results[1]
   final_coords = final_embedding.coords.T
   
   # High-precision INTENSE analysis with mixed features
   from driada.intense import compute_cell_feat_significance
   
   results = compute_cell_feat_significance(
       exp,
       mode='two_stage',
       n_shuffles_stage1=100,     # Pre-screening
       n_shuffles_stage2=5000,    # High precision
       allow_mixed_dimensions=True,  # Handle MultiTimeSeries
       find_optimal_delays=False,  # Skip temporal alignment
       ds=5,  # Downsample for speed
       verbose=True
   )
   
   # Save complete analysis results
   from driada.utils.data import write_dict_to_hdf5
   
   # Package all results
   analysis_results = {
       'experiment_signature': exp.signature,
       'intense_stats': results[0],
       'intense_significance': results[1],
       'embedding_coords': {
           'pca': pca_result.coords.T,  # Convert to numpy array
           'umap': final_embedding.coords.T  # Convert to numpy array
       },
       'significant_neurons': exp.get_significant_neurons()
   }
   
   # Save to HDF5 (remove file if exists for clean save)
   import os
   if os.path.exists('analysis_results.h5'):
       os.remove('analysis_results.h5')
   write_dict_to_hdf5(analysis_results, 'analysis_results.h5')
   

Next Steps
----------

Explore comprehensive examples demonstrating real-world workflows:

**Getting Started:**

- ``examples/basic_usage/basic_usage.py`` - Basic DRIADA workflow with synthetic data
- ``examples/dr_simplified_api/dr_simplified_api_demo.py`` - Simple dimensionality reduction API usage

**Core Analysis Workflows:**

- ``examples/circular_manifold/extract_circular_manifold.py`` - Extract ring attractor structure from head direction cells
- ``examples/circular_manifold/test_metrics.py`` - Validate circular manifold reconstruction quality
- ``examples/spatial_map/extract_spatial_map.py`` - Analyze place cells and spatial representations
- ``examples/spatial_analysis/visualize_spatial_maps.py`` - Visualize spatial coding properties
- ``examples/task_variables/extract_task_variables.py`` - Decode task variables from mixed selectivity populations
- ``examples/network_analysis/cell_cell_network_example.py`` - Build and analyze functional networks

**Dimensionality Reduction:**

- ``examples/compare_dr_methods/compare_dr_methods.py`` - Systematic comparison of DR algorithms
- ``examples/dr_sequence/dr_sequence_neural_example.py`` - Sequential DR pipeline for optimal results
- ``examples/recursive_embedding/recursive_embedding_example.py`` - Multi-scale manifold analysis

**Complete Pipelines:**

- ``examples/full_pipeline/full_pipeline.py`` - Complete INTENSE + DR workflow from start to finish
- ``examples/intense_dr_pipeline/intense_dr_pipeline.py`` - Integration of single-cell and population analysis
- ``examples/mixed_selectivity/mixed_selectivity.py`` - Analyze neurons with mixed feature selectivity

**Advanced Techniques:**

- ``examples/spike_reconstruction/spike_reconstruction_comparison.py`` - Compare spike deconvolution methods
- ``examples/rsa/rsa_example.py`` - Representational similarity analysis for comparing neural codes
- ``examples/visual_utils/visual_utils_demo.py`` - Advanced visualization utilities and techniques

**Experimental (Under Construction):**

- ``examples/under_construction/selectivity_manifold_mapper/`` - Map selectivity to manifold structure

For more information:

- Read the :doc:`api/index` for comprehensive documentation
- Check out :doc:`tutorials` for in-depth guides
- Visit our `GitHub repository <https://github.com/iabs-neuro/driada>`_ for latest updates
- Join our community for support and discussions