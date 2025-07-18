"""Integration tests for DRIADA modules working together"""

import pytest
import numpy as np
import warnings
from driada import (
    Experiment, 
    compute_cell_feat_significance,
    generate_circular_manifold_exp,
    generate_2d_manifold_exp,
    generate_mixed_population_exp
)
from driada.dimensionality import nn_dimension, pca_dimension, effective_rank
from driada.dim_reduction import MVData
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap


class TestINTENSEToDRIntegration:
    """Test INTENSE → Dimensionality Reduction pipeline"""
    
    def test_intense_to_pca_pipeline(self):
        """Test full pipeline from INTENSE selectivity to PCA reduction"""
        # Generate synthetic data with known selectivity
        exp = generate_circular_manifold_exp(
            n_neurons=30,
            duration=300,
            noise_std=0.1,
            seed=42
        )
        
        # Run INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=50,
            n_shuffles_stage2=100,
            allow_mixed_dimensions=True,
            find_optimal_delays=False,  # Can't use with multifeatures
            verbose=False
        )
        
        # Get significant neurons
        sig_neurons = exp.get_significant_neurons()
        assert len(sig_neurons) > 0, "No significant neurons found"
        
        # Extract neural data for DR
        neural_data = exp.calcium
        
        # Apply PCA
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(neural_data.T)
        
        # Check embedding quality
        assert embedding.shape == (exp.n_frames, 2)
        assert pca.explained_variance_ratio_[0] > 0.1  # First PC explains >10%
        
    def test_intense_to_manifold_learning(self):
        """Test INTENSE with nonlinear DR methods"""
        # Generate 2D spatial data
        exp = generate_2d_manifold_exp(
            n_neurons=25,
            duration=200,
            n_environments=1,
            noise_std=0.2,
            seed=123
        )
        
        # INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=50,
            n_shuffles_stage2=100,
            verbose=False
        )
        
        # Extract selective neurons' activity
        sig_neurons = exp.get_significant_neurons()
        if len(sig_neurons) > 0:
            selective_ids = list(sig_neurons.keys())
            neural_subset = exp.calcium[selective_ids, :]
        else:
            neural_subset = exp.calcium
            
        # Apply Isomap
        iso = Isomap(n_components=2, n_neighbors=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = iso.fit_transform(neural_subset.T)
        
        assert embedding.shape[1] == 2
        assert not np.any(np.isnan(embedding))
        
    def test_mvdata_experiment_integration(self):
        """Test MVData integration with Experiment objects"""
        # Create experiment
        exp = generate_mixed_population_exp(
            n_neurons=50,
            manifold_type='circular',
            manifold_fraction=0.5,
            duration=300,
            seed=42
        )
        
        # Convert to MVData
        mvdata = MVData(exp.calcium)
        
        # Test multiple DR methods
        methods = {
            'pca': PCA(n_components=3),
            'isomap': {'nn': 10},
            'umap': {'nn': 15}
        }
        
        for method_name, method in methods.items():
            if isinstance(method, dict):
                # Graph-based methods
                params = {
                    'e_method_name': method_name,
                    'dim': 2,
                    'graph_params': {
                        'g_method_name': 'knn',
                        'nn': method['nn'],
                        'weighted': False,
                        'max_deleted_nodes': 0.5
                    },
                    'metric_params': {
                        'metric_name': 'euclidean'
                    }
                }
                embedding = mvdata.get_embedding(params)
            else:
                # Direct methods like PCA
                embedding = method.fit_transform(mvdata.data.T)
            
            assert embedding is not None
            assert embedding.shape[0] == exp.n_frames
            
    def test_dimensionality_estimation_integration(self):
        """Test dimensionality estimation on INTENSE-processed data"""
        # Generate low-dimensional manifold data
        exp = generate_circular_manifold_exp(
            n_neurons=40,
            duration=400,
            noise_std=0.05,
            seed=42
        )
        
        # Get neural data
        neural_data = exp.calcium.T  # (n_timepoints, n_neurons)
        
        # Estimate dimensionality with different methods
        intrinsic_dim = nn_dimension(neural_data, k=5)
        linear_dim = pca_dimension(neural_data, threshold=0.95)
        eff_rank = effective_rank(neural_data)
        
        # For circular manifold embedded in high-D, expect:
        # - Intrinsic dimension around 2-4 (circular manifold + noise)
        # - Linear dimension higher due to noise
        # - Effective rank between the two
        assert 1 <= intrinsic_dim <= 10
        assert linear_dim >= intrinsic_dim
        assert eff_rank > 0
        
    def test_memory_efficiency(self):
        """Test memory efficiency of integrated pipeline"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate larger dataset
        exp = generate_2d_manifold_exp(
            n_neurons=64,
            duration=500,
            environments=['env1'],
            seed=42
        )
        
        # Run INTENSE with downsampling
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=20,
            n_shuffles_stage2=50,
            ds=5,  # Downsample by factor of 5
            verbose=False
        )
        
        # Apply DR
        mvdata = MVData(exp.calcium)
        embedding = mvdata.get_embedding({
            'e_method_name': 'pca',
            'dim': 10
        })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<500MB for this test)
        assert memory_increase < 500, f"Memory increased by {memory_increase}MB"
        
    def test_data_flow_validation(self):
        """Validate data flows correctly through pipeline"""
        # Create synthetic experiment
        n_neurons = 20
        n_frames = 200
        calcium_data = np.random.randn(n_neurons, n_frames)
        calcium_data[0, :] = np.sin(np.linspace(0, 4*np.pi, n_frames)) + 0.1 * np.random.randn(n_frames)
        
        exp = Experiment(
            signature='test_flow',
            calcium=calcium_data,
            spikes=None,
            exp_identificators={},
            static_features={'fps': 10.0},
            dynamic_features={
                'phase': np.linspace(0, 4*np.pi, n_frames)
            }
        )
        
        # Verify data shapes at each stage
        assert exp.calcium.shape == (n_neurons, n_frames)
        
        # INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=20,
            n_shuffles_stage2=50,
            verbose=False
        )
        
        # Check INTENSE outputs
        assert hasattr(exp, 'stats_table')
        assert exp.stats_table is not None
        
        # DR pipeline
        mvdata = MVData(exp.calcium)
        pca_embedding = mvdata.get_embedding({
            'e_method_name': 'pca',
            'dim': 3
        })
        
        # Validate final output
        assert pca_embedding.shape == (n_frames, 3)
        assert not np.any(np.isnan(pca_embedding))
        
    def test_error_handling_integration(self):
        """Test error handling across module boundaries"""
        # Create experiment with problematic data
        exp = Experiment(
            signature='test_errors',
            calcium=np.ones((10, 100)),  # Constant data - problematic for some methods
            spikes=None,
            exp_identificators={},
            static_features={'fps': 10.0},
            dynamic_features={'dummy': np.zeros(100)}
        )
        
        # INTENSE should handle constant features gracefully
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=10,
            n_shuffles_stage2=20,
            verbose=False
        )
        
        # MVData should handle constant data
        mvdata = MVData(exp.calcium)
        
        # PCA should work (though not meaningful)
        pca_embedding = mvdata.get_embedding({
            'e_method_name': 'pca', 
            'dim': 2
        })
        assert pca_embedding is not None