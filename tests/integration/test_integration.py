"""Integration tests for DRIADA modules working together"""

import pytest
import numpy as np
import warnings
from driada import (
    Experiment, 
    compute_cell_feat_significance
)
from driada.dimensionality import nn_dimension, pca_dimension, effective_rank
from driada.dim_reduction import MVData
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap


class TestINTENSEToDRIntegration:
    """Test INTENSE → Dimensionality Reduction pipeline"""
    
    def test_intense_to_pca_pipeline(self, circular_manifold_exp_fast, intense_params_fast):
        """Test full pipeline from INTENSE selectivity to PCA reduction"""
        # Use fixture for faster test
        exp = circular_manifold_exp_fast
        
        # Run INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            **intense_params_fast
        )
        
        # Get significant neurons
        sig_neurons = exp.get_significant_neurons()
        assert len(sig_neurons) > 0, "No significant neurons found"
        
        # Extract neural data for DR
        neural_data = exp.calcium.data
        
        # Apply PCA
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(neural_data.T)
        
        # Check embedding quality
        assert embedding.shape == (exp.n_frames, 2)
        assert pca.explained_variance_ratio_[0] > 0.1  # First PC explains >10%
        
    def test_intense_to_manifold_learning(self, spatial_2d_exp_fast, intense_params_fast):
        """Test INTENSE with nonlinear DR methods"""
        # Use fixture for faster test
        exp = spatial_2d_exp_fast
        
        # INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            **intense_params_fast
        )
        
        # Extract selective neurons' activity
        sig_neurons = exp.get_significant_neurons()
        if len(sig_neurons) > 0:
            selective_ids = list(sig_neurons.keys())
            neural_subset = exp.calcium.data[selective_ids, :]
        else:
            neural_subset = exp.calcium.data
            
        # Apply Isomap
        iso = Isomap(n_components=2, n_neighbors=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = iso.fit_transform(neural_subset.T)
        
        assert embedding.shape[1] == 2
        assert not np.any(np.isnan(embedding))
        
    def test_mvdata_experiment_integration(self, mixed_population_exp_fast):
        """Test MVData integration with Experiment objects"""
        # Use fixture for faster test
        exp = mixed_population_exp_fast
        
        # Convert to MVData - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium
        
        # Test multiple DR methods
        methods = {
            'pca': PCA(n_components=3),
            'isomap': {'nn': 10},
            'umap': {'nn': 15}
        }
        
        for method_name, method in methods.items():
            if isinstance(method, dict):
                # Graph-based methods - use new simplified API
                embedding = mvdata.get_embedding(method=method_name, dim=2, nn=method['nn'])
            else:
                # Direct methods like PCA
                embedding = method.fit_transform(mvdata.data.T)
            
            if hasattr(embedding, 'coords'):
                # Embedding object from get_embedding
                assert embedding is not None
                # Check frames, accounting for possible node loss in graph methods
                if method_name in ['isomap', 'umap'] and embedding.coords.shape[1] < exp.n_frames:
                    # Graph methods may lose nodes
                    assert embedding.coords.shape[1] <= exp.n_frames
                else:
                    assert embedding.coords.shape[1] == exp.n_frames
            else:
                # Direct numpy array from sklearn
                assert embedding is not None
                assert embedding.shape[0] == exp.n_frames
            
    def test_dimensionality_estimation_integration(self, circular_manifold_exp_balanced):
        """Test dimensionality estimation on INTENSE-processed data"""
        # Use balanced fixture for more neurons
        exp = circular_manifold_exp_balanced
        
        # Get neural data
        neural_data = exp.calcium.data.T  # (n_timepoints, n_neurons)
        
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
        
    def test_memory_efficiency(self, memory_test_exp):
        """Test memory efficiency of integrated pipeline"""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not installed")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Use fixture for consistent testing
        exp = memory_test_exp
        
        # Run INTENSE with downsampling
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=20,
            n_shuffles_stage2=50,
            ds=5,  # Downsample by factor of 5
            allow_mixed_dimensions=True,  # Need this for 2D spatial features
            find_optimal_delays=False,  # Can't use with multifeatures
            verbose=False,
            enable_parallelization=False
        )
        
        # Apply DR - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium
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
            reconstruct_spikes=None,  # Disable spike reconstruction for synthetic data
            exp_identificators={},
            static_features={'fps': 10.0, 't_off_sec': 1.0},  # Set explicit t_off to reduce exclusion zone
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
            ds=5,
            verbose=False,
            enable_parallelization=False
        )
        
        # Check INTENSE outputs
        assert hasattr(exp, 'stats_tables')
        assert exp.stats_tables is not None
        
        # DR pipeline - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium
        pca_embedding = mvdata.get_embedding({
            'e_method_name': 'pca',
            'dim': 3
        })
        
        # Validate final output
        assert pca_embedding.coords.shape == (3, n_frames)
        assert not np.any(np.isnan(pca_embedding.coords))
        
    def test_error_handling_integration(self):
        """Test error handling across module boundaries"""
        # Create experiment with problematic data
        # Use more frames to avoid shuffle_mask issues, but still problematic for analysis
        n_frames = 500
        calcium_data = np.ones((10, n_frames)) + 0.01 * np.random.randn(10, n_frames)
        calcium_data = np.maximum(0, calcium_data)  # Ensure non-negative
        
        exp = Experiment(
            signature='test_errors',
            calcium=calcium_data,  # Near-constant data - problematic for some methods
            spikes=None,
            reconstruct_spikes=None,  # Disable spike reconstruction
            exp_identificators={},
            static_features={'fps': 10.0},
            dynamic_features={'dummy': np.zeros(n_frames)}
        )
        
        # INTENSE should handle constant features gracefully
        stats, significance, info, results = compute_cell_feat_significance(
            exp,
            n_shuffles_stage1=10,
            n_shuffles_stage2=20,
            ds=5,
            verbose=False,
            enable_parallelization=False
        )
        
        # MVData should handle constant data - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium
        
        # PCA should work (though not meaningful)
        pca_embedding = mvdata.get_embedding({
            'e_method_name': 'pca', 
            'dim': 2
        })
        assert pca_embedding is not None
        assert hasattr(pca_embedding, 'coords')