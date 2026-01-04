"""Integration tests for DRIADA modules working together"""

import pytest
import numpy as np
import warnings
from driada import compute_cell_feat_significance
from driada.dimensionality import nn_dimension, pca_dimension, effective_rank
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


class TestINTENSEToDRIntegration:
    """Test INTENSE → Dimensionality Reduction pipeline"""

    def test_intense_to_pca_pipeline(
        self, circular_manifold_exp_fast, intense_params_fast
    ):
        """Test full pipeline from INTENSE selectivity to PCA reduction"""
        # Use fixture for faster test
        exp = circular_manifold_exp_fast

        # Run INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp, **intense_params_fast
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

    def test_intense_to_manifold_learning(
        self, spatial_2d_exp_fast, intense_params_fast
    ):
        """Test INTENSE with nonlinear DR methods"""
        # Use fixture for faster test
        exp = spatial_2d_exp_fast

        # INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp, **intense_params_fast
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

        # Test PCA integration (other DR methods tested separately)
        pca = PCA(n_components=3)
        embedding = pca.fit_transform(mvdata.data.T)

        # Verify embedding
        assert embedding is not None
        assert embedding.shape == (exp.n_frames, 3)
        assert not np.any(np.isnan(embedding))

    def test_dimensionality_estimation_integration(
        self, circular_manifold_exp_balanced
    ):
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

    def test_memory_efficiency(self, memory_test_exp, intense_params_fast):
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
            exp, **intense_params_fast
        )

        # Apply DR - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium
        embedding = mvdata.get_embedding({"e_method_name": "pca", "dim": 10})

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (<500MB for this test)
        assert memory_increase < 500, f"Memory increased by {memory_increase}MB"

    def test_data_flow_validation(
        self, circular_manifold_exp_fast, intense_params_fast
    ):
        """Validate data flows correctly through pipeline"""
        # Use standard fixture
        exp = circular_manifold_exp_fast

        # Verify data shapes at each stage
        assert exp.calcium.shape == (exp.n_cells, exp.n_frames)

        # INTENSE analysis
        stats, significance, info, results = compute_cell_feat_significance(
            exp, **intense_params_fast
        )

        # Check INTENSE outputs
        assert hasattr(exp, "stats_tables")
        assert exp.stats_tables is not None

        # DR pipeline - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium
        pca_embedding = mvdata.get_embedding({"e_method_name": "pca", "dim": 3})

        # Validate final output
        assert pca_embedding.coords.shape == (3, exp.n_frames)
        assert not np.any(np.isnan(pca_embedding.coords))

    def test_error_handling_integration(
        self, circular_manifold_exp_fast, intense_params_fast
    ):
        """Test error handling across module boundaries"""
        # Use standard fixture but modify data to be problematic
        exp = circular_manifold_exp_fast

        # Make calcium data near-constant (problematic for some methods)
        exp.calcium.data = np.ones_like(exp.calcium.data) + 0.01 * np.random.randn(
            *exp.calcium.data.shape
        )
        exp.calcium.data = np.maximum(0, exp.calcium.data)

        # INTENSE should handle constant features gracefully
        stats, significance, info, results = compute_cell_feat_significance(
            exp, **intense_params_fast
        )

        # MVData should handle constant data - exp.calcium is already a MultiTimeSeries which inherits from MVData
        mvdata = exp.calcium

        # PCA should work (though not meaningful)
        pca_embedding = mvdata.get_embedding({"e_method_name": "pca", "dim": 2})
        assert pca_embedding is not None
        assert hasattr(pca_embedding, "coords")
