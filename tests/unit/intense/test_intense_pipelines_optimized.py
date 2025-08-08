"""Optimized INTENSE pipeline tests with reduced parameters for fast execution.

This file contains the same tests as test_intense_pipelines.py but with
optimized parameters for fast test execution:
- Using small experiments exclusively
- Reduced shuffle counts (5-10 instead of 10-25)
- Shorter durations (20-30s instead of 30-200s)
- Downsampling enabled (ds=5)
- Stage1 mode only (no two_stage)
"""

import pytest
import numpy as np
from driada.intense.pipelines import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance,
    compute_embedding_selectivity
)
from driada.information.info_base import TimeSeries
from driada.experiment.synthetic import (
    generate_synthetic_exp,
    generate_synthetic_exp_with_mixed_selectivity,
    generate_multiselectivity_patterns,
    discretize_via_roi
)


# Fast test parameters
FAST_PARAMS = {
    'mode': 'stage1',
    'n_shuffles_stage1': 5,
    'ds': 5,  # Aggressive downsampling
    'enable_parallelization': False,  # Disable for small tests
    'seed': 42
}


def test_compute_cell_feat_significance_with_disentanglement_fast(small_experiment):
    """Fast test for cell-feat significance with disentanglement."""
    exp = small_experiment
    
    # Run with minimal parameters
    stats, significance, info, results, disent_results = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],  # Just 2 neurons
        feat_bunch=None,
        with_disentanglement=True,
        **FAST_PARAMS
    )
    
    # Basic checks only
    assert isinstance(disent_results, dict)
    assert 'feat_feat_significance' in disent_results
    assert 'disent_matrix' in disent_results


@pytest.mark.parametrize("continuous_only_experiment", ["small"], indirect=True)
def test_compute_cell_feat_significance_continuous_fast(continuous_only_experiment):
    """Fast test for continuous features."""
    exp = continuous_only_experiment
    
    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],  # Minimal neurons
        with_disentanglement=False,
        **FAST_PARAMS
    )
    
    assert len(result) == 4


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_compute_feat_feat_significance_fast(mixed_features_experiment):
    """Fast test for feature-feature correlation."""
    exp = mixed_features_experiment
    
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        **FAST_PARAMS
    )
    
    n_features = len(feat_ids)
    assert sim_mat.shape == (n_features, n_features)
    assert np.allclose(np.diag(sim_mat), 0)


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_compute_cell_cell_significance_fast(discrete_only_experiment):
    """Fast test for neuron-neuron correlation."""
    exp = discrete_only_experiment
    
    # Make neurons correlated
    np.random.seed(42)
    exp.neurons[1].ca = TimeSeries(
        exp.neurons[0].ca.data + np.random.randn(len(exp.neurons[0].ca.data)) * 0.1,
        discrete=False
    )
    
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],  # Just 3 neurons
        data_type='calcium',
        **FAST_PARAMS
    )
    
    assert sim_mat.shape == (3, 3)
    assert np.allclose(np.diag(sim_mat), 0)


def test_mixed_selectivity_generation_fast():
    """Fast test for mixed selectivity pattern generation."""
    # Minimal generation - check the actual API
    from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
    
    # Generate minimal mixed selectivity experiment
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=1,
        n_continuous_feats=2,
        n_neurons=5,
        duration=60,  # Increased to avoid shuffle mask error
        fps=10,
        selectivity_prob=0.8,
        seed=42,
        verbose=False
    )
    
    assert exp.n_cells == 5
    assert 'matrix' in selectivity_info
    assert selectivity_info['matrix'].shape == (5, 5)  # neurons x features


def test_disentanglement_minimal():
    """Minimal disentanglement test with tiny data."""
    # Generate minimal data with longer duration to avoid shuffle mask issues
    exp = generate_synthetic_exp(
        n_dfeats=2,
        n_cfeats=2,
        nneurons=3,
        duration=60,  # Increased to avoid shuffle mask error
        fps=10,  # Low fps
        seed=42,
        with_spikes=False
    )
    
    # Run with minimal parameters
    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],
        with_disentanglement=True,
        **FAST_PARAMS
    )
    
    assert len(result) == 5  # Including disentanglement results


class TestEdgeCasesFast:
    """Fast edge case tests."""
    
    def test_empty_cell_bunch(self, small_experiment):
        """Test with empty neuron list."""
        # Empty cell bunch should raise ValueError
        with pytest.raises(ValueError, match="ts_bunch1 cannot be empty"):
            compute_cell_feat_significance(
                small_experiment,
                cell_bunch=[],
                **FAST_PARAMS
            )
    
    def test_single_neuron(self, small_experiment):
        """Test with single neuron."""
        result = compute_cell_feat_significance(
            small_experiment,
            cell_bunch=[0],
            **FAST_PARAMS
        )
        assert len(result) == 4
    
    def test_single_feature(self, small_experiment):
        """Test with single feature."""
        feat_names = list(small_experiment.dynamic_features.keys())
        sim_mat, _, _, feat_ids, _ = compute_feat_feat_significance(
            small_experiment,
            feat_bunch=[feat_names[0]],
            **FAST_PARAMS
        )
        assert sim_mat.shape == (1, 1)
        assert sim_mat[0, 0] == 0


class TestPerformanceBenchmarks:
    """Benchmark tests to ensure optimization works."""
    
    def test_all_functions_under_5s(self, small_experiment):
        """Ensure all main functions complete in under 5 seconds."""
        import time
        
        # Test cell-feat
        start = time.time()
        compute_cell_feat_significance(
            small_experiment,
            cell_bunch=[0, 1, 2],
            **FAST_PARAMS
        )
        assert time.time() - start < 5.0
        
        # Test feat-feat
        start = time.time()
        compute_feat_feat_significance(
            small_experiment,
            **FAST_PARAMS
        )
        assert time.time() - start < 5.0
        
        # Test cell-cell
        start = time.time()
        compute_cell_cell_significance(
            small_experiment,
            cell_bunch=[0, 1, 2],
            **FAST_PARAMS
        )
        assert time.time() - start < 5.0


# Additional comprehensive tests for better coverage

def test_compute_cell_cell_significance_comprehensive(small_experiment):
    """Comprehensive test for cell-cell significance with all code paths."""
    exp = small_experiment
    
    # Ensure each neuron has unique calcium data to avoid self-comparison issues
    np.random.seed(42)
    for i in range(exp.n_cells):
        # Add unique noise pattern to each neuron
        noise = np.random.randn(len(exp.neurons[i].ca.data)) * 0.1 * (i + 1)
        exp.neurons[i].ca = TimeSeries(
            exp.neurons[i].ca.data + noise,
            discrete=False
        )
    
    # Test with calcium data
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=None,  # Test with all neurons
        data_type='calcium',
        verbose=True,  # Test verbose output
        **FAST_PARAMS
    )
    
    n_cells = exp.n_cells
    assert sim_mat.shape == (n_cells, n_cells)
    assert sig_mat.shape == (n_cells, n_cells)
    assert pval_mat.shape == (n_cells, n_cells)
    assert len(cell_ids) == n_cells
    
    # Check diagonal is zero
    assert np.allclose(np.diag(sim_mat), 0)
    assert np.allclose(np.diag(sig_mat), 0)
    assert np.allclose(np.diag(pval_mat), 1)
    
    # Check symmetry
    assert np.allclose(sim_mat, sim_mat.T)
    assert np.allclose(sig_mat, sig_mat.T)
    assert np.allclose(pval_mat, pval_mat.T)


def test_compute_cell_cell_significance_with_spikes(small_experiment):
    """Test cell-cell significance with spike data."""
    exp = small_experiment
    
    # Add spike data to neurons
    for i in range(exp.n_cells):
        neuron = exp.neurons[i]
        # Create synthetic spike data
        spike_data = (neuron.ca.data > np.percentile(neuron.ca.data, 80)).astype(int)
        neuron.sp = TimeSeries(spike_data, discrete=True)
    
    # Test with spike data
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],
        data_type='spikes',
        **FAST_PARAMS
    )
    
    assert sim_mat.shape == (3, 3)
    assert np.allclose(np.diag(sim_mat), 0)


def test_compute_cell_cell_significance_error_cases(small_experiment):
    """Test error handling in cell-cell significance."""
    exp = small_experiment
    
    # Test invalid data type
    with pytest.raises(ValueError, match='"data_type" can be either'):
        compute_cell_cell_significance(
            exp,
            data_type='invalid',
            **FAST_PARAMS
        )
    
    # Test with no spike data
    for i in range(exp.n_cells):
        neuron = exp.neurons[i]
        neuron.sp = None
    
    with pytest.raises(ValueError, match="Some neurons have no spike data"):
        compute_cell_cell_significance(
            exp,
            data_type='spikes',
            **FAST_PARAMS
        )


def test_compute_embedding_selectivity_basic(small_experiment):
    """Test compute_embedding_selectivity with basic embeddings."""
    exp = small_experiment
    
    # Create and store a simple PCA embedding
    from sklearn.decomposition import PCA
    
    # Get neural data
    neural_data = np.array([exp.neurons[i].ca.data for i in range(exp.n_cells)]).T
    
    # Compute PCA
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(neural_data)
    
    # Store embedding
    exp.store_embedding(
        embedding,
        method_name='pca',
        data_type='calcium',
        metadata={'method': 'PCA', 'n_components': 2}
    )
    
    # Test embedding selectivity
    results = compute_embedding_selectivity(
        exp,
        embedding_methods=['pca'],
        cell_bunch=[0, 1, 2],
        verbose=True,  # Test verbose output
        **FAST_PARAMS
    )
    
    assert 'pca' in results
    pca_results = results['pca']
    assert 'stats' in pca_results
    assert 'significance' in pca_results
    assert 'significant_neurons' in pca_results
    assert 'component_selectivity' in pca_results
    assert pca_results['n_components'] == 2


def test_compute_embedding_selectivity_multiple_methods(small_experiment):
    """Test embedding selectivity with multiple methods."""
    exp = small_experiment
    
    # Get neural data
    neural_data = np.array([exp.neurons[i].ca.data for i in range(exp.n_cells)]).T
    
    # Store multiple embeddings
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # PCA
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(neural_data)
    exp.store_embedding(pca_embedding, method_name='pca', data_type='calcium')
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)  # Small perplexity for small data
    tsne_embedding = tsne.fit_transform(neural_data)
    exp.store_embedding(tsne_embedding, method_name='tsne', data_type='calcium')
    
    # Test with all embeddings
    results = compute_embedding_selectivity(
        exp,
        embedding_methods=None,  # Test with None (all methods)
        **FAST_PARAMS
    )
    
    assert 'pca' in results
    assert 'tsne' in results
    
    # Test specific method
    results_single = compute_embedding_selectivity(
        exp,
        embedding_methods='pca',  # Test with string input
        **FAST_PARAMS
    )
    
    assert len(results_single) == 1
    assert 'pca' in results_single


def test_compute_embedding_selectivity_error_cases(small_experiment):
    """Test error handling in embedding selectivity."""
    exp = small_experiment
    
    # Test with no embeddings - should raise KeyError per actual implementation
    with pytest.raises(KeyError, match="No embedding found"):
        compute_embedding_selectivity(
            exp,
            embedding_methods=['nonexistent'],
            **FAST_PARAMS
        )


def test_compute_feat_feat_significance_edge_cases(small_experiment):
    """Test edge cases in feat-feat significance."""
    exp = small_experiment
    
    # Test with empty feature list
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch=[],  # Empty list
        verbose=True,  # Test verbose path
        **FAST_PARAMS
    )
    
    assert sim_mat.shape == (0, 0)
    assert sig_mat.shape == (0, 0)
    assert pval_mat.shape == (0, 0)
    assert len(feat_ids) == 0
    
    # Test with multifeatures
    multifeature = ('d_feat_0', 'd_feat_1')  # Assuming these exist
    sim_mat2, sig_mat2, pval_mat2, feat_ids2, info2 = compute_feat_feat_significance(
        exp,
        feat_bunch=['d_feat_0', multifeature],
        verbose=True,
        **FAST_PARAMS
    )
    
    assert len(feat_ids2) == 2
    assert 'd_feat_0' in feat_ids2
    assert multifeature in feat_ids2


def test_compute_cell_feat_significance_error_paths(small_experiment):
    """Test error handling paths in cell-feat significance."""
    exp = small_experiment
    
    # Test with invalid data type
    with pytest.raises(ValueError, match='"data_type" can be either'):
        compute_cell_feat_significance(
            exp,
            data_type='invalid',
            **FAST_PARAMS
        )
    
    # Test with non-existent feature
    with pytest.raises(ValueError, match="Feature .* not found in experiment"):
        compute_cell_feat_significance(
            exp,
            feat_bunch=['nonexistent_feature'],
            allow_mixed_dimensions=True,
            **FAST_PARAMS
        )
    
    # Test mixed dimensions with multifeatures
    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],
        feat_bunch=['d_feat_0', ('d_feat_0', 'd_feat_1')],
        allow_mixed_dimensions=True,
        verbose=True,  # Test verbose paths
        **FAST_PARAMS
    )
    
    assert len(result) == 4


def test_cell_cell_with_identical_spike_data(small_experiment):
    """Test warning when all neurons have identical spike data."""
    exp = small_experiment
    
    # Create identical spike data for all neurons
    identical_spikes = TimeSeries(np.zeros(exp.n_frames, dtype=int), discrete=True)
    for i in range(exp.n_cells):
        neuron = exp.neurons[i]
        neuron.sp = identical_spikes
    
    # Should generate a warning
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        compute_cell_cell_significance(
            exp,
            data_type='spikes',
            **FAST_PARAMS
        )
        
        assert len(w) == 1
        assert "identical spike data" in str(w[0].message)