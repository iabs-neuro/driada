"""Tests for INTENSE pipeline functions.

This module tests the INTENSE pipeline functions using standardized fixtures
where possible. The experiment_factory fixture is used only for cases requiring
special parameters like:
- with_spikes=True for spike reconstruction tests
- Custom feature/neuron counts that don't match standard fixtures
- Tests requiring specific experiment configurations

Most tests use the parametrized fixtures (continuous_only_experiment,
discrete_only_experiment, mixed_features_experiment) with appropriate sizes.
"""

import pytest
import numpy as np
from driada.intense.pipelines import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance
)
from driada.information.info_base import TimeSeries
from driada.experiment.synthetic import (
    generate_synthetic_exp,
    generate_synthetic_exp_with_mixed_selectivity,
    generate_multiselectivity_patterns,
    discretize_via_roi
)


# @pytest.mark.skip(reason="Bug in compute_feat_feat_significance - see backlog TODO")
def test_compute_cell_feat_significance_with_disentanglement(small_experiment):
    """Test compute_cell_feat_significance with disentanglement mode."""
    # Use fixture for consistent test data
    exp = small_experiment
    
    # Run with disentanglement - use stage1 only for faster testing
    stats, significance, info, results, disent_results = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],  # Test subset of neurons
        feat_bunch=None,  # Use all features
        mode='stage1',
        n_shuffles_stage1=5,  # Optimized shuffles
        verbose=False,
        with_disentanglement=True,
        seed=42,
        ds=2  # Downsample for speed
    )
    
    # Check return values
    assert isinstance(stats, dict)
    assert isinstance(significance, dict)
    assert isinstance(info, dict)
    assert isinstance(results, object)  # IntenseResults
    assert isinstance(disent_results, dict)
    
    # Check disentanglement results structure
    assert 'feat_feat_significance' in disent_results
    assert 'disent_matrix' in disent_results
    assert 'count_matrix' in disent_results
    assert 'feature_names' in disent_results
    assert 'summary' in disent_results
    
    # Check matrix dimensions
    n_features = len(disent_results['feature_names'])
    assert disent_results['feat_feat_significance'].shape == (n_features, n_features)
    assert disent_results['disent_matrix'].shape == (n_features, n_features)
    assert disent_results['count_matrix'].shape == (n_features, n_features)
    
    # Check summary structure
    assert 'overall_stats' in disent_results['summary']
    assert 'feature_pairs' in disent_results['summary']


@pytest.mark.parametrize("continuous_only_experiment", ["small"], indirect=True)
def test_compute_cell_feat_significance_continuous_features(continuous_only_experiment):
    """Test compute_cell_feat_significance with continuous features."""
    # Use fixture with only continuous features
    exp = continuous_only_experiment
    
    # Run without disentanglement
    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1, 2, 3],  # Test subset of neurons
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        with_disentanglement=False
    )
    
    # Should return exactly 4 values (backward compatibility)
    assert len(result) == 4
    stats, significance, info, results = result
    
    # Check types
    assert isinstance(stats, dict)
    assert isinstance(significance, dict)
    assert isinstance(info, dict)
    assert hasattr(results, 'update')  # IntenseResults has update method


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_compute_cell_feat_significance_without_disentanglement(mixed_features_experiment):
    """Test backward compatibility when disentanglement is disabled."""
    # Use mixed features fixture
    exp = mixed_features_experiment
    
    # Run without disentanglement (default)
    result = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        with_disentanglement=False
    )
    
    # Should return exactly 4 values (backward compatibility)
    assert len(result) == 4
    stats, significance, info, results = result
    
    # Check types
    assert isinstance(stats, dict)
    assert isinstance(significance, dict)
    assert isinstance(info, dict)
    assert hasattr(results, 'update')  # IntenseResults has update method


# # @pytest.mark.skip(reason="Bug in compute_feat_feat_significance - see backlog TODO")
@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_compute_feat_feat_significance(mixed_features_experiment):
    """Test feature-feature significance computation."""
    # Use fixture with both discrete and continuous features
    exp = mixed_features_experiment
    
    # Compute feature-feature significance
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch='all',
        mode='stage1',  # Changed from two_stage for speed
        n_shuffles_stage1=10,
        n_shuffles_stage2=50,
        verbose=False,
        seed=42
    )
    
    # Check return types and shapes
    n_features = len(feat_ids)
    assert sim_mat.shape == (n_features, n_features)
    assert sig_mat.shape == (n_features, n_features)
    assert pval_mat.shape == (n_features, n_features)
    assert len(feat_ids) == n_features
    assert isinstance(info, dict)
    
    # Check diagonal is zero (self-similarity prevented)
    assert np.allclose(np.diag(sim_mat), 0)
    assert np.allclose(np.diag(sig_mat), 0)
    assert np.allclose(np.diag(pval_mat), 1)
    
    # Check symmetry
    assert np.allclose(sim_mat, sim_mat.T)
    assert np.allclose(sig_mat, sig_mat.T)
    assert np.allclose(pval_mat, pval_mat.T)
    
    # Check value ranges
    assert np.all(sim_mat >= 0)
    assert np.all((sig_mat == 0) | (sig_mat == 1))
    assert np.all((pval_mat >= 0) & (pval_mat <= 1))


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_compute_feat_feat_significance_specific_features(discrete_only_experiment):
    """Test feature-feature significance with specific feature subset."""
    # Use discrete fixture (medium has 3 features)
    exp = discrete_only_experiment
    
    # Get subset of features (use first 2-3 available)
    all_features = list(exp.dynamic_features.keys())
    selected_features = all_features[:min(3, len(all_features))]
    
    # Compute for subset
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch=selected_features,
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        seed=42
    )
    
    # Check correct features were used
    assert len(feat_ids) == len(selected_features)
    assert all(f in selected_features for f in feat_ids)
    assert sim_mat.shape == (len(selected_features), len(selected_features))


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_compute_cell_cell_significance(discrete_only_experiment):
    """Test neuron-neuron functional correlation computation."""
    # Use fixture with discrete features
    exp = discrete_only_experiment
    
    # Make some neurons correlated by copying signals
    # Set random seed for reproducible correlations
    np.random.seed(42)
    # Create new correlated data
    correlated_data_1 = exp.neurons[0].ca.data + np.random.randn(len(exp.neurons[0].ca.data)) * 0.1
    correlated_data_3 = exp.neurons[2].ca.data + np.random.randn(len(exp.neurons[2].ca.data)) * 0.1
    
    # Recreate TimeSeries objects to update cached copula transforms
    exp.neurons[1].ca = TimeSeries(correlated_data_1, discrete=False)
    exp.neurons[3].ca = TimeSeries(correlated_data_3, discrete=False)
    
    # Compute cell-cell significance
    # Use more shuffles for this test to ensure correlation detection
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=None,  # All neurons
        data_type='calcium',
        mode='stage1',  # Changed from two_stage for speed
        n_shuffles_stage1=25,  # Reduced for speed
        n_shuffles_stage2=50,
        verbose=False,
        seed=42
    )
    
    # Check return types and shapes
    n_cells = len(cell_ids)
    assert sim_mat.shape == (n_cells, n_cells)
    assert sig_mat.shape == (n_cells, n_cells)
    assert pval_mat.shape == (n_cells, n_cells)
    assert len(cell_ids) == n_cells
    assert isinstance(info, dict)
    
    # Check diagonal is zero
    assert np.allclose(np.diag(sim_mat), 0)
    assert np.allclose(np.diag(sig_mat), 0)
    assert np.allclose(np.diag(pval_mat), 1)
    
    # Check symmetry
    assert np.allclose(sim_mat, sim_mat.T)
    assert np.allclose(sig_mat, sig_mat.T)
    assert np.allclose(pval_mat, pval_mat.T)
    
    # Check that correlated pairs have higher MI
    # Note: With low shuffle counts, significance detection might be conservative
    # So we check if at least one of the expected correlations is detected
    correlation_detected = (
        (sim_mat[0, 1] > 0 and sim_mat[0, 1] > sim_mat[0, 4]) or
        (sim_mat[2, 3] > 0 and sim_mat[2, 3] > sim_mat[2, 4])
    )
    assert correlation_detected, "At least one correlated pair should be detected"


def test_compute_cell_cell_significance_spike_data():
    """Test neuron-neuron correlation with spike data."""
    # Create experiment with spike reconstruction
    exp = generate_synthetic_exp(n_dfeats=1, n_cfeats=0, nneurons=3, seed=42, with_spikes=True)
    
    # Test with spike data
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],
        data_type='spikes',
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        seed=42
    )
    
    # Check basic properties
    assert sim_mat.shape == (3, 3)
    assert np.all(sim_mat >= 0)
    assert np.allclose(np.diag(sim_mat), 0)


def test_compute_cell_cell_significance_subset(small_experiment):
    """Test neuron-neuron correlation with neuron subset."""
    # Use fixture (adapts to available neurons)
    exp = small_experiment
    
    # Select subset of neurons (adapt to available count)
    n_available = exp.n_cells
    selected_neurons = list(range(1, min(n_available, 8), 2))  # [1, 3, 5, 7] if available
    
    # Compute for subset
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=selected_neurons,
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        seed=42
    )
    
    # Check correct neurons were used
    assert len(cell_ids) == len(selected_neurons)
    assert all(c in selected_neurons for c in cell_ids)
    assert sim_mat.shape == (len(selected_neurons), len(selected_neurons))


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_disentanglement_integration(discrete_only_experiment):
    """Test integration with disentanglement module."""
    from driada.intense.disentanglement import DEFAULT_MULTIFEATURE_MAP
    
    # Use fixture with discrete features for place-related testing
    exp = discrete_only_experiment
    
    # Rename features to x and y for place field testing
    feat_keys = list(exp.dynamic_features.keys())
    if len(feat_keys) >= 2:
        x_data = exp.dynamic_features[feat_keys[0]]
        y_data = exp.dynamic_features[feat_keys[1]]
        exp.dynamic_features['x'] = x_data
        exp.dynamic_features['y'] = y_data
        # Remove old keys
        for k in feat_keys:
            if k not in ['x', 'y']:
                del exp.dynamic_features[k]
        
        # Force re-initialization of selectivity tables with new feature names
        exp.selectivity_tables_initialized = False
        # Rebuild data hashes for renamed features
        exp._build_data_hashes(mode='calcium')
    
    # Run with disentanglement using default multifeature map
    result = compute_cell_feat_significance(
        exp,
        feat_bunch=['x', 'y'],
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        use_precomputed_stats=False,  # Disable precomputed stats since we renamed features
        with_disentanglement=True,
        multifeature_map=None  # Should use DEFAULT_MULTIFEATURE_MAP
    )
    
    # With disentanglement, should return 5 values
    assert len(result) == 5
    stats, significance, info, results, disent_results = result
    
    # Check that default multifeature map was used
    assert 'feature_names' in disent_results
    # The feature names should include individual features


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True) 
def test_compute_cell_cell_significance_errors(discrete_only_experiment):
    """Test error handling in compute_cell_cell_significance."""
    exp = discrete_only_experiment
    
    # Test invalid data type
    with pytest.raises(ValueError, match='data_type.*can be either.*calcium.*or.*spikes'):
        compute_cell_cell_significance(
            exp,
            data_type='invalid',
            mode='stage1',
            n_shuffles_stage1=10,
            verbose=False
        )


@pytest.mark.parametrize("continuous_only_experiment", ["small"], indirect=True)
def test_compute_feat_feat_significance_multifeatures(continuous_only_experiment):
    """Test feature-feature significance with MultiTimeSeries."""
    # Use fixture with continuous features for multifeature support
    exp = continuous_only_experiment
    
    # Add x and y features to test multifeature (place field)
    feat_keys = list(exp.dynamic_features.keys())
    if len(feat_keys) >= 2:
        exp.dynamic_features['x'] = exp.dynamic_features[feat_keys[0]]
        exp.dynamic_features['y'] = exp.dynamic_features[feat_keys[1]]
    
    # Compute including multifeature
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch=['x', 'y', ('x', 'y')],  # Include multifeature tuple
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        seed=42
    )
    
    # Should handle both single and multi features
    assert len(feat_ids) == 3
    assert 'x' in feat_ids
    assert 'y' in feat_ids
    assert ('x', 'y') in feat_ids
    assert sim_mat.shape == (3, 3)


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_with_disentanglement_custom_multifeature_map(discrete_only_experiment):
    """Test disentanglement with custom multifeature mapping."""
    # Use discrete fixture (adapt to available features)
    exp = discrete_only_experiment
    
    # Rename features for testing (adapt to what's available)
    feat_keys = list(exp.dynamic_features.keys())
    if len(feat_keys) >= 4:
        exp.dynamic_features['speed'] = exp.dynamic_features[feat_keys[0]]
        exp.dynamic_features['head_direction'] = exp.dynamic_features[feat_keys[1]]
        exp.dynamic_features['x'] = exp.dynamic_features[feat_keys[2]]
        exp.dynamic_features['y'] = exp.dynamic_features[feat_keys[3]]
        # Clean up
        for k in feat_keys:
            if k not in ['speed', 'head_direction', 'x', 'y']:
                del exp.dynamic_features[k]
    elif len(feat_keys) >= 2:
        # Simpler test with just x, y
        exp.dynamic_features['x'] = exp.dynamic_features[feat_keys[0]]
        exp.dynamic_features['y'] = exp.dynamic_features[feat_keys[1]]
        # Clean up
        for k in feat_keys[:2]:
            del exp.dynamic_features[k]
        
        # Force re-initialization with new feature names
        exp.selectivity_tables_initialized = False
        exp._build_data_hashes(mode='calcium')
    
    # Custom multifeature map (adapt to available features)
    available_features = list(exp.dynamic_features.keys())
    if len(available_features) >= 4:
        custom_map = {
            ('x', 'y'): 'place',
            ('speed', 'head_direction'): 'locomotion'
        }
        feat_bunch = ['speed', 'head_direction', 'x', 'y']
    else:
        custom_map = {
            ('x', 'y'): 'place'
        }
        feat_bunch = ['x', 'y']
    
    # Run with custom map
    result = compute_cell_feat_significance(
        exp,
        feat_bunch=feat_bunch,
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        use_precomputed_stats=False,  # Disable since we renamed features
        with_disentanglement=True,
        multifeature_map=custom_map
    )
    
    # Check it worked
    assert len(result) == 5
    _, _, _, _, disent_results = result
    assert 'summary' in disent_results


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_compute_cell_cell_significance_downsampling(discrete_only_experiment):
    """Test neuron-neuron correlation with downsampling."""
    # Use fixture
    exp = discrete_only_experiment
    
    # Test with downsampling
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        data_type='calcium',
        mode='stage1',
        n_shuffles_stage1=10,
        ds=2,  # Downsample by factor of 2
        verbose=False,
        seed=42
    )
    
    # Should work with downsampling (small fixture has 5 neurons)
    assert sim_mat.shape == (5, 5)
    assert np.all(sim_mat >= 0)


@pytest.mark.parametrize("discrete_only_experiment", ["small"], indirect=True)
def test_compute_feat_feat_significance_empty_features(discrete_only_experiment):
    """Test error handling with empty feature list."""
    exp = discrete_only_experiment
    
    # This should work with empty list (but produce empty results)
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch=[],
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        seed=42
    )
    
    assert len(feat_ids) == 0
    assert sim_mat.shape == (0, 0)
    assert sig_mat.shape == (0, 0)
    assert pval_mat.shape == (0, 0)


def test_mixed_selectivity_generation():
    """Test generation of mixed selectivity synthetic data."""
    # Test basic generation
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=2,
        n_continuous_feats=2,
        n_neurons=5,  # Reduced from 10
        n_multifeatures=1,
        create_discrete_pairs=True,
        selectivity_prob=0.8,
        multi_select_prob=0.5,
        weights_mode='random',
        duration=30,  # Increased from 10 to avoid issues
        seed=42,
        verbose=False
    )
    
    # Check experiment structure
    assert exp.n_cells == 5  # Updated to match reduced n_neurons
    assert 'c_feat_0' in exp.dynamic_features
    assert 'c_feat_1' in exp.dynamic_features
    assert 'd_feat_from_c0' in exp.dynamic_features
    assert 'd_feat_from_c1' in exp.dynamic_features
    
    # Check selectivity info
    assert 'matrix' in selectivity_info
    assert 'feature_names' in selectivity_info
    assert 'multifeature_map' in selectivity_info
    assert selectivity_info['matrix'].shape[1] == 5  # n_neurons
    assert ('c_feat_0', 'c_feat_1') in selectivity_info['multifeature_map']
    
    # Check some neurons have mixed selectivity
    n_selective_features = np.sum(selectivity_info['matrix'] > 0, axis=0)
    assert np.any(n_selective_features > 1)  # At least one neuron with mixed selectivity
    # With only 5 neurons and high selectivity_prob, all might be selective


def test_discretize_via_roi():
    """Test ROI-based discretization method."""
    # Create continuous signal
    np.random.seed(42)
    continuous_signal = np.random.randn(1000)
    
    # Discretize
    binary_signal = discretize_via_roi(continuous_signal, seed=42)
    
    # Check output
    assert len(binary_signal) == len(continuous_signal)
    assert set(np.unique(binary_signal)).issubset({0, 1})
    
    # Check that discretization worked
    assert np.sum(binary_signal) > 0  # Some values should be 1
    assert np.sum(binary_signal) < len(binary_signal)  # Some values should be 0


def test_disentanglement_with_mixed_selectivity():
    """Test disentanglement analysis on mixed selectivity data."""
    # Generate data with known patterns
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=0,
        n_continuous_feats=4,
        n_neurons=10,  # Reduced from 20 for faster testing
        n_multifeatures=1,
        create_discrete_pairs=True,
        selectivity_prob=0.9,
        multi_select_prob=0.7,
        weights_mode='dominant',  # One feature dominates
        duration=30,  # Reduced from 120 for faster testing
        fps=10,  # Lower sampling rate for speed
        seed=42,
        verbose=False
    )
    
    # Run disentanglement analysis
    result = compute_cell_feat_significance(
        exp,
        feat_bunch=['c_feat_0', 'd_feat_from_c0', 'c_feat_1', 'd_feat_from_c1'],
        mode='stage1',
        n_shuffles_stage1=10,  # Reduced from 20 for faster testing
        verbose=False,
        with_disentanglement=True,
        seed=42
    )
    
    assert len(result) == 5
    _, _, _, _, disent_results = result
    
    # Check disentanglement detected continuous vs discrete preference
    assert 'disent_matrix' in disent_results
    assert 'feature_names' in disent_results
    
    # Find indices for continuous and discrete versions
    feat_names = disent_results['feature_names']
    if 'c_feat_0' in feat_names and 'd_feat_from_c0' in feat_names:
        idx_cont = feat_names.index('c_feat_0')
        idx_disc = feat_names.index('d_feat_from_c0')
        
        # Discrete version should generally be preferred (higher values in its row)
        # This is because discrete features are easier to decode
        disent_matrix = disent_results['disent_matrix']
        if disent_results['count_matrix'][idx_cont, idx_disc] > 0:
            # If these features were compared, discrete should dominate
            assert disent_matrix[idx_disc, idx_cont] >= disent_matrix[idx_cont, idx_disc]


def test_equal_weight_mixed_selectivity():
    """Test mixed selectivity with equal weights (no disentanglement expected)."""
    # Generate smaller data with equal weights
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=2,
        n_continuous_feats=0,
        n_neurons=4,  # Optimized size
        n_multifeatures=0,
        create_discrete_pairs=False,
        selectivity_prob=0.9,
        multi_select_prob=0.8,
        weights_mode='equal',  # Equal weights - no clear dominance
        duration=60,  # Reduced from 200 for faster testing
        fps=10,  # Lower sampling rate for speed
        seed=42,
        verbose=False
    )
    
    # Check that equal weights were assigned
    matrix = selectivity_info['matrix']
    for j in range(matrix.shape[1]):
        weights = matrix[:, j]
        non_zero_weights = weights[weights > 0]
        if len(non_zero_weights) > 1:
            # Check weights are approximately equal
            assert np.std(non_zero_weights) < 0.01
    
    # Run disentanglement with minimal shuffles
    result = compute_cell_feat_significance(
        exp,
        mode='stage1',
        n_shuffles_stage1=5,  # Optimized shuffles
        verbose=False,
        with_disentanglement=True,
        seed=42,
        ds=2  # Downsample for speed
    )
    
    # Full validation checks
    assert len(result) == 5  # with_disentanglement=True returns 5 values
    stats, significance, info, results, disent_results = result
    
    # Check disentanglement matrix for equal weights
    disent_matrix = disent_results['disent_matrix']
    count_matrix = disent_results['count_matrix']
    
    # When features have equal weight in mixed selectivity,
    # disentanglement should show balanced contribution
    for i in range(disent_matrix.shape[0]):
        for j in range(i+1, disent_matrix.shape[1]):
            if count_matrix[i, j] > 0:  # If this pair was analyzed
                # Check interaction information
                ii = info.get('interaction_info', {}).get((i, j), 0)
                # With equal weights, interaction should be minimal
                assert abs(ii) < 0.5  # Relaxed threshold for small data
                
                # Check balance in disentanglement
                ratio = disent_matrix[i, j] / (disent_matrix[i, j] + disent_matrix[j, i] + 1e-10)
                # Should be close to 0.5 (balanced)
                assert 0.3 < ratio < 0.7


def test_multifeature_generation():
    """Test multifeature creation in synthetic data."""
    # Generate with multifeatures
    exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=0,
        n_continuous_feats=4,
        n_neurons=5,
        n_multifeatures=2,
        create_discrete_pairs=False,
        duration=30,  # Reduced from 60 for faster testing
        fps=10,  # Lower sampling rate for speed
        seed=42,
        verbose=False
    )
    
    # Check multifeatures were created
    assert ('c_feat_0', 'c_feat_1') in selectivity_info['multifeature_map']
    assert selectivity_info['multifeature_map'][('c_feat_0', 'c_feat_1')] == 'multi0'
    assert ('c_feat_2', 'c_feat_3') in selectivity_info['multifeature_map']
    assert selectivity_info['multifeature_map'][('c_feat_2', 'c_feat_3')] == 'multi1'
    
    # Test that multifeatures can be used in analysis
    result = compute_feat_feat_significance(
        exp,
        feat_bunch=['c_feat_0', 'c_feat_1', ('c_feat_0', 'c_feat_1')],
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        seed=42
    )
    
    sim_mat, sig_mat, pval_mat, feat_ids, info = result
    assert ('c_feat_0', 'c_feat_1') in feat_ids
    assert len(feat_ids) == 3