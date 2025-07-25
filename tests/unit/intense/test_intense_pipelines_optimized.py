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
    compute_cell_cell_significance
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
    'verbose': False,
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