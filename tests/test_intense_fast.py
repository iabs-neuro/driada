"""
Fast versions of INTENSE tests for quick validation.

These tests use reduced parameters while maintaining statistical validity.
Run with: pytest tests/test_intense_fast.py
"""

import pytest
import numpy as np
from src.driada.intense.intense_base import compute_me_stats
from src.driada.information.info_base import TimeSeries
from src.driada.utils.data import retrieve_relevant_from_nested_dict
from src.driada.experiment.synthetic import generate_synthetic_exp


def create_correlated_ts_fast(n=20, T=2000):
    """Fast version with reduced size."""
    np.random.seed(42)
    C = np.eye(n)
    # Add key correlations
    C[1, n-1] = 0.9
    C[2, n-2] = 0.8
    C = (C + C.T)
    np.fill_diagonal(C, 1)
    
    signals = np.random.multivariate_normal(np.zeros(n), C, size=T).T
    signals += np.random.randn(n, T) * 0.2
    
    tslist1 = [TimeSeries(sig, discrete=False) for sig in signals[:n//2]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in signals[n//2:]]
    
    for ts in tslist1 + tslist2:
        ts.shuffle_mask[:50] = 0
        
    return tslist1, tslist2


def test_stage1_fast():
    """Fast version of stage1 test."""
    n = 20  # Reduced from 40
    tslist1, tslist2 = create_correlated_ts_fast(n, T=2000)  # Reduced from 10000
    
    computed_stats, computed_significance, info = compute_me_stats(
        tslist1,
        tslist2,
        mode='stage1',
        n_shuffles_stage1=20,  # Reduced from 100
        joint_distr=False,
        metric_distr_type='gamma',
        noise_ampl=1e-3,
        ds=2,  # Downsample by 2
        topk1=1,
        verbose=False  # No verbose output
    )
    
    rel_stats_pairs = retrieve_relevant_from_nested_dict(computed_stats, 'pre_rval', 1)
    rel_sig_pairs = retrieve_relevant_from_nested_dict(computed_significance, 'stage1', True)
    assert rel_sig_pairs == rel_stats_pairs


def test_two_stage_fast():
    """Fast version of two-stage test."""
    n = 10  # Reduced from 20
    tslist1, tslist2 = create_correlated_ts_fast(n, T=1000)  # Reduced from 10000
    
    computed_stats, computed_significance, info = compute_me_stats(
        tslist1,
        tslist2,
        mode='two_stage',
        n_shuffles_stage1=10,  # Reduced from 100
        n_shuffles_stage2=50,  # Reduced from 1000
        joint_distr=False,
        metric_distr_type='gamma',
        noise_ampl=1e-3,
        ds=2,  # Downsample
        verbose=False
    )
    
    # Check basic properties - significance is nested dict
    # Structure: {neuron_idx: {feature_idx: {'stage1': bool, 'stage2': bool}}}
    assert isinstance(computed_significance, dict)
    assert len(computed_significance) > 0
    # Check that some pairs were tested
    first_neuron = list(computed_significance.keys())[0]
    first_feature = list(computed_significance[first_neuron].keys())[0]
    assert 'stage1' in computed_significance[first_neuron][first_feature]
    assert 'stage2' in computed_significance[first_neuron][first_feature]
    

def test_compute_cell_feat_significance_fast():
    """Fast integration test."""
    from src.driada.intense.pipelines import compute_cell_feat_significance
    
    # Small synthetic experiment
    exp = generate_synthetic_exp(
        n_dfeats=2,
        n_cfeats=1,
        nneurons=3,  # Reduced from 8
        duration=400,  # Reduced from default 1200
        seed=42,
        fps=10  # Lower fps for fewer data points
    )
    
    stats, significance, info, results = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],  # Test subset
        mode='stage1',
        n_shuffles_stage1=10,
        verbose=False,
        enable_parallelization=True,
        ds=2  # Downsample
    )
    
    assert isinstance(stats, dict)
    assert isinstance(significance, dict)
    

@pytest.mark.parametrize("n,T,expected_pairs", [
    (10, 500, 1),   # Tiny test
    (20, 1000, 2),  # Small test
    (30, 2000, 3),  # Medium test
])
def test_correlation_detection_scaled(n, T, expected_pairs):
    """Test correlation detection at different scales."""
    tslist1, tslist2 = create_correlated_ts_fast(n, T)
    
    computed_stats, computed_significance, info = compute_me_stats(
        tslist1,
        tslist2,
        mode='stage1',
        n_shuffles_stage1=10,
        ds=2,
        verbose=False
    )
    
    # Should detect at least expected_pairs correlations
    sig_pairs = retrieve_relevant_from_nested_dict(computed_significance, 'stage1', True)
    assert len(sig_pairs) >= expected_pairs