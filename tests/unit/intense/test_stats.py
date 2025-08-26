"""Tests for statistical functions in stats.py."""

import numpy as np
import pytest
from scipy import stats as scipy_stats
from driada.intense.stats import (
    chebyshev_ineq,
    get_lognormal_p,
    get_gamma_p,
    get_distribution_function,
    get_mi_distr_pvalue,
    get_mask,
    stats_not_empty,
    criterion1,
    criterion2,
    get_all_nonempty_pvals,
    get_table_of_stats,
    merge_stage_stats,
    merge_stage_significance,
)


def test_chebyshev_ineq():
    """Test Chebyshev's inequality calculation."""
    np.random.seed(42)

    # Test 1: Normal distribution
    data = np.random.normal(10, 2, 1000)

    # Test value 2 standard deviations away
    val = 14  # mean=10, std≈2, so 2 stds away
    p_bound = chebyshev_ineq(data, val)
    # Chebyshev: P(|X-μ| ≥ kσ) ≤ 1/k²
    # For k=2: bound should be ≤ 0.25
    assert p_bound <= 0.26  # Small tolerance for sample std

    # Test 2: Value at mean (k=0, should give infinity)
    val = np.mean(data)
    p_bound = chebyshev_ineq(data, val)
    assert p_bound > 1e6  # Should be very large

    # Test 3: Value 3 standard deviations away
    val = 16  # ~3 stds away
    p_bound = chebyshev_ineq(data, val)
    assert p_bound <= 0.12  # 1/9 ≈ 0.111

    # Test 4: Edge case - single value data (should raise error)
    data_single = np.array([5.0])
    with pytest.raises(ValueError, match="zero variance"):
        chebyshev_ineq(data_single, 6.0)


def test_get_lognormal_p():
    """Test log-normal p-value calculation."""
    np.random.seed(42)

    # Test 1: Known log-normal distribution
    # Generate log-normal data with known parameters
    data = np.random.lognormal(mean=0, sigma=1, size=1000)

    # Test p-value at median (should be ~0.5)
    median_val = np.exp(0)  # For lognormal(0,1), median = exp(μ) = 1
    p_val = get_lognormal_p(data, median_val)
    assert 0.3 < p_val < 0.7  # Should be around 0.5

    # Test 2: Large value (should give small p-value)
    large_val = 10.0
    p_val = get_lognormal_p(data, large_val)
    assert p_val < 0.05

    # Test 3: Small value (should give large p-value)
    small_val = 0.1
    p_val = get_lognormal_p(data, small_val)
    assert p_val > 0.8

    # Test 4: Edge case - all same values
    data_same = np.ones(100) * 2.0
    p_val = get_lognormal_p(data_same, 2.5)
    assert 0 <= p_val <= 1
    
    # Test 5: Negative values should raise error
    data_negative = np.array([-1, 1, 2, 3])
    with pytest.raises(ValueError, match="positive values"):
        get_lognormal_p(data_negative, 1.0)


def test_get_gamma_p():
    """Test gamma p-value calculation."""
    np.random.seed(42)

    # Test 1: Known gamma distribution
    # Generate gamma data with shape=2, scale=2
    data = np.random.gamma(shape=2, scale=2, size=1000)

    # Test p-value at mean (shape*scale = 4)
    mean_val = 4.0
    p_val = get_gamma_p(data, mean_val)
    assert 0.3 < p_val < 0.7  # Should be around 0.5

    # Test 2: Large value
    large_val = 15.0
    p_val = get_gamma_p(data, large_val)
    assert p_val < 0.1

    # Test 3: Small value
    small_val = 0.5
    p_val = get_gamma_p(data, small_val)
    assert p_val > 0.8

    # Test 4: Zero value (gamma starts at 0)
    p_val = get_gamma_p(data, 0.0)
    assert p_val == 1.0
    
    # Test 5: Negative values should raise error
    data_negative = np.array([-1, 1, 2, 3])
    with pytest.raises(ValueError, match="positive values"):
        get_gamma_p(data_negative, 1.0)


def test_get_distribution_function():
    """Test distribution function retrieval."""
    # Test 1: Valid distributions
    gamma_dist = get_distribution_function("gamma")
    assert gamma_dist == scipy_stats.gamma

    norm_dist = get_distribution_function("norm")
    assert norm_dist == scipy_stats.norm

    lognorm_dist = get_distribution_function("lognorm")
    assert lognorm_dist == scipy_stats.lognorm

    # Test 2: Invalid distribution
    with pytest.raises(ValueError, match="Distribution 'invalid_dist' not found"):
        get_distribution_function("invalid_dist")


def test_get_mi_distr_pvalue():
    """Test general MI distribution p-value calculation."""
    np.random.seed(42)

    # Test 1: Gamma distribution (default)
    data = np.random.gamma(2, 2, 1000)
    val = 4.0
    p_val = get_mi_distr_pvalue(data, val)
    assert 0.3 < p_val < 0.7

    # Test 2: Normal distribution
    data = np.random.normal(10, 2, 1000)
    val = 12.0
    p_val = get_mi_distr_pvalue(data, val, distr_type="norm")
    assert 0.1 < p_val < 0.3  # ~1 std above mean

    # Test 3: Log-normal distribution
    data = np.random.lognormal(0, 1, 1000)
    val = 1.0
    p_val = get_mi_distr_pvalue(data, val, distr_type="lognorm")
    assert 0.3 < p_val < 0.7

    # Test 4: Edge case - very small sample
    data_small = np.array([1.0, 2.0, 3.0])
    p_val = get_mi_distr_pvalue(data_small, 2.5)
    assert 0 <= p_val <= 1


def test_get_mask():
    """Test binary mask creation."""
    # Test 1: Basic masking
    ptable = np.array([[0.01, 0.05], [0.1, 0.001]])
    rtable = np.array([[0.99, 0.95], [0.9, 0.999]])
    pval_thr = 0.05
    rank_thr = 0.95

    mask = get_mask(ptable, rtable, pval_thr, rank_thr)
    # p <= pval_thr AND r >= rank_thr pass
    # (0,0): p=0.01<=0.05 AND r=0.99>=0.95 → 1
    # (0,1): p=0.05<=0.05 AND r=0.95>=0.95 → 1
    # (1,0): p=0.1>0.05 → 0
    # (1,1): p=0.001<=0.05 AND r=0.999>=0.95 → 1
    expected = np.array([[1, 1], [0, 1]])
    np.testing.assert_array_equal(mask, expected)

    # Test 2: All pass
    ptable = np.array([[0.001, 0.001], [0.001, 0.001]])
    rtable = np.array([[0.99, 0.99], [0.99, 0.99]])
    mask = get_mask(ptable, rtable, pval_thr, rank_thr)
    np.testing.assert_array_equal(mask, np.ones((2, 2)))

    # Test 3: None pass
    ptable = np.array([[0.1, 0.2], [0.3, 0.4]])
    rtable = np.array([[0.9, 0.8], [0.7, 0.6]])
    mask = get_mask(ptable, rtable, pval_thr, rank_thr)
    np.testing.assert_array_equal(mask, np.zeros((2, 2)))

    # Test 4: Different shapes
    ptable = np.array([0.01, 0.1, 0.001])
    rtable = np.array([0.99, 0.9, 0.999])
    mask = get_mask(ptable, rtable, pval_thr, rank_thr)
    expected = np.array([1, 0, 1])
    np.testing.assert_array_equal(mask, expected)


def test_stats_not_empty():
    """Test statistics validation."""
    # Test 1: Valid stage 1 stats
    pair_stats = {"data_hash": "abc123", "pre_rval": 0.95, "pre_pval": 0.01, "me": 0.5}
    assert stats_not_empty(pair_stats, "abc123", stage=1) == True

    # Test 2: Invalid hash
    assert stats_not_empty(pair_stats, "wrong_hash", stage=1) == False

    # Test 3: Missing stage 1 stats
    pair_stats_incomplete = {"data_hash": "abc123", "pre_rval": None, "pre_pval": 0.01}
    assert stats_not_empty(pair_stats_incomplete, "abc123", stage=1) == False

    # Test 4: Valid stage 2 stats
    pair_stats_stage2 = {"data_hash": "abc123", "rval": 0.98, "pval": 0.001, "me": 0.7}
    assert stats_not_empty(pair_stats_stage2, "abc123", stage=2) == True

    # Test 5: Invalid stage
    with pytest.raises(ValueError, match="Stage should be 1 or 2"):
        stats_not_empty(pair_stats, "abc123", stage=3)


def test_criterion1():
    """Test stage 1 significance criterion."""
    # Test 1: Passes criterion (top-1)
    pair_stats = {"pre_rval": 1.0}
    assert criterion1(pair_stats, nsh1=100, topk=1) == True

    # Test 2: Passes with topk=5
    pair_stats = {"pre_rval": 0.96}  # In top 5 of 100
    assert criterion1(pair_stats, nsh1=100, topk=5) == True

    # Test 3: Fails criterion
    pair_stats = {"pre_rval": 0.9}
    assert criterion1(pair_stats, nsh1=100, topk=1) == False

    # Test 4: Missing pre_rval
    pair_stats = {}
    assert criterion1(pair_stats, nsh1=100) == False

    # Test 5: None pre_rval
    pair_stats = {"pre_rval": None}
    assert criterion1(pair_stats, nsh1=100) == False


def test_criterion2():
    """Test stage 2 significance criterion."""
    # Test 1: Passes all criteria
    pair_stats = {"rval": 0.999, "pval": 0.0001}  # Top 1 of 1000  # < 0.01
    assert criterion2(pair_stats, nsh2=1000, pval_thr=0.01, topk=5) == True

    # Test 2: Fails rank criterion
    pair_stats = {"rval": 0.99, "pval": 0.0001}  # Not in top 5
    assert criterion2(pair_stats, nsh2=1000, pval_thr=0.01, topk=5) == False

    # Test 3: Fails p-value criterion
    pair_stats = {"rval": 0.999, "pval": 0.02}  # > 0.01
    assert criterion2(pair_stats, nsh2=1000, pval_thr=0.01, topk=5) == False

    # Test 4: Missing values
    pair_stats = {"rval": 0.999}  # Missing pval
    assert criterion2(pair_stats, nsh2=1000, pval_thr=0.01) == False

    # Test 5: None values
    pair_stats = {"rval": None, "pval": None}
    assert criterion2(pair_stats, nsh2=1000, pval_thr=0.01) == False


def test_get_all_nonempty_pvals():
    """Test p-value extraction from nested dictionary."""
    # Test 1: Normal case
    all_stats = {
        "id1": {"id_a": {"pval": 0.01}, "id_b": {"pval": 0.05}, "id_c": {"pval": None}},
        "id2": {
            "id_a": {"pval": 0.001},
            "id_b": {},  # Missing pval
            "id_c": {"pval": 0.1},
        },
    }
    ids1 = ["id1", "id2"]
    ids2 = ["id_a", "id_b", "id_c"]

    pvals = get_all_nonempty_pvals(all_stats, ids1, ids2)
    expected = [0.01, 0.05, 0.001, 0.1]
    assert sorted(pvals) == sorted(expected)

    # Test 2: All None
    all_stats_none = {
        "id1": {"id_a": {"pval": None}, "id_b": {"pval": None}},
        "id2": {"id_a": {"pval": None}, "id_b": {}},
    }
    pvals = get_all_nonempty_pvals(all_stats_none, ["id1", "id2"], ["id_a", "id_b"])
    assert pvals == []

    # Test 3: Empty stats
    pvals = get_all_nonempty_pvals({}, [], [])
    assert pvals == []


def test_get_table_of_stats():
    """Test conversion of metric table to statistics dictionary."""
    np.random.seed(42)

    # Test 1: Basic conversion
    # metable shape: (n1, n2, nsh+1) where [:,:,0] is true values
    metable = np.random.gamma(2, 0.1, size=(2, 3, 101))
    metable[:, :, 0] = np.array([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]])  # True values

    optimal_delays = np.array([[1, 2, 3], [4, 5, 6]])

    stats = get_table_of_stats(
        metable, optimal_delays, metric_distr_type="gamma", nsh=100, stage=1
    )

    # Check structure
    assert len(stats) == 2
    assert len(stats[0]) == 3

    # Check stage 1 stats
    assert "pre_rval" in stats[0][0]
    assert "pre_pval" in stats[0][0]
    assert "opt_delay" in stats[0][0]
    assert "me" in stats[0][0]
    assert stats[0][0]["opt_delay"] == 1
    assert stats[0][0]["me"] == 0.5

    # Test 2: Stage 2
    stats2 = get_table_of_stats(
        metable, optimal_delays, metric_distr_type="gamma", nsh=100, stage=2
    )

    assert "rval" in stats2[0][0]
    assert "pval" in stats2[0][0]
    assert "me" in stats2[0][0]
    assert "opt_delay" in stats2[0][0]

    # Test 3: With mask
    mask = np.array([[1, 0, 1], [0, 1, 0]])
    stats_masked = get_table_of_stats(
        metable,
        optimal_delays,
        precomputed_mask=mask,
        metric_distr_type="gamma",
        nsh=100,
        stage=1,
    )

    # Check only masked entries have stats
    assert "pre_rval" in stats_masked[0][0]  # mask[0,0] = 1
    assert "pre_rval" not in stats_masked[0][1]  # mask[0,1] = 0


def test_merge_stage_stats():
    """Test merging statistics from stage 1 and stage 2."""
    # Test 1: Basic merge
    stage1_stats = {
        0: {
            0: {"pre_rval": 0.95, "pre_pval": 0.01, "opt_delay": 1},
            1: {"pre_rval": 0.90, "pre_pval": 0.05, "opt_delay": 2},
        },
        1: {0: {"pre_rval": 0.98, "pre_pval": 0.001, "opt_delay": 3}, 1: {}},  # Empty
    }

    stage2_stats = {
        0: {
            0: {"rval": 0.99, "pval": 0.001, "me": 0.5},
            1: {"rval": 0.95, "pval": 0.01, "me": 0.3},
        },
        1: {
            0: {"rval": 0.999, "pval": 0.0001, "me": 0.7},
            1: {"rval": 0.8, "pval": 0.1, "me": 0.1},
        },
    }

    merged = merge_stage_stats(stage1_stats, stage2_stats)

    # Check merge
    assert merged[0][0]["pre_rval"] == 0.95
    assert merged[0][0]["rval"] == 0.99
    assert merged[0][0]["me"] == 0.5

    # Check empty entry handling
    assert "pre_rval" not in merged[1][1]
    assert merged[1][1]["rval"] == 0.8

    # Test 2: Missing keys in stage1
    stage1_partial = {0: {0: {"pre_rval": 0.95}}}
    stage2_full = {
        0: {0: {"rval": 0.99, "pval": 0.001}},
        1: {0: {"rval": 0.98, "pval": 0.01}},
    }

    merged = merge_stage_stats(stage1_partial, stage2_full)
    assert merged[0][0]["pre_rval"] == 0.95
    assert merged[1][0]["rval"] == 0.98
    assert "pre_rval" not in merged[1][0]


def test_merge_stage_significance():
    """Test merging significance results from stages."""
    # Test 1: Basic merge
    stage1_sig = {
        0: {
            0: {"stage1_passed": True, "stage1_pval": 0.01},
            1: {"stage1_passed": False, "stage1_pval": 0.1},
        },
        1: {
            0: {"stage1_passed": True, "stage1_pval": 0.001},
            1: {"stage1_passed": False, "stage1_pval": 0.2},
        },
    }

    stage2_sig = {
        0: {
            0: {"stage2_passed": True, "final_pval": 0.001},
            1: {"stage2_passed": False, "final_pval": 0.05},
        },
        1: {
            0: {"stage2_passed": True, "final_pval": 0.0001},
            1: {"stage2_passed": False, "final_pval": 0.15},
        },
    }

    merged = merge_stage_significance(stage1_sig, stage2_sig)

    # Check merge
    assert merged[0][0]["stage1_passed"] == True
    assert merged[0][0]["stage2_passed"] == True
    assert merged[0][0]["stage1_pval"] == 0.01
    assert merged[0][0]["final_pval"] == 0.001

    # Test 2: Empty values in stage 1
    stage1_empty = {0: {0: {}, 1: {}}, 1: {0: {}, 1: {}}}
    merged = merge_stage_significance(stage1_empty, stage2_sig)

    assert merged[0][0]["stage2_passed"] == True
    assert merged[0][0]["final_pval"] == 0.001


def test_edge_cases_and_numerical_stability():
    """Test edge cases and numerical stability."""
    # Test 1: Empty arrays
    empty_data = np.array([])

    # Chebyshev with empty data returns nan
    p_bound = chebyshev_ineq(empty_data, 1.0)
    assert np.isnan(p_bound)

    # Test 3: Extreme values
    extreme_data = np.array([1e-10, 1e10])
    p_val = get_mi_distr_pvalue(extreme_data, 1.0)
    assert 0 <= p_val <= 1

    # Test 4: Very small positive values (gamma requires positive)
    small_positive = np.array([1e-10, 1e-9, 1e-8])
    p_val = get_gamma_p(small_positive, 0.1)
    assert 0 <= p_val <= 1

    # Test 5: Handle edge case with only positive values for lognormal
    pos_data = np.array([1, 2, 3, 4, 5])
    p_val = get_lognormal_p(pos_data, 2.5)
    assert 0 <= p_val <= 1


def test_distribution_fitting_edge_cases():
    """Test distribution fitting with challenging data."""
    np.random.seed(42)

    # Test 1: Very small variance
    small_var_data = np.ones(100) + np.random.normal(0, 1e-6, 100)
    p_val = get_gamma_p(small_var_data, 1.0)
    assert 0 <= p_val <= 1

    # Test 2: Bimodal data (challenging for single distribution)
    bimodal = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500)])
    p_val = get_mi_distr_pvalue(bimodal, 2.5, distr_type="norm")
    assert 0 <= p_val <= 1

    # Test 3: Heavy-tailed data
    heavy_tail = np.random.standard_t(df=2, size=1000)
    p_val = get_mi_distr_pvalue(heavy_tail, 0.0, distr_type="norm")
    assert 0 <= p_val <= 1
