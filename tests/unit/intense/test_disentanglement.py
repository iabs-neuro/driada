"""Tests for disentanglement module functions."""

import pytest
import numpy as np
from driada.intense.disentanglement import (
    disentangle_pair,
    disentangle_all_selectivities,
    create_multifeature_map,
    get_disentanglement_summary,
    DEFAULT_MULTIFEATURE_MAP,
)
from driada.information.info_base import TimeSeries


def create_redundant_timeseries(n_points=1000):
    """Create time series with redundant information."""
    np.random.seed(42)
    base = np.random.randn(n_points)
    ts1_data = base + 0.5 * np.random.randn(n_points)  # Neural activity
    ts2_data = base + 0.3 * np.random.randn(n_points)  # Behavior 1
    ts3_data = base + 0.3 * np.random.randn(n_points)  # Behavior 2

    return (
        TimeSeries(ts1_data, discrete=False),
        TimeSeries(ts2_data, discrete=False),
        TimeSeries(ts3_data, discrete=False),
    )


def create_synergistic_timeseries(n_points=1000):
    """Create time series with synergistic information."""
    np.random.seed(42)
    ts2_data = np.random.randn(n_points)
    ts3_data = np.random.randn(n_points)
    ts1_data = np.sign(ts2_data) * np.sign(ts3_data) + 0.2 * np.random.randn(n_points)

    return (
        TimeSeries(ts1_data, discrete=False),
        TimeSeries(ts2_data, discrete=False),
        TimeSeries(ts3_data, discrete=False),
    )


def create_undistinguishable_timeseries(n_points=1000):
    """Create time series with undistinguishable contributions."""
    np.random.seed(42)
    ts2_data = np.random.randn(n_points)
    ts3_data = np.random.randn(n_points)
    ts1_data = 0.5 * ts2_data + 0.5 * ts3_data + 0.3 * np.random.randn(n_points)

    return (
        TimeSeries(ts1_data, discrete=False),
        TimeSeries(ts2_data, discrete=False),
        TimeSeries(ts3_data, discrete=False),
    )


def test_disentangle_pair_redundant():
    """Test disentangle_pair with redundant features."""
    ts1, ts2, ts3 = create_redundant_timeseries()
    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)

    # Result should be 0, 1, or 0.5
    assert 0 <= result <= 1
    assert result in [0, 0.5, 1]


def test_disentangle_pair_redundant_verbose():
    """Test verbose output of disentangle_pair."""
    ts1, ts2, ts3 = create_redundant_timeseries()

    # Run with verbose=True to test print statements
    result = disentangle_pair(ts1, ts2, ts3, verbose=True, ds=1)
    assert 0 <= result <= 1


def test_disentangle_pair_synergistic():
    """Test disentangle_pair with synergistic features."""
    ts1, ts2, ts3 = create_synergistic_timeseries()
    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)

    # Should return valid result
    assert 0 <= result <= 1


def test_disentangle_pair_undistinguishable():
    """Test disentangle_pair with undistinguishable features."""
    ts1, ts2, ts3 = create_undistinguishable_timeseries()
    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)

    # Undistinguishable contributions should return 0.5
    assert result == 0.5


def test_disentangle_pair_zero_mi():
    """Test when one feature has zero MI with neural activity."""
    np.random.seed(42)
    n_points = 1000
    ts2_data = np.random.randn(n_points)
    ts3_data = np.random.randn(n_points)
    ts1_data = ts2_data + 0.2 * np.random.randn(n_points)  # Only depends on ts2

    ts1 = TimeSeries(ts1_data, discrete=False)
    ts2 = TimeSeries(ts2_data, discrete=False)
    ts3 = TimeSeries(ts3_data, discrete=False)

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)

    # ts2 should be primary (result = 0)
    assert result == 0


def test_disentangle_pair_dominant_feature():
    """Test when one feature strongly dominates."""
    np.random.seed(42)
    n_points = 1000
    ts2_data = np.random.randn(n_points)
    ts3_data = np.random.randn(n_points)
    ts1_data = 2.5 * ts2_data + 0.1 * ts3_data + 0.2 * np.random.randn(n_points)

    ts1 = TimeSeries(ts1_data, discrete=False)
    ts2 = TimeSeries(ts2_data, discrete=False)
    ts3 = TimeSeries(ts3_data, discrete=False)

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)

    # ts2 should be dominant (result = 0)
    assert result == 0


def test_disentangle_pair_downsampling():
    """Test disentangle_pair with different downsampling factors."""
    ts1, ts2, ts3 = create_redundant_timeseries()

    result_ds1 = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)
    result_ds2 = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=2)
    result_ds4 = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=4)

    # All should return valid results
    assert all(0 <= r <= 1 for r in [result_ds1, result_ds2, result_ds4])


def test_disentangle_pair_discrete():
    """Test with discrete time series."""
    np.random.seed(42)
    n_points = 1000

    # Create discrete time series with dependencies
    # Note: conditional MI requires continuous X (ts1), so make ts1 continuous
    ts2_data = np.random.choice([0, 1], size=n_points)
    ts3_data = np.random.choice([0, 1], size=n_points)
    # ts1 depends on both but is continuous
    ts1_data = 0.5 * ts2_data + 0.5 * ts3_data + 0.1 * np.random.randn(n_points)

    ts1 = TimeSeries(ts1_data, discrete=False)  # Must be continuous
    ts2 = TimeSeries(ts2_data, discrete=True)
    ts3 = TimeSeries(ts3_data, discrete=True)

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)
    assert 0 <= result <= 1


@pytest.mark.parametrize("mixed_features_experiment", ["medium"], indirect=True)
def test_disentangle_all_selectivities_basic(mixed_features_experiment):
    """Test basic functionality of disentangle_all_selectivities."""
    # Use fixture (medium has sufficient duration and neurons)
    exp = mixed_features_experiment

    # Initialize selectivity tables
    exp._set_selectivity_tables("calcium")

    # Get actual feature names from experiment
    feat_names = list(exp.dynamic_features.keys())

    # Run analysis
    disent_matrix, count_matrix = disentangle_all_selectivities(exp, feat_names, ds=2)

    # Check dimensions
    n_features = len(feat_names)
    assert disent_matrix.shape == (n_features, n_features)
    assert count_matrix.shape == (n_features, n_features)

    # Check values are non-negative
    assert np.all(disent_matrix >= 0)
    assert np.all(count_matrix >= 0)


@pytest.mark.parametrize("medium_experiment", ["medium"], indirect=True)
def test_disentangle_all_selectivities_cell_bunch(medium_experiment):
    """Test with specific cell subset."""
    # Use medium fixture (has mixed features, just use subset of cells)
    exp = medium_experiment

    # Initialize selectivity tables
    exp._set_selectivity_tables("calcium")

    # Get actual feature names from experiment (use first 3)
    feat_names = list(exp.dynamic_features.keys())[:3]

    # Test with subset of cells
    disent_matrix, count_matrix = disentangle_all_selectivities(
        exp, feat_names, ds=2, cell_bunch=[0, 1, 2]
    )

    assert disent_matrix.shape == (3, 3)
    assert count_matrix.shape == (3, 3)


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_disentangle_all_selectivities_with_significance(mixed_features_experiment):
    """Test with feature-feature significance matrix."""
    # Use small fixture
    exp = mixed_features_experiment

    # Initialize selectivity tables
    exp._set_selectivity_tables("calcium")

    # Get actual feature names from experiment
    feat_names = list(exp.dynamic_features.keys())
    n_features = len(feat_names)

    # Create significance matrix appropriate for actual feature count
    feat_feat_significance = np.zeros((n_features, n_features))
    if n_features >= 2:
        feat_feat_significance[0, 1] = 1
        feat_feat_significance[1, 0] = 1
    if n_features >= 4:
        feat_feat_significance[2, 3] = 1
        feat_feat_significance[3, 2] = 1

    disent_matrix, count_matrix = disentangle_all_selectivities(
        exp, feat_names, ds=2, feat_feat_significance=feat_feat_significance
    )

    assert disent_matrix.shape == (n_features, n_features)
    assert count_matrix.shape == (n_features, n_features)


@pytest.mark.parametrize("multifeature_experiment", ["small"], indirect=True)
def test_disentangle_all_selectivities_multifeature(multifeature_experiment):
    """Test with multifeature mapping."""
    # Use multifeature fixture (guaranteed to have at least 4 continuous features)
    exp = multifeature_experiment

    # Initialize selectivity tables
    exp._set_selectivity_tables("calcium")

    # Get available continuous features
    c_feats = [k for k in exp.dynamic_features.keys() if k.startswith("fbm_")]

    # Add x and y attributes from first two continuous features
    exp.x = exp.dynamic_features[c_feats[0]]
    exp.y = exp.dynamic_features[c_feats[1]]

    multifeature_map = {("x", "y"): "place"}
    remaining_feats = c_feats[2:] if len(c_feats) > 2 else []
    feat_names = ["place"] + remaining_feats
    expected_shape = len(feat_names)

    # This might raise error if no neurons have selectivity
    try:
        disent_matrix, count_matrix = disentangle_all_selectivities(
            exp, feat_names, ds=2, multifeature_map=multifeature_map
        )
        assert disent_matrix.shape == (expected_shape, expected_shape)
        assert count_matrix.shape == (expected_shape, expected_shape)
    except ValueError as e:
        # Expected if features not found
        assert "not in feat_names" in str(e) or "not found" in str(e)


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_disentangle_all_selectivities_empty_neurons(mixed_features_experiment):
    """Test when no neurons have significant selectivity."""
    # Use small fixture (will mock empty neurons)
    exp = mixed_features_experiment

    # Mock empty significant neurons
    exp.get_significant_neurons = lambda min_nspec=2, cbunch=None: {}

    # Use actual feature names
    feat_names = list(exp.dynamic_features.keys())[:2]
    disent_matrix, count_matrix = disentangle_all_selectivities(exp, feat_names, ds=1)

    # Should return zero matrices
    assert np.all(disent_matrix == 0)
    assert np.all(count_matrix == 0)


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_disentangle_all_selectivities_error_handling(mixed_features_experiment):
    """Test error handling in disentangle_all_selectivities."""
    # Use small fixture
    exp = mixed_features_experiment

    # Initialize selectivity tables
    exp._set_selectivity_tables("calcium")

    # Include real features and non-existent one
    real_features = list(exp.dynamic_features.keys())[:2]
    feat_names = real_features + ["nonexistent"]

    # Should handle gracefully
    disent_matrix, count_matrix = disentangle_all_selectivities(exp, feat_names, ds=2)

    assert disent_matrix.shape == (3, 3)
    assert count_matrix.shape == (3, 3)


@pytest.mark.parametrize("multifeature_experiment", ["small"], indirect=True)
def test_create_multifeature_map_valid(multifeature_experiment):
    """Test creating valid multifeature map."""
    # Use multifeature fixture (guaranteed to have at least 4 continuous features)
    exp = multifeature_experiment

    # Get actual continuous features
    c_feats = [k for k in exp.dynamic_features.keys() if k.startswith("fbm_")]

    # Add attributes from actual features
    exp.x = exp.dynamic_features[c_feats[0]]
    exp.y = exp.dynamic_features[c_feats[1]]
    exp.speed = exp.dynamic_features[c_feats[2]]
    exp.head_direction = exp.dynamic_features[c_feats[3]]

    mapping_dict = {("x", "y"): "place", ("speed", "head_direction"): "locomotion"}

    validated_map = create_multifeature_map(exp, mapping_dict)

    # Check tuples are sorted
    assert ("x", "y") in validated_map
    assert ("head_direction", "speed") in validated_map  # Sorted
    assert validated_map[("x", "y")] == "place"
    assert validated_map[("head_direction", "speed")] == "locomotion"


@pytest.mark.parametrize("continuous_only_experiment", ["small"], indirect=True)
def test_create_multifeature_map_invalid(continuous_only_experiment):
    """Test error when component doesn't exist."""
    # Use continuous fixture
    exp = continuous_only_experiment

    # Get first continuous feature
    c_feats = [k for k in exp.dynamic_features.keys() if k.startswith("fbm_")]
    if c_feats:
        exp.x = exp.dynamic_features[c_feats[0]]
    # Don't add y

    mapping_dict = {("x", "y"): "place"}

    with pytest.raises(ValueError, match="Component 'y'.*not found"):
        create_multifeature_map(exp, mapping_dict)


@pytest.mark.parametrize("mixed_features_experiment", ["small"], indirect=True)
def test_create_multifeature_map_empty(mixed_features_experiment):
    """Test with empty mapping."""
    # Use mixed fixture
    exp = mixed_features_experiment

    validated_map = create_multifeature_map(exp, {})
    assert validated_map == {}


def test_get_disentanglement_summary_basic():
    """Test basic summary generation."""
    feat_names = ["feat1", "feat2", "feat3"]

    # Create test matrices
    disent_matrix = np.array([[0, 5, 2], [3, 0, 1], [2, 4, 0]])
    count_matrix = np.array([[0, 8, 4], [8, 0, 5], [4, 5, 0]])

    summary = get_disentanglement_summary(disent_matrix, count_matrix, feat_names)

    # Check structure
    assert "feature_pairs" in summary
    assert "overall_stats" in summary

    # Check feature pairs
    assert "feat1_vs_feat2" in summary["feature_pairs"]
    assert "feat1_vs_feat3" in summary["feature_pairs"]
    assert "feat2_vs_feat3" in summary["feature_pairs"]

    # Check pair summary contents
    pair = summary["feature_pairs"]["feat1_vs_feat2"]
    assert "total_neurons" in pair
    assert "feat1_primary" in pair
    assert "feat2_primary" in pair
    assert "undistinguishable_pct" in pair
    assert "redundant_pct" in pair

    # Check overall stats
    stats = summary["overall_stats"]
    assert "total_neuron_pairs" in stats
    assert "redundancy_rate" in stats
    assert "undistinguishable_rate" in stats


def test_get_disentanglement_summary_with_significance():
    """Test summary with significance matrix."""
    feat_names = ["feat1", "feat2"]

    disent_matrix = np.array([[0, 3], [2, 0]])
    count_matrix = np.array([[0, 5], [5, 0]])
    feat_feat_significance = np.array([[0, 1], [1, 0]])

    summary = get_disentanglement_summary(
        disent_matrix,
        count_matrix,
        feat_names,
        feat_feat_significance=feat_feat_significance,
    )

    # Should include significance breakdown
    stats = summary["overall_stats"]
    assert "significant_behavior_pairs" in stats
    assert "nonsignificant_behavior_pairs" in stats
    assert "true_mixed_selectivity_rate" in stats


def test_get_disentanglement_summary_empty():
    """Test with empty matrices."""
    feat_names = ["feat1", "feat2"]

    disent_matrix = np.zeros((2, 2))
    count_matrix = np.zeros((2, 2))

    summary = get_disentanglement_summary(disent_matrix, count_matrix, feat_names)

    # Should handle gracefully
    assert summary["feature_pairs"] == {}
    assert "overall_stats" not in summary or summary["overall_stats"] == {}


def test_get_disentanglement_summary_redundant():
    """Test summary with redundant features (correlated features that passed significance)."""
    feat_names = ["feat1", "feat2"]

    # Test case: 5 neurons analyzed, mixed results
    # - 3 neurons where feat1 is primary (disres = 0)
    # - 1 neuron where feat2 is primary (disres = 1)
    # - 1 neuron where both contribute (disres = 0.5)
    disent_matrix = np.array(
        [
            [0, 3.5],  # feat1 column: 3 primary + 0.5 from shared
            [1.5, 0],  # feat2 column: 1 primary + 0.5 from shared
        ]
    )
    count_matrix = np.array([[0, 5], [5, 0]])

    summary = get_disentanglement_summary(disent_matrix, count_matrix, feat_names)

    pair = summary["feature_pairs"]["feat1_vs_feat2"]
    # Check percentages
    assert pair["feat1_primary"] == 3.5 / 5 * 100  # 70%
    assert pair["feat2_primary"] == 1.5 / 5 * 100  # 30%

    # With the corrected formula:
    # Fractional parts: 0.5 each, so n_undistinguishable = 0.5 * 2 = 1
    # n_redundant = 5 - 1 = 4
    assert pair["undistinguishable_pct"] == 20.0  # 1 out of 5
    assert pair["redundant_pct"] == 80.0  # 4 out of 5


# Edge case tests


def test_disentangle_pair_short_timeseries():
    """Test with very short time series."""
    ts1 = TimeSeries(np.random.randn(10), discrete=False)
    ts2 = TimeSeries(np.random.randn(10), discrete=False)
    ts3 = TimeSeries(np.random.randn(10), discrete=False)

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)
    assert 0 <= result <= 1


def test_disentangle_pair_identical_timeseries():
    """Test with nearly identical time series."""
    data = np.random.randn(100)

    # Add tiny noise to avoid numerical issues
    ts1 = TimeSeries(data + 1e-6 * np.random.randn(100), discrete=False)
    ts2 = TimeSeries(data + 1e-6 * np.random.randn(100), discrete=False)
    ts3 = TimeSeries(data + 1e-6 * np.random.randn(100), discrete=False)

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)
    assert 0 <= result <= 1


def test_disentangle_pair_constant_timeseries():
    """Test with constant time series."""
    # Add tiny noise to constant to avoid issues
    ts1 = TimeSeries(np.ones(100) + 1e-8 * np.random.randn(100), discrete=False)
    ts2 = TimeSeries(np.random.randn(100), discrete=False)
    ts3 = TimeSeries(np.random.randn(100), discrete=False)

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)
    assert 0 <= result <= 1


def test_disentangle_pair_mixed_discrete_continuous():
    """Test with mixed discrete and continuous time series."""
    ts1 = TimeSeries(np.random.randn(100), discrete=False)  # Continuous
    ts2 = TimeSeries(np.random.choice([0, 1], 100), discrete=True)  # Discrete
    ts3 = TimeSeries(np.random.randn(100), discrete=False)  # Continuous

    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=1)
    assert 0 <= result <= 1


def test_disentangle_pair_high_downsampling():
    """Test with very high downsampling factor."""
    ts1, ts2, ts3 = create_redundant_timeseries(n_points=1000)

    # Test with ds=10 (only 100 points remain)
    result = disentangle_pair(ts1, ts2, ts3, verbose=False, ds=10)
    assert 0 <= result <= 1


def test_default_multifeature_map():
    """Test that DEFAULT_MULTIFEATURE_MAP is properly defined."""
    assert isinstance(DEFAULT_MULTIFEATURE_MAP, dict)
    assert ("x", "y") in DEFAULT_MULTIFEATURE_MAP
    assert DEFAULT_MULTIFEATURE_MAP[("x", "y")] == "place"
