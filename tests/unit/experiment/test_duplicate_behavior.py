"""Test duplicate behavior parameter functionality."""

import pytest
import numpy as np
from driada.intense.intense_base import compute_me_stats
from driada.intense.pipelines import (
    compute_cell_feat_significance,
    compute_feat_feat_significance,
    compute_cell_cell_significance,
)
from driada.information.info_base import TimeSeries


def test_duplicate_behavior_ignore():
    """Test that 'ignore' behavior processes duplicates normally."""
    # Create simple time series
    ts1 = TimeSeries(np.sin(np.linspace(0, 4 * np.pi, 100)), "ts1")
    ts2 = TimeSeries(np.cos(np.linspace(0, 4 * np.pi, 100)), "ts2")
    ts1_dup = TimeSeries(
        np.sin(np.linspace(0, 4 * np.pi, 100)), "ts1_dup"
    )  # Duplicate data

    # Test with duplicates in both bunches
    ts_bunch1 = [ts1, ts1_dup]  # Contains duplicate
    ts_bunch2 = [ts2]

    # Should run without error
    stats, significance, info = compute_me_stats(
        ts_bunch1,
        ts_bunch2,
        names1=["ts1", "ts1_dup"],
        names2=["ts2"],
        metric="mi",
        mode="stage1",
        n_shuffles_stage1=10,
        duplicate_behavior="ignore",
        verbose=False,
    )

    assert isinstance(stats, dict)
    assert len(stats) == 2  # Both ts1 and ts1_dup should be compared with ts2


def test_duplicate_behavior_warn(capfd):
    """Test that 'warn' behavior prints warning but continues."""
    # Create time series with shared data reference
    data = np.sin(np.linspace(0, 4 * np.pi, 100))
    ts1 = TimeSeries(data, "ts1")
    ts1_dup = TimeSeries(data, "ts1_dup")  # Same data reference
    ts2 = TimeSeries(np.cos(np.linspace(0, 4 * np.pi, 100)), "ts2")

    # Test with duplicates
    ts_bunch1 = [ts1, ts1_dup]
    ts_bunch2 = [ts2]

    # Should print warning but continue
    stats, significance, info = compute_me_stats(
        ts_bunch1,
        ts_bunch2,
        names1=["ts1", "ts1_dup"],
        names2=["ts2"],
        metric="mi",
        mode="stage1",
        n_shuffles_stage1=10,
        duplicate_behavior="warn",
        verbose=False,
    )

    # Capture output
    captured = capfd.readouterr()

    # Check warning was printed
    assert "Warning: Duplicate TimeSeries objects found" in captured.out
    assert "ts_bunch1" in captured.out

    # Check processing continued
    assert isinstance(stats, dict)
    assert len(stats) == 2


def test_duplicate_behavior_raise():
    """Test that 'raise' behavior raises error on duplicates."""
    # Create time series with shared data reference
    data = np.sin(np.linspace(0, 4 * np.pi, 100))
    ts1 = TimeSeries(data, "ts1")
    ts1_dup = TimeSeries(data, "ts1_dup")  # Same data reference
    ts2 = TimeSeries(np.cos(np.linspace(0, 4 * np.pi, 100)), "ts2")

    # Test with duplicates in ts_bunch1
    ts_bunch1 = [ts1, ts1_dup]
    ts_bunch2 = [ts2]

    # Should raise ValueError
    with pytest.raises(ValueError, match="Duplicate TimeSeries objects found"):
        compute_me_stats(
            ts_bunch1,
            ts_bunch2,
            names1=["ts1", "ts1_dup"],
            names2=["ts2"],
            metric="mi",
            mode="stage1",
            n_shuffles_stage1=10,
            duplicate_behavior="raise",
            verbose=False,
        )


def test_duplicate_behavior_in_pipelines(small_experiment):
    """Test duplicate behavior parameter in pipeline functions."""
    # Use fixture for consistent test data
    exp = small_experiment

    # Test compute_cell_feat_significance
    stats, significance, info, results = compute_cell_feat_significance(
        exp,
        cell_bunch=[0, 1],
        feat_bunch=["d_feat_0"],
        mode="stage1",
        n_shuffles_stage1=5,
        duplicate_behavior="ignore",
        verbose=False,
        seed=42,
    )
    assert isinstance(stats, dict)

    # Test compute_feat_feat_significance
    sim_mat, sig_mat, pval_mat, feat_ids, info = compute_feat_feat_significance(
        exp,
        feat_bunch=["d_feat_0", "d_feat_1"],
        mode="stage1",
        n_shuffles_stage1=5,
        duplicate_behavior="warn",
        verbose=False,
        seed=42,
    )
    assert sim_mat.shape == (2, 2)

    # Test compute_cell_cell_significance
    sim_mat, sig_mat, pval_mat, cell_ids, info = compute_cell_cell_significance(
        exp,
        cell_bunch=[0, 1, 2],
        mode="stage1",
        n_shuffles_stage1=5,
        duplicate_behavior="ignore",
        verbose=False,
        seed=42,
    )
    assert sim_mat.shape == (3, 3)


def test_no_duplicates_all_behaviors():
    """Test that all behaviors work correctly when no duplicates exist."""
    # Create unique time series
    ts1 = TimeSeries(np.sin(np.linspace(0, 4 * np.pi, 100)), "ts1")
    ts2 = TimeSeries(np.cos(np.linspace(0, 4 * np.pi, 100)), "ts2")
    ts3 = TimeSeries(np.sin(np.linspace(0, 2 * np.pi, 100)), "ts3")

    # Test bunches with no duplicates
    ts_bunch1 = [ts1]
    ts_bunch2 = [ts2, ts3]

    # All behaviors should work identically
    for behavior in ["ignore", "warn", "raise"]:
        stats, significance, info = compute_me_stats(
            ts_bunch1,
            ts_bunch2,
            names1=["ts1"],
            names2=["ts2", "ts3"],
            metric="mi",
            mode="stage1",
            n_shuffles_stage1=10,
            duplicate_behavior=behavior,
            verbose=False,
        )

        assert isinstance(stats, dict)
        assert len(stats) == 1  # ts1
        assert len(stats["ts1"]) == 2  # compared with ts2 and ts3
