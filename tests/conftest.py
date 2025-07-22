"""Configuration for tests.

This module provides shared fixtures and utilities for the DRIADA test suite.
It leverages existing data generation functions from the main codebase to avoid
duplication and ensure consistency.
"""

import os
import pytest
import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve
from src.driada.utils.data import create_correlated_gaussian_data


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture(scope="session")
def test_mode():
    """Determine if we're in fast test mode based on environment variable."""
    return os.environ.get("INTENSE_FAST_TESTS", "").lower() in ("1", "true", "yes")


@pytest.fixture
def test_params(test_mode):
    """Get test parameters based on test mode.
    
    In fast test mode, use smaller parameters for quicker execution.
    In normal mode, use larger parameters for thorough testing.
    """
    if test_mode:
        # Fast mode - optimized for speed
        return {
            "n_shuffles_stage1": 10,
            "n_shuffles_stage2": 50,
            "n_neurons": 5,
            "n_features": 2,
            "time_series_length": 2000,
            "experiment_duration": 10,
            "enable_parallelization": True,
            "verbose": False,
            "ds": 2,  # Add downsampling for faster computation
        }
    else:
        # Normal mode - thorough testing
        return {
            "n_shuffles_stage1": 100,
            "n_shuffles_stage2": 1000,
            "n_neurons": 20,
            "n_features": 4,
            "time_series_length": 10000,
            "experiment_duration": 60,
            "enable_parallelization": True,
            "verbose": False,
            "ds": 1,  # No downsampling in thorough mode
        }


# Data Generation Fixtures
# These fixtures use existing utilities from the codebase to avoid duplication

@pytest.fixture
def small_experiment():
    """Small test experiment using existing generator.
    
    Suitable for quick unit tests.
    """
    from src.driada.experiment.synthetic import generate_synthetic_exp
    return generate_synthetic_exp(
        n_dfeats=2, 
        n_cfeats=2, 
        nneurons=5, 
        duration=10,  # 10 seconds
        fps=20, 
        seed=42
    )


@pytest.fixture
def medium_experiment():
    """Medium-sized test experiment.
    
    Suitable for integration tests.
    """
    from src.driada.experiment.synthetic import generate_synthetic_exp
    return generate_synthetic_exp(
        n_dfeats=3, 
        n_cfeats=3, 
        nneurons=20, 
        duration=60,  # 1 minute
        fps=20, 
        seed=42
    )


@pytest.fixture
def swiss_roll_data():
    """Standard swiss roll data for manifold tests.
    
    Returns
    -------
    data : ndarray
        Swiss roll data of shape (3, 1000)
    color : ndarray
        Color values for visualization
    """
    data, color = make_swiss_roll(
        n_samples=1000, 
        noise=0.1, 
        random_state=42
    )
    return data.T, color


@pytest.fixture
def s_curve_data():
    """Standard S-curve data for manifold tests.
    
    Returns
    -------
    data : ndarray
        S-curve data of shape (3, 1000)
    color : ndarray
        Color values for visualization
    """
    data, color = make_s_curve(
        n_samples=1000, 
        noise=0.1, 
        random_state=42
    )
    return data.T, color


@pytest.fixture
def correlation_pattern():
    """Standard correlation pattern used in multiple tests.
    
    Returns a list of (i, j, correlation) tuples.
    """
    return [(1, 9, 0.9), (2, 8, 0.8), (3, 7, 0.7), (4, 6, 0.6)]


@pytest.fixture
def correlated_gaussian_data(correlation_pattern):
    """Generate correlated Gaussian data using standard pattern.
    
    Uses the correlation pattern fixture and the utility function
    from driada.utils.data to create consistent test data.
    """
    return create_correlated_gaussian_data(
        n_features=10,
        n_samples=1000,
        correlation_pairs=correlation_pattern,
        seed=42
    )


@pytest.fixture
def simple_timeseries():
    """Create a simple TimeSeries for unit tests."""
    from src.driada.information.info_base import TimeSeries
    
    values = np.sin(np.linspace(0, 4*np.pi, 1000)) + 0.1 * np.random.randn(1000)
    return TimeSeries(
        name="test_sine",
        values=values,
        start_time=0,
        fps=20
    )


@pytest.fixture
def multi_timeseries():
    """Create a MultiTimeSeries for unit tests."""
    from src.driada.information.info_base import TimeSeries, MultiTimeSeries
    
    n_features = 3
    n_samples = 1000
    
    ts_list = []
    for i in range(n_features):
        values = np.random.randn(n_samples) + i * 0.5
        ts = TimeSeries(
            name=f"feature_{i}",
            values=values,
            start_time=0,
            fps=20
        )
        ts_list.append(ts)
    
    return MultiTimeSeries(
        name="test_multi_ts",
        ts_list=ts_list
    )


@pytest.fixture
def circular_manifold_data():
    """Generate circular manifold data for testing."""
    from src.driada.experiment.synthetic import generate_circular_manifold_data
    
    data = generate_circular_manifold_data(
        n_samples=1000,
        n_neurons=20,
        noise_level=0.1,
        seed=42
    )
    return data


@pytest.fixture
def spatial_2d_data():
    """Generate 2D spatial manifold data for testing."""
    from src.driada.experiment.synthetic import generate_2d_manifold_data
    
    data = generate_2d_manifold_data(
        grid_size=20,
        n_neurons=50,
        noise_level=0.1,
        seed=42
    )
    return data


# Test parameter sets for parametrized tests
SMALL_TEST_SIZES = [10, 50, 100]
MEDIUM_TEST_SIZES = [100, 500, 1000]
LARGE_TEST_SIZES = [1000, 5000, 10000]

# Common random seeds for reproducibility
TEST_SEEDS = [42, 123, 456, 789]

# Standard noise levels
NOISE_LEVELS = [0.0, 0.1, 0.5, 1.0]