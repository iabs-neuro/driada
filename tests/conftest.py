"""Configuration for tests.

This module provides shared fixtures and utilities for the DRIADA test suite.
It leverages existing data generation functions from the main codebase to avoid
duplication and ensure consistency.

Experiment Fixtures Usage:
-------------------------
1. Basic fixtures (fixed size):
   - small_experiment: 5 neurons, 10s, mixed features
   - medium_experiment: 20 neurons, 60s, mixed features
   - large_experiment: 50 neurons, 300s, mixed features

2. Parametrized fixtures (3 sizes):
   - continuous_only_experiment: Only continuous features
   - discrete_only_experiment: Only discrete features
   - mixed_features_experiment: Both feature types

   Use with: @pytest.mark.parametrize("fixture_name", ["small"], indirect=True)
   Or test all sizes: (default behavior if not parametrized)

Example:
    def test_my_function(small_experiment):
        # Uses small fixed-size experiment

    @pytest.mark.parametrize("continuous_only_experiment", ["medium"], indirect=True)
    def test_continuous(continuous_only_experiment):
        # Uses medium-sized continuous-only experiment
"""

import os
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
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
    from driada.experiment.synthetic import generate_synthetic_exp

    return generate_synthetic_exp(
        n_dfeats=2,
        n_cfeats=2,
        nneurons=5,
        duration=60,  # 60 seconds for sufficient shuffling
        fps=20,
        seed=42,
    )


@pytest.fixture
def medium_experiment():
    """Medium-sized test experiment.

    Suitable for integration tests.
    """
    from driada.experiment.synthetic import generate_synthetic_exp

    return generate_synthetic_exp(
        n_dfeats=3,
        n_cfeats=3,
        nneurons=20,
        duration=200,  # 200 seconds for thorough testing
        fps=20,
        seed=42,
    )


@pytest.fixture
def large_experiment():
    """Large test experiment.

    Suitable for performance tests and thorough validation.
    """
    from driada.experiment.synthetic import generate_synthetic_exp

    return generate_synthetic_exp(
        n_dfeats=5,
        n_cfeats=5,
        nneurons=50,
        duration=1000,  # 1000 seconds for extensive testing
        fps=20,
        seed=42,
    )


# Specialized fixtures for different feature types
@pytest.fixture(params=["small", "medium", "large"])
def continuous_only_experiment(request):
    """Experiment with only continuous features in 3 sizes."""
    from driada.experiment.synthetic import generate_synthetic_exp

    sizes = {
        "small": (0, 2, 5, 60),  # n_dfeats, n_cfeats, nneurons, duration
        "medium": (0, 3, 10, 200),
        "large": (0, 5, 30, 1000),
    }
    n_dfeats, n_cfeats, nneurons, duration = sizes[request.param]
    return generate_synthetic_exp(
        n_dfeats=n_dfeats,
        n_cfeats=n_cfeats,
        nneurons=nneurons,
        duration=duration,
        fps=20,
        seed=42,
    )


@pytest.fixture(params=["small", "medium", "large"])
def discrete_only_experiment(request):
    """Experiment with only discrete features in 3 sizes."""
    from driada.experiment.synthetic import generate_synthetic_exp

    sizes = {
        "small": (2, 0, 5, 60),  # n_dfeats, n_cfeats, nneurons, duration
        "medium": (3, 0, 10, 200),
        "large": (5, 0, 30, 1000),
    }
    n_dfeats, n_cfeats, nneurons, duration = sizes[request.param]
    return generate_synthetic_exp(
        n_dfeats=n_dfeats,
        n_cfeats=n_cfeats,
        nneurons=nneurons,
        duration=duration,
        fps=20,
        seed=42,
    )


@pytest.fixture(params=["small", "medium", "large"])
def mixed_features_experiment(request):
    """Experiment with mixed discrete and continuous features in 3 sizes."""
    from driada.experiment.synthetic import generate_synthetic_exp

    sizes = {
        "small": (2, 2, 5, 60),  # n_dfeats, n_cfeats, nneurons, duration
        "medium": (3, 3, 20, 200),
        "large": (5, 5, 50, 1000),
    }
    n_dfeats, n_cfeats, nneurons, duration = sizes[request.param]
    return generate_synthetic_exp(
        n_dfeats=n_dfeats,
        n_cfeats=n_cfeats,
        nneurons=nneurons,
        duration=duration,
        fps=20,
        seed=42,
    )


@pytest.fixture(params=["small", "medium", "large"])
def multifeature_experiment(request):
    """Experiment with at least 4 continuous features for multifeature tests.

    This fixture ensures there are always enough features for tests that require
    multiple feature pairs (e.g., place from x,y and locomotion from speed,head_direction).
    """
    from driada.experiment.synthetic import generate_synthetic_exp

    sizes = {
        "small": (0, 4, 10, 100),  # n_dfeats, n_cfeats, nneurons, duration
        "medium": (0, 6, 20, 200),  # More features for complex tests
        "large": (0, 8, 30, 500),  # Even more features
    }
    n_dfeats, n_cfeats, nneurons, duration = sizes[request.param]
    return generate_synthetic_exp(
        n_dfeats=n_dfeats,
        n_cfeats=n_cfeats,
        nneurons=nneurons,
        duration=duration,
        fps=20,
        seed=42,
    )


@pytest.fixture
def experiment_factory():
    """Factory fixture for creating experiments with custom parameters.

    This fixture provides a function that creates experiments with specific
    configurations while still benefiting from pytest's fixture management.

    Usage in tests:
        def test_something(experiment_factory):
            # Create experiment with specific parameters
            exp = experiment_factory(n_dfeats=5, n_cfeats=0, nneurons=2)

            # Or with additional parameters
            exp = experiment_factory(
                n_dfeats=1,
                n_cfeats=0,
                nneurons=3,
                with_spikes=True,
                fps=30
            )

    Returns
    -------
    factory : callable
        Function that creates experiments with given parameters.
        Accepts all parameters of generate_synthetic_exp.
    """
    from driada.experiment.synthetic import generate_synthetic_exp

    def factory(**kwargs):
        # Set default parameters that can be overridden
        defaults = {
            "n_dfeats": 2,
            "n_cfeats": 2,
            "nneurons": 10,
            "duration": 200,  # Default to medium duration
            "fps": 20,
            "seed": 42,
        }
        # Update defaults with provided kwargs
        params = {**defaults, **kwargs}
        return generate_synthetic_exp(**params)

    return factory


@pytest.fixture(scope="session")
def spike_reconstruction_experiment():
    """Fixture for experiment with spike reconstruction enabled.

    This fixture creates a standard experiment configuration with spike
    reconstruction from calcium signals. Used for testing spike-related
    functionality.

    Returns
    -------
    exp : Experiment
        Experiment with n_dfeats=1, n_cfeats=0, nneurons=5, with_spikes=True
    """
    from driada.experiment.synthetic import generate_synthetic_exp

    return generate_synthetic_exp(
        n_dfeats=1,
        n_cfeats=0,
        nneurons=5,
        duration=200,
        fps=20,
        seed=42,
        with_spikes=True,  # Enable spike reconstruction
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
    from sklearn.datasets import make_swiss_roll

    data, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
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
    from sklearn.datasets import make_s_curve

    data, color = make_s_curve(n_samples=1000, noise=0.1, random_state=42)
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
    from driada.utils.data import create_correlated_gaussian_data

    return create_correlated_gaussian_data(
        n_features=10, n_samples=1000, correlation_pairs=correlation_pattern, seed=42
    )


@pytest.fixture
def simple_timeseries():
    """Create a simple TimeSeries for unit tests."""
    import numpy as np
    from driada.information.info_base import TimeSeries

    values = np.sin(np.linspace(0, 4 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
    return TimeSeries(name="test_sine", values=values, start_time=0, fps=20)


@pytest.fixture
def multi_timeseries():
    """Create a MultiTimeSeries for unit tests."""
    import numpy as np
    from driada.information.info_base import TimeSeries, MultiTimeSeries

    n_features = 3
    n_samples = 1000

    ts_list = []
    for i in range(n_features):
        values = np.random.randn(n_samples) + i * 0.5
        ts = TimeSeries(name=f"feature_{i}", values=values, start_time=0, fps=20)
        ts_list.append(ts)

    return MultiTimeSeries(name="test_multi_ts", ts_list=ts_list)


@pytest.fixture
def circular_manifold_data():
    """Generate circular manifold data for testing."""
    from driada.experiment.synthetic import generate_circular_manifold_data

    data = generate_circular_manifold_data(
        n_samples=1000, n_neurons=20, noise_level=0.1, seed=42
    )
    return data


@pytest.fixture
def spatial_2d_data():
    """Generate 2D spatial manifold data for testing."""
    from driada.experiment.synthetic import generate_2d_manifold_data

    data = generate_2d_manifold_data(
        grid_size=20, n_neurons=50, noise_level=0.1, seed=42
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
