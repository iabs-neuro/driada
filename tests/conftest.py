"""Configuration for tests."""

import os
import pytest


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
        }