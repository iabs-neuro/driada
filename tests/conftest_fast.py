"""
Optimized test configuration for fast test execution.
"""

import os
import pytest
import numpy as np
from functools import lru_cache


@pytest.fixture(scope="session")
def fast_test_mode():
    """Enable fast test mode with reduced computational load."""
    return os.environ.get("INTENSE_FAST_TESTS", "1").lower() in ("1", "true", "yes")


@pytest.fixture
def fast_test_params(fast_test_mode):
    """Optimized test parameters for speed.

    Key optimizations:
    - Reduced shuffles: 10 for stage1, 50 for stage2 (from 100/1000)
    - Smaller data: 1000 timepoints (from 10000)
    - Fewer neurons/features for synthetic data
    - Parallel execution enabled
    - No verbose output
    """
    if fast_test_mode:
        return {
            # Minimal viable shuffles for statistical validity
            "n_shuffles_stage1": 10,
            "n_shuffles_stage2": 50,
            # Reduced data sizes
            "n_neurons": 3,
            "n_features": 2,
            "time_series_length": 1000,
            "experiment_duration": 5,
            # Performance settings
            "enable_parallelization": True,
            "verbose": False,
            "ds": 2,  # Downsampling for faster computation
            # Test-specific overrides
            "correlated_ts_length": 2000,  # Reduced from 10000
            "slow_test_threshold": 5.0,  # Tests over 5s are slow
        }
    else:
        # Full test parameters for thorough testing
        return {
            "n_shuffles_stage1": 100,
            "n_shuffles_stage2": 1000,
            "n_neurons": 20,
            "n_features": 4,
            "time_series_length": 10000,
            "experiment_duration": 60,
            "enable_parallelization": True,
            "verbose": False,
            "ds": 1,
            "correlated_ts_length": 10000,
            "slow_test_threshold": 10.0,
        }


@pytest.fixture(scope="session")
def cached_test_data():
    """Cache commonly used test data across sessions."""
    cache = {}

    @lru_cache(maxsize=32)
    def get_correlated_signals(n, T, seed=42):
        """Generate and cache correlated signals."""
        key = f"corr_{n}_{T}_{seed}"
        if key not in cache:
            np.random.seed(seed)
            C = np.eye(n)
            # Add specific correlations
            if n > 1:
                C[1, n - 1] = C[n - 1, 1] = 0.9
            if n > 2:
                C[2, n - 2] = C[n - 2, 2] = 0.8
            if n > 5:
                C[5, n - 5] = C[n - 5, 5] = 0.7

            signals = np.random.multivariate_normal(
                np.zeros(n), C, size=T, check_valid="raise"
            ).T
            cache[key] = signals
        return cache[key]

    return get_correlated_signals


@pytest.fixture
def numba_precompile():
    """Precompile numba functions to avoid JIT overhead in tests."""
    from driada.information.gcmi import ent_g, mi_gg, mi_model_gd, demean
    from driada.information.info_utils import py_fast_digamma_arr

    # Trigger compilation with small data
    dummy_data = np.random.randn(2, 10)
    dummy_discrete = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Compile all JIT functions
    _ = demean(dummy_data)
    _ = ent_g(dummy_data)
    _ = mi_gg(dummy_data, dummy_data)
    _ = mi_model_gd(dummy_data, dummy_discrete, 2)
    _ = py_fast_digamma_arr(np.array([1.0, 2.0, 3.0]))

    return True


@pytest.fixture(autouse=True)
def setup_test_environment(numba_precompile):
    """Automatically setup test environment."""
    # Ensure numba functions are precompiled
    assert numba_precompile

    # Set numpy random seed for reproducibility
    np.random.seed(42)

    # Disable warnings during tests
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    yield

    # Cleanup if needed
    warnings.resetwarnings()


def pytest_configure(config):
    """Configure pytest with optimized settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "performance: marks tests that measure performance"
    )

    # Set parallel execution by default
    if not hasattr(config.option, "numprocesses"):
        config.option.numprocesses = "auto"
