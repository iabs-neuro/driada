"""Fixtures for INTENSE test suite.

This module provides optimized fixtures for INTENSE tests to improve performance
by caching expensive data generation operations and providing standard test datasets.

Key optimizations:
1. Session-scoped fixtures for base data (generated once per test session)
2. Function-scoped fixtures for modified data (copies to avoid mutation)
3. Cached multivariate normal generation
4. Pre-computed correlation patterns
"""

import pytest
import numpy as np
from driada.information.info_base import TimeSeries, MultiTimeSeries
from driada.utils.data import create_correlated_gaussian_data


# Cache for expensive computations
_noise_cache = {}


def _get_cached_noise(n, T, seed=42):
    """Get cached multivariate normal noise to speed up tests."""
    key = (n, T, seed)
    if key not in _noise_cache:
        np.random.seed(seed)
        _noise_cache[key] = np.random.multivariate_normal(
            np.zeros(n), np.eye(n), size=T, check_valid="raise"
        ).T
    return _noise_cache[key].copy()


@pytest.fixture(scope="session")
def base_correlated_signals_small():
    """Small correlated signals (n=10, T=500) for fast tests."""
    n, T = 10, 500
    correlation_pairs = [(1, n - 1, 0.9), (2, n - 2, 0.8), (5, n - 5, 0.7)]
    signals, _ = create_correlated_gaussian_data(
        n_features=n, n_samples=T, correlation_pairs=correlation_pairs, seed=42
    )
    return signals, n, T


@pytest.fixture(scope="session")
def base_correlated_signals_medium():
    """Medium correlated signals (n=20, T=1000) for standard tests."""
    n, T = 20, 1000
    correlation_pairs = [(1, n - 1, 0.9), (2, n - 2, 0.8), (5, n - 5, 0.7)]
    signals, _ = create_correlated_gaussian_data(
        n_features=n, n_samples=T, correlation_pairs=correlation_pairs, seed=42
    )
    return signals, n, T


@pytest.fixture(scope="session")
def base_correlated_signals_large():
    """Large correlated signals (n=30, T=3000) for thorough tests."""
    n, T = 30, 3000
    correlation_pairs = [(1, n - 1, 0.9), (2, n - 2, 0.8), (5, n - 5, 0.7)]
    signals, _ = create_correlated_gaussian_data(
        n_features=n, n_samples=T, correlation_pairs=correlation_pairs, seed=42
    )
    return signals, n, T


def _create_windowed_signals(
    signals, n, T, w=100, noise_scale=0.02
):  # Reduced from 0.2
    """Apply windowing and noise to signals."""
    # Create windowed version
    np.random.seed(42)
    starts = np.random.choice(np.arange(w, T - w), size=10)
    nnz_time_inds = []
    for st in starts:
        nnz_time_inds.extend([st + _ for _ in range(w)])

    cropped_signals = np.zeros((n, T))
    cropped_signals[:, np.array(nnz_time_inds)] = signals[:, np.array(nnz_time_inds)]

    # Add noise
    small_noise = _get_cached_noise(n, T, seed=42) * noise_scale
    cropped_signals += small_noise

    return cropped_signals


def _binarize_ts(ts, thr="av"):
    """Helper to binarize a TimeSeries."""
    if not ts.discrete:
        if thr == "av":
            thr = np.mean(ts.data)
        bin_data = np.zeros(len(ts.data))
        bin_data[ts.data >= thr] = 1
    else:
        raise ValueError("binarize_ts called on discrete TimeSeries")

    return TimeSeries(bin_data, discrete=True)


@pytest.fixture
def correlated_ts_small(base_correlated_signals_small):
    """Small correlated time series lists for fast tests."""
    signals, n, T = base_correlated_signals_small
    cropped_signals = _create_windowed_signals(signals, n, T)

    # Create time series lists
    tslist1 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[: n // 2, :]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[n // 2 :, :]]

    # Set shuffle masks
    for ts in tslist1 + tslist2:
        ts.shuffle_mask[:50] = 0

    return tslist1, tslist2, n


@pytest.fixture
def correlated_ts_medium(base_correlated_signals_medium):
    """Medium correlated time series lists for standard tests."""
    signals, n, T = base_correlated_signals_medium
    cropped_signals = _create_windowed_signals(signals, n, T)

    # Create time series lists
    tslist1 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[: n // 2, :]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[n // 2 :, :]]

    # Set shuffle masks
    for ts in tslist1 + tslist2:
        ts.shuffle_mask[:50] = 0

    return tslist1, tslist2, n


@pytest.fixture
def correlated_ts_binarized(base_correlated_signals_medium):
    """Correlated time series with second half binarized."""
    signals, n, T = base_correlated_signals_medium
    cropped_signals = _create_windowed_signals(signals, n, T, noise_scale=0.01)

    # Create time series lists
    tslist1 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[: n // 2, :]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[n // 2 :, :]]

    # Binarize second half
    tslist2 = [_binarize_ts(ts, "av") for ts in tslist2]

    # Set shuffle masks
    for ts in tslist1 + tslist2:
        ts.shuffle_mask[:50] = 0

    return tslist1, tslist2, n


@pytest.fixture
def aggregate_two_ts_func():
    """Function to aggregate two time series into MultiTimeSeries."""

    def aggregate(ts1, ts2):
        # Add small noise to break degeneracy
        np.random.seed(42)
        mod_lts1 = TimeSeries(
            ts1.data + np.random.random(size=len(ts1.data)) * 1e-6, discrete=False
        )
        mod_lts2 = TimeSeries(
            ts2.data + np.random.random(size=len(ts2.data)) * 1e-6, discrete=False
        )
        mts = MultiTimeSeries([mod_lts1, mod_lts2])
        return mts

    return aggregate


@pytest.fixture
def fast_test_params():
    """Optimized parameters for fast test execution."""
    return {
        "n_shuffles_stage1": 20,
        "n_shuffles_stage2": 100,
        "ds": 5,
        "noise_ampl": 1e-4,  # Reduced noise
        "enable_parallelization": False,  # Disabled for faster single tests
        "verbose": False,
    }


@pytest.fixture
def balanced_test_params():
    """Balanced parameters for accuracy vs speed."""
    return {
        "n_shuffles_stage1": 50,
        "n_shuffles_stage2": 500,
        "ds": 5,
        "noise_ampl": 1e-4,  # Reduced noise
        "enable_parallelization": False,  # Disabled for faster single tests
        "verbose": False,
    }


@pytest.fixture
def strict_test_params():
    """Strict parameters to reduce false positives."""
    return {
        "n_shuffles_stage1": 100,
        "n_shuffles_stage2": 1000,
        "ds": 5,  # Changed from 2 to 5
        "noise_ampl": 1e-4,  # Reduced noise
        "multicomp_correction": "holm",
        "pval_thr": 0.001,
        "enable_parallelization": False,  # Disabled for faster single tests
        "verbose": False,
    }


@pytest.fixture(params=[(10, 500, 1), (20, 1000, 2), (30, 2000, 3)])
def scaled_correlated_ts(request):
    """Parametrized fixture for correlation detection scaling tests."""
    n, T, expected_pairs = request.param

    # Generate correlated signals
    correlation_pairs = [(1, n - 1, 0.9), (2, n - 2, 0.8), (5, n - 5, 0.7)]
    signals, _ = create_correlated_gaussian_data(
        n_features=n, n_samples=T, correlation_pairs=correlation_pairs, seed=42
    )

    # Apply windowing
    cropped_signals = _create_windowed_signals(signals, n, T)

    # Create time series lists
    tslist1 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[: n // 2, :]]
    tslist2 = [TimeSeries(sig, discrete=False) for sig in cropped_signals[n // 2 :, :]]

    # Set shuffle masks
    for ts in tslist1 + tslist2:
        ts.shuffle_mask[:50] = 0

    return tslist1, tslist2, n, T, expected_pairs
