"""
Tests for signal generation and analysis utilities.
"""

import pytest
import numpy as np
from driada.utils.signals import brownian, approximate_entropy
from driada.information import TimeSeries


def test_brownian_single_path():
    """Test single Brownian motion path generation."""
    n_steps = 1000
    x0 = 0.0

    path = brownian(x0, n_steps)

    assert path.shape == (n_steps,)
    assert isinstance(path, np.ndarray)
    # First value should be close to x0 (not exactly due to first step)
    assert abs(path[0] - x0) < 5  # Within reasonable range


def test_brownian_multiple_paths():
    """Test multiple Brownian motion paths."""
    n_steps = 500
    x0 = [0.0, 1.0, -1.0]

    paths = brownian(x0, n_steps)

    assert paths.shape == (3, n_steps)
    # Each path should start near its initial condition
    for i in range(3):
        assert abs(paths[i, 0] - x0[i]) < 5


def test_brownian_with_parameters():
    """Test Brownian motion with custom parameters."""
    n_steps = 1000  # More steps for better statistics
    x0 = 0.0
    dt = 0.01
    delta = 2.0

    # Generate multiple paths for better statistical test
    n_paths = 100
    paths = []
    for _ in range(n_paths):
        path = brownian(x0, n_steps, dt=dt, delta=delta)
        paths.append(path)

    paths = np.array(paths)

    # Check shape
    assert paths.shape == (n_paths, n_steps)

    # Check that variance grows approximately as delta^2 * t
    # For Brownian motion, Var[X(t)] = delta^2 * t
    t_final = n_steps * dt
    expected_std = delta * np.sqrt(t_final)

    # Calculate actual standard deviation at final time
    final_values = paths[:, -1]
    actual_std = np.std(final_values)

    # Should be within reasonable range (allowing for statistical variation)
    assert 0.7 * expected_std < actual_std < 1.3 * expected_std


def test_approximate_entropy_regular_signal():
    """Test ApEn for regular/predictable signal."""
    # Perfectly regular signal
    regular = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]  # More repetitions

    apen = approximate_entropy(regular, m=2, r=0.1)

    # Very low entropy for regular signal (not exactly 0 due to finite length)
    assert apen < 0.1


def test_approximate_entropy_random_signal():
    """Test ApEn for random signal."""
    np.random.seed(42)
    random_signal = np.random.randn(500)  # Longer signal for better statistics

    apen = approximate_entropy(random_signal, m=2, r=0.2 * np.std(random_signal))

    # Random signal should have moderate to high entropy
    assert apen > 0.5  # Lower threshold since ApEn can vary
    assert apen < 3.0  # But not infinite


def test_approximate_entropy_sine_wave():
    """Test ApEn for periodic signal."""
    t = np.linspace(0, 10 * np.pi, 500)
    sine_wave = np.sin(t)

    apen = approximate_entropy(sine_wave, m=2, r=0.1)

    # Periodic signal should have low entropy
    assert apen < 0.5


def test_timeseries_approximate_entropy_continuous():
    """Test ApEn method for continuous TimeSeries."""
    np.random.seed(42)
    data = np.random.randn(200)
    ts = TimeSeries(data, discrete=False)

    # With default parameters
    apen1 = ts.approximate_entropy()
    assert apen1 > 0

    # With custom parameters
    apen2 = ts.approximate_entropy(m=3, r=0.3)
    assert apen2 > 0
    assert apen1 != apen2  # Different parameters should give different results


def test_timeseries_approximate_entropy_discrete_raises():
    """Test that ApEn raises error for discrete TimeSeries."""
    data = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    ts = TimeSeries(data, discrete=True)

    with pytest.raises(
        ValueError, match="approximate_entropy is only valid for continuous"
    ):
        ts.approximate_entropy()


# Signal class deprecation test removed since signals module is deleted


def test_filter_1d_timeseries_gaussian():
    """Test Gaussian filtering of 1D time series."""
    # Create noisy signal
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * t)
    noisy_signal = clean_signal + 0.2 * np.random.randn(len(t))

    from driada.utils.signals import filter_1d_timeseries

    # Apply Gaussian filter
    filtered = filter_1d_timeseries(noisy_signal, method="gaussian", sigma=2.0)

    assert len(filtered) == len(noisy_signal)
    # Should reduce noise
    assert np.std(filtered - clean_signal) < np.std(noisy_signal - clean_signal)


def test_filter_1d_timeseries_savgol():
    """Test Savitzky-Golay filtering."""
    # Create signal with sharp peak
    t = np.linspace(0, 10, 1000)
    signal = np.exp(-((t - 5) ** 2))
    noisy_signal = signal + 0.05 * np.random.randn(len(t))

    from driada.utils.signals import filter_1d_timeseries

    # Apply Savitzky-Golay filter
    filtered = filter_1d_timeseries(
        noisy_signal, method="savgol", window_length=11, polyorder=3
    )

    assert len(filtered) == len(noisy_signal)
    # Should preserve peak better than Gaussian
    peak_idx = np.argmax(signal)
    assert abs(filtered[peak_idx] - signal[peak_idx]) < 0.1


def test_filter_signals_2d():
    """Test filtering of 2D array (multiple signals)."""
    from driada.utils import filter_signals

    # Create multiple noisy signals
    n_signals = 5
    n_samples = 500
    signals = np.random.randn(n_signals, n_samples)

    # Apply filtering
    filtered = filter_signals(signals, method="gaussian", sigma=1.5)

    assert filtered.shape == signals.shape
    # Each signal should be smoother
    for i in range(n_signals):
        assert np.std(np.diff(filtered[i])) < np.std(np.diff(signals[i]))


def test_adaptive_filter_signals():
    """Test adaptive filtering based on SNR."""
    from driada.utils import adaptive_filter_signals

    # Create signals with different noise levels
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)

    # High SNR signal
    high_snr = np.sin(2 * np.pi * t) + 0.05 * np.random.randn(n_samples)
    # Low SNR signal
    low_snr = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(n_samples)

    signals = np.vstack([high_snr, low_snr])

    # Apply adaptive filtering
    filtered = adaptive_filter_signals(signals, snr_threshold=2.0)

    assert filtered.shape == signals.shape
    # Low SNR signal should be filtered more strongly
    high_snr_change = np.std(filtered[0] - signals[0])
    low_snr_change = np.std(filtered[1] - signals[1])
    assert low_snr_change > high_snr_change
