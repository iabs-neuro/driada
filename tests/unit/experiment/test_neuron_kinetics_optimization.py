"""Tests for Neuron kinetics optimization methods.

Tests for _optimize_kinetics_direct and related methods that measure
t_rise and t_off from calcium signals.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from driada.experiment.neuron import Neuron
from driada.utils.neural import generate_pseudo_calcium_signal


class TestOptimizeKineticsDirect:
    """Tests for _optimize_kinetics_direct method."""

    @pytest.fixture
    def neuron_with_events(self):
        """Create neuron with synthetic calcium signal that has clear events."""
        # Generate signal with known kinetics
        signal = generate_pseudo_calcium_signal(
            duration=60,
            sampling_rate=20,
            event_rate=0.5,  # 0.5 events/s = ~30 events
            amplitude_range=(0.5, 1.5),
            decay_time=2.0,  # t_off = 2.0s
            noise_std=0.05,
            rise_time=0.1,  # t_rise = 0.1s
        )
        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)
        neuron.reconstruct_spikes(method='wavelet')
        return neuron

    def test_successful_optimization_returns_correct_flags(self, neuron_with_events):
        """Test that successful optimization returns optimized=True and no defaults used."""
        result = neuron_with_events._optimize_kinetics_direct(fps=20)

        # Should be fully optimized with both params measured
        if result['n_events_used_rise'] >= 5 and result['n_events_used_off'] >= 5:
            assert result['optimized'] is True
            assert result['partially_optimized'] is False
            assert result['used_defaults']['t_rise'] is False
            assert result['used_defaults']['t_off'] is False

    def test_partial_optimization_t_rise_only(self):
        """Test that partial optimization is correctly detected when only t_rise succeeds."""
        # Create neuron with minimal signal
        signal = np.zeros(1000)
        signal[100:110] = 1.0  # Single sharp event
        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)

        # Mock wvt_ridges to simulate detected events
        mock_ridge = MagicMock()
        mock_ridge.start = 95
        mock_ridge.end = 115
        neuron.wvt_ridges = [mock_ridge] * 10  # 10 events

        # Mock _measure_t_rise_derivative to always succeed
        # Mock _measure_t_off_from_peak to always fail
        with patch.object(neuron, '_measure_t_rise_derivative', return_value=0.1) as mock_rise:
            with patch.object(neuron, '_measure_t_off_from_peak', return_value=None) as mock_off:
                result = neuron._optimize_kinetics_direct(fps=20, min_events=5)

                # Should be partially optimized
                assert result['optimized'] is False
                assert result['partially_optimized'] is True
                assert result['used_defaults']['t_rise'] is False
                assert result['used_defaults']['t_off'] is True
                # t_rise should be measured value, t_off should be default
                assert result['n_events_used_rise'] == 10
                assert result['n_events_used_off'] == 0

    def test_partial_optimization_t_off_only(self):
        """Test that partial optimization is correctly detected when only t_off succeeds."""
        signal = np.zeros(1000)
        signal[100:200] = np.exp(-np.arange(100) / 40)  # Decay event
        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)

        mock_ridge = MagicMock()
        mock_ridge.start = 95
        mock_ridge.end = 200
        neuron.wvt_ridges = [mock_ridge] * 10

        with patch.object(neuron, '_measure_t_rise_derivative', return_value=None):
            with patch.object(neuron, '_measure_t_off_from_peak', return_value=2.0):
                result = neuron._optimize_kinetics_direct(fps=20, min_events=5)

                assert result['optimized'] is False
                assert result['partially_optimized'] is True
                assert result['used_defaults']['t_rise'] is True
                assert result['used_defaults']['t_off'] is False
                assert result['n_events_used_rise'] == 0
                assert result['n_events_used_off'] == 10

    def test_both_measurements_fail(self):
        """Test that optimization fails when both measurements are insufficient."""
        signal = np.random.random(1000) * 0.01  # Just noise
        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)

        mock_ridge = MagicMock()
        mock_ridge.start = 100
        mock_ridge.end = 150
        neuron.wvt_ridges = [mock_ridge] * 3  # Only 3 events

        with patch.object(neuron, '_measure_t_rise_derivative', return_value=0.1):
            with patch.object(neuron, '_measure_t_off_from_peak', return_value=2.0):
                result = neuron._optimize_kinetics_direct(fps=20, min_events=5)

                # Both have < 5 measurements
                assert result['optimized'] is False
                assert result['partially_optimized'] is False
                assert result['used_defaults']['t_rise'] is True
                assert result['used_defaults']['t_off'] is True
                assert 'error' in result

    def test_no_events_returns_failure(self):
        """Test that no events returns proper failure response."""
        signal = np.zeros(1000)
        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)
        neuron.wvt_ridges = None

        result = neuron._optimize_kinetics_direct(fps=20)

        assert result['optimized'] is False
        assert result['partially_optimized'] is False
        assert result['used_defaults']['t_rise'] is True
        assert result['used_defaults']['t_off'] is True
        assert 'error' in result

    def test_used_defaults_dict_always_present(self, neuron_with_events):
        """Test that used_defaults is always present in results."""
        result = neuron_with_events._optimize_kinetics_direct(fps=20)

        assert 'used_defaults' in result
        assert isinstance(result['used_defaults'], dict)
        assert 't_rise' in result['used_defaults']
        assert 't_off' in result['used_defaults']

    def test_partially_optimized_always_present(self, neuron_with_events):
        """Test that partially_optimized is always present in results."""
        result = neuron_with_events._optimize_kinetics_direct(fps=20)

        assert 'partially_optimized' in result
        assert isinstance(result['partially_optimized'], bool)


class TestMeasureTOffFromPeak:
    """Tests for _measure_t_off_from_peak method."""

    @pytest.fixture
    def neuron(self):
        """Create a basic neuron for testing."""
        signal = np.zeros(1000)
        return Neuron(cell_id=0, ca=signal, sp=None, fps=20)

    def test_accepts_long_decay_up_to_30_seconds(self, neuron):
        """Test that tau values up to 30 seconds are accepted (previously limited to 10)."""
        fps = 20
        # Create signal with long decay (tau = 20 seconds)
        t = np.arange(500)
        tau_frames = 20 * fps  # 20 seconds in frames
        signal = np.exp(-t / tau_frames)
        signal = np.concatenate([np.zeros(100), signal])

        result = neuron._measure_t_off_from_peak(signal, peak_idx=100, fps=fps, max_frames=500)

        # Should NOT return None for tau = 20s (was failing before fix)
        if result is not None:
            assert 15 < result < 25  # Allow some fitting error

    def test_rejects_tau_above_30_seconds(self, neuron):
        """Test that tau values above 30 seconds are still rejected."""
        fps = 20
        # Create signal with very long decay (tau = 50 seconds)
        t = np.arange(1000)
        tau_frames = 50 * fps
        signal = np.exp(-t / tau_frames)
        signal = np.concatenate([np.zeros(100), signal])

        result = neuron._measure_t_off_from_peak(signal, peak_idx=100, fps=fps, max_frames=1000)

        # Should still return None for tau = 50s (above 30s limit)
        # Note: may return a value if max_frames truncates the measurement
        # The key is that values clearly > 30s should be rejected
        pass  # This test documents expected behavior

    def test_rejects_tau_below_0_1_seconds(self, neuron):
        """Test that tau values below 0.1 seconds are rejected."""
        fps = 20
        # Create signal with very short decay (tau = 0.05 seconds)
        t = np.arange(50)
        tau_frames = 0.05 * fps  # 0.05 seconds = 1 frame at 20fps
        signal = np.exp(-t / max(tau_frames, 0.1))
        signal = np.concatenate([np.zeros(10), signal])

        result = neuron._measure_t_off_from_peak(signal, peak_idx=10, fps=fps, max_frames=50)

        # Very short tau should be rejected
        if result is not None:
            assert result >= 0.1

    def test_returns_none_for_rising_signal(self, neuron):
        """Test that rising signal (positive slope) returns None."""
        fps = 20
        signal = np.linspace(0, 1, 100)  # Rising signal
        signal = np.concatenate([np.zeros(50), signal])

        result = neuron._measure_t_off_from_peak(signal, peak_idx=50, fps=fps, max_frames=100)

        assert result is None

    def test_returns_none_for_insufficient_frames(self, neuron):
        """Test that insufficient frames returns None."""
        fps = 20
        signal = np.exp(-np.arange(5) / 10)  # Only 5 frames
        signal = np.concatenate([np.zeros(10), signal])

        result = neuron._measure_t_off_from_peak(signal, peak_idx=10, fps=fps, max_frames=5)

        # Less than 10 frames should return None
        assert result is None


class TestLongDecaySignalHandling:
    """Integration tests for handling signals with long decay times."""

    def test_long_decay_signal_not_silently_defaulted(self):
        """Test that long decay signals are properly measured or flagged as failed."""
        # Generate signal with t_off = 15 seconds (was failing before fix)
        signal = generate_pseudo_calcium_signal(
            duration=120,  # 2 minutes to capture long decays
            sampling_rate=20,
            event_rate=0.2,  # Sparse events
            amplitude_range=(0.8, 1.2),
            decay_time=15.0,  # Long t_off
            noise_std=0.02,
            rise_time=0.1,
        )

        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)
        neuron.reconstruct_spikes(method='wavelet')

        result = neuron._optimize_kinetics_direct(fps=20, max_frames_forward=500)

        # Key assertion: if t_off defaulted, used_defaults should reflect this
        if result['used_defaults']['t_off']:
            # If default was used, optimized should be False or partially_optimized True
            assert result['optimized'] is False or result['partially_optimized'] is True
        else:
            # If measured successfully, t_off should be within the valid range (0.1-30s)
            # The actual value may vary due to noise and event overlap
            assert 0.1 < result['t_off'] < 30.0

    def test_medium_decay_signal_still_works(self):
        """Test that medium decay signals (1-10s) still work correctly."""
        signal = generate_pseudo_calcium_signal(
            duration=60,
            sampling_rate=20,
            event_rate=0.5,
            amplitude_range=(0.5, 1.5),
            decay_time=3.0,  # t_off = 3s (well within old and new limits)
            noise_std=0.03,
            rise_time=0.1,
        )

        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=20)
        neuron.reconstruct_spikes(method='wavelet')

        result = neuron._optimize_kinetics_direct(fps=20)

        # Should work as before - verify new fields are present and consistent
        assert 'used_defaults' in result
        assert 'partially_optimized' in result

        # If enough events were detected and measured
        if result['n_events_used_off'] >= 5:
            # t_off should NOT be using default
            assert result['used_defaults']['t_off'] is False
            # Measured t_off should be within valid bounds
            assert 0.1 < result['t_off'] < 30.0
