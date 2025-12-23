"""Tests for wavelet-based event detection functions.

These tests verify the correctness of wavelet event detection -
functionality that users rely on for spike/event detection.
"""

import pytest
import numpy as np

from driada.experiment.wavelet_event_detection import (
    get_adaptive_wavelet_scales,
    events_to_ts_array,
    extract_wvt_events,
    get_cwt_ridges,
    events_from_trace,
    MIN_EVENT_DUR,
    MAX_EVENT_DUR,
)


class TestGetAdaptiveWaveletScales:
    """Test FPS-adaptive wavelet scale computation."""

    def test_physical_time_coverage_invariance(self):
        """Different FPS should cover same physical time range."""
        scales_20hz = get_adaptive_wavelet_scales(fps=20)
        scales_30hz = get_adaptive_wavelet_scales(fps=30)

        # Convert scales back to time: t = scale / fps
        # First scale: ~0.25 seconds, last scale: ~2.5 seconds
        time_min_20 = scales_20hz[0] / 20
        time_max_20 = scales_20hz[-1] / 20
        time_min_30 = scales_30hz[0] / 30
        time_max_30 = scales_30hz[-1] / 30

        # Both should cover approximately the same time range
        assert time_min_20 == pytest.approx(time_min_30, rel=0.01)
        assert time_max_20 == pytest.approx(time_max_30, rel=0.01)

    def test_scale_proportional_to_fps(self):
        """Scales should scale proportionally with FPS."""
        scales_20hz = get_adaptive_wavelet_scales(fps=20)
        scales_40hz = get_adaptive_wavelet_scales(fps=40)

        # At 2x FPS, scales should be 2x larger
        assert scales_40hz[0] == pytest.approx(scales_20hz[0] * 2, rel=0.01)
        assert scales_40hz[-1] == pytest.approx(scales_20hz[-1] * 2, rel=0.01)

    def test_n_scales_correct(self):
        """Should return requested number of scales."""
        for n_scales in [10, 50, 100]:
            scales = get_adaptive_wavelet_scales(fps=20, n_scales=n_scales)
            assert len(scales) == n_scales

    def test_scales_logarithmically_spaced(self):
        """Scales should be logarithmically spaced."""
        scales = get_adaptive_wavelet_scales(fps=20)

        # Check log spacing: ratio between consecutive scales should be constant
        log_scales = np.log(scales)
        log_diffs = np.diff(log_scales)

        # All differences should be approximately equal
        assert np.allclose(log_diffs, log_diffs[0], rtol=1e-10)

    def test_custom_time_range(self):
        """Should respect custom min/max time parameters."""
        custom_min = 0.5  # seconds
        custom_max = 5.0  # seconds
        fps = 20

        scales = get_adaptive_wavelet_scales(
            fps=fps, min_time_sec=custom_min, max_time_sec=custom_max
        )

        # First scale should correspond to custom_min
        expected_min_scale = custom_min * fps
        assert scales[0] == pytest.approx(expected_min_scale, rel=0.01)

        # Last scale should correspond to custom_max
        expected_max_scale = custom_max * fps
        assert scales[-1] == pytest.approx(expected_max_scale, rel=0.01)


class TestEventsToTsArray:
    """Test conversion of events to binary time series."""

    def test_single_event_marked_correctly(self):
        """Single event should create continuous 1s in output."""
        length = 100
        fps = 20.0
        # Event from frame 30 to 50 (1 second = 20 frames)
        st_ev_inds = [[30]]
        end_ev_inds = [[50]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        assert result.shape == (1, length)
        # Event region should have 1s
        assert np.all(result[0, 30:50] == 1)
        # Before and after should be 0s
        assert np.all(result[0, :30] == 0)
        assert np.all(result[0, 50:] == 0)

    def test_multiple_neurons(self):
        """Should handle multiple neurons correctly."""
        length = 100
        fps = 20.0
        # Two neurons with different events
        st_ev_inds = [[20], [60]]
        end_ev_inds = [[40], [80]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        assert result.shape == (2, length)
        # First neuron active 20-40
        assert np.sum(result[0, 20:40]) > 0
        # Second neuron active 60-80
        assert np.sum(result[1, 60:80]) > 0

    def test_short_events_extended_to_min_duration(self):
        """Events shorter than MIN_EVENT_DUR should be extended."""
        length = 200
        fps = 20.0
        min_dur_frames = int(MIN_EVENT_DUR * fps)  # 0.5s * 20 = 10 frames

        # Create very short event (2 frames)
        st_ev_inds = [[50]]
        end_ev_inds = [[52]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Event should be extended to at least min duration
        event_duration = np.sum(result[0, :])
        assert event_duration >= min_dur_frames

    def test_long_events_truncated_to_max_duration(self):
        """Events longer than MAX_EVENT_DUR should be truncated."""
        length = 200
        fps = 20.0
        max_dur_frames = int(MAX_EVENT_DUR * fps)  # 2.5s * 20 = 50 frames

        # Create very long event (100 frames = 5 seconds)
        st_ev_inds = [[20]]
        end_ev_inds = [[120]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Event should be truncated to max duration
        event_duration = np.sum(result[0, :])
        assert event_duration <= max_dur_frames

    def test_no_events_returns_zeros(self):
        """Empty event lists should return all zeros."""
        length = 100
        fps = 20.0
        st_ev_inds = [[]]
        end_ev_inds = [[]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        assert result.shape == (1, length)
        assert np.all(result == 0)

    def test_handles_reversed_indices(self):
        """Should handle case where start > end (reversed)."""
        length = 100
        fps = 20.0
        # Reversed indices
        st_ev_inds = [[50]]
        end_ev_inds = [[30]]

        # Should not raise error
        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Should still mark some region
        assert np.sum(result) > 0


class TestExtractWvtEvents:
    """Test end-to-end wavelet event extraction."""

    def test_detects_clear_transients(self):
        """Should detect clear calcium-like transients."""
        np.random.seed(42)
        fps = 20.0
        n_frames = 500

        # Create signal with clear transient
        trace = np.random.randn(n_frames) * 0.01  # Low noise baseline

        # Add clear calcium transient at frame 200
        # Fast rise, slow decay (typical calcium dynamics)
        event_start = 200
        event_duration = 40  # 2 seconds at 20 fps
        decay_tau = 15
        for i in range(event_duration):
            trace[event_start + i] += 0.5 * np.exp(-i / decay_tau)

        traces = trace.reshape(1, -1)
        wvt_kwargs = {
            "fps": fps,
            "scale_length_thr": 20,  # Lower threshold for test
            "max_ampl_thr": 0.01,  # Lower amplitude threshold
        }

        st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(
            traces, wvt_kwargs, show_progress=False
        )

        # Should detect at least one event
        assert len(st_ev_inds[0]) >= 1

        # At least one detected event should be near the true event
        detected_near_true = any(
            abs(start - event_start) < 30 for start in st_ev_inds[0]
        )
        assert detected_near_true, f"No event detected near frame {event_start}"

    def test_multiple_separated_events(self):
        """Should detect multiple well-separated events."""
        np.random.seed(42)
        fps = 20.0
        n_frames = 1000

        trace = np.random.randn(n_frames) * 0.01
        event_times = [100, 400, 700]

        for event_start in event_times:
            event_duration = 40
            decay_tau = 15
            for i in range(event_duration):
                trace[event_start + i] += 0.4 * np.exp(-i / decay_tau)

        traces = trace.reshape(1, -1)
        wvt_kwargs = {
            "fps": fps,
            "scale_length_thr": 15,
            "max_ampl_thr": 0.01,
        }

        st_ev_inds, end_ev_inds, _ = extract_wvt_events(
            traces, wvt_kwargs, show_progress=False
        )

        # Should detect at least 2 events (might miss some due to thresholds)
        assert len(st_ev_inds[0]) >= 2

    def test_fps_adaptation(self):
        """Detection should work across different FPS values."""
        np.random.seed(42)

        for fps in [15.0, 20.0, 30.0]:
            n_frames = int(25 * fps)  # 25 seconds

            trace = np.random.randn(n_frames) * 0.01
            # Add event at 10 seconds
            event_start = int(10 * fps)
            event_duration = int(2 * fps)
            decay_tau = int(0.75 * fps)
            for i in range(event_duration):
                trace[event_start + i] += 0.5 * np.exp(-i / decay_tau)

            traces = trace.reshape(1, -1)
            wvt_kwargs = {
                "fps": fps,
                "scale_length_thr": 15,
                "max_ampl_thr": 0.01,
            }

            st_ev_inds, _, _ = extract_wvt_events(
                traces, wvt_kwargs, show_progress=False
            )

            # Should detect at least one event at each FPS
            assert len(st_ev_inds[0]) >= 1, f"Failed to detect event at {fps} Hz"

    def test_handles_constant_trace(self):
        """Should handle constant signal without errors."""
        fps = 20.0
        n_frames = 200

        # Constant trace (no events)
        trace = np.ones((1, n_frames))

        wvt_kwargs = {"fps": fps}

        # Should not raise but return empty or handle gracefully
        # Note: This might raise ValueError due to zero range normalization
        try:
            st_ev_inds, end_ev_inds, _ = extract_wvt_events(
                trace, wvt_kwargs, show_progress=False
            )
            # If it doesn't raise, should return empty lists
            assert len(st_ev_inds[0]) == 0
        except ValueError:
            # Expected for constant signal
            pass

    def test_validates_2d_input(self):
        """Should raise error for non-2D input."""
        trace_1d = np.random.randn(100)
        wvt_kwargs = {"fps": 20.0}

        with pytest.raises(ValueError, match="must be 2D"):
            extract_wvt_events(trace_1d, wvt_kwargs)

    def test_validates_wvt_kwargs_type(self):
        """Should raise error for non-dict wvt_kwargs."""
        traces = np.random.randn(1, 100)

        with pytest.raises(TypeError, match="must be dict"):
            extract_wvt_events(traces, "not a dict")


class TestEventDurationConstraints:
    """Test that event duration constraints are enforced correctly."""

    def test_min_duration_floor(self):
        """Events should never be shorter than MIN_EVENT_DUR."""
        length = 200
        fps = 20.0
        min_dur_frames = int(MIN_EVENT_DUR * fps)

        # Create multiple very short events
        st_ev_inds = [[10, 50, 90]]
        end_ev_inds = [[11, 51, 91]]  # All 1-frame events

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Find connected regions
        events = np.diff(np.concatenate([[0], result[0], [0]]))
        starts = np.where(events == 1)[0]
        ends = np.where(events == -1)[0]

        for start, end in zip(starts, ends):
            duration = end - start
            assert duration >= min_dur_frames

    def test_max_duration_ceiling(self):
        """Events should never be longer than MAX_EVENT_DUR."""
        length = 300
        fps = 20.0
        max_dur_frames = int(MAX_EVENT_DUR * fps)

        # Create very long event
        st_ev_inds = [[10]]
        end_ev_inds = [[250]]  # 240 frames = 12 seconds

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Total active frames should not exceed max duration
        total_active = np.sum(result[0, :])
        assert total_active <= max_dur_frames


class TestGetCwtRidges:
    """Test ridge detection from continuous wavelet transform."""

    def test_detects_single_transient(self):
        """Should detect ridge for single clear transient."""
        np.random.seed(42)
        fps = 20.0
        n_frames = 500

        # Create signal with single clear transient
        sig = np.random.randn(n_frames) * 0.02
        event_start = 200
        for i in range(60):
            sig[event_start + i] += 0.8 * np.exp(-i / 15)

        # Use default GMW parameters: beta=2, gamma=3
        ridges = get_cwt_ridges(
            sig, wavelet=('gmw', {'beta': 2, 'gamma': 3}),
            fps=fps, scmin=10, scmax=50
        )

        # Should find at least one ridge
        assert len(ridges) > 0

    def test_detects_multiple_separated_transients(self):
        """Should detect separate ridges for well-separated events."""
        np.random.seed(42)
        fps = 20.0
        n_frames = 800

        sig = np.random.randn(n_frames) * 0.02
        event_times = [100, 400, 650]

        for event_start in event_times:
            for i in range(50):
                sig[event_start + i] += 0.6 * np.exp(-i / 15)

        ridges = get_cwt_ridges(
            sig, wavelet=('gmw', {'beta': 2, 'gamma': 3}),
            fps=fps, scmin=10, scmax=50
        )

        # Should find multiple ridges (at least 2)
        assert len(ridges) >= 2

    def test_validates_scmin_scmax(self):
        """Should raise error when scmin >= scmax."""
        sig = np.random.randn(200)

        with pytest.raises(ValueError, match="scmin.*must be less than scmax"):
            get_cwt_ridges(
                sig, wavelet=('gmw', {'beta': 2, 'gamma': 3}),
                fps=20, scmin=50, scmax=30
            )

    def test_validates_1d_input(self):
        """Should raise error for non-1D input."""
        sig_2d = np.random.randn(10, 100)

        with pytest.raises(ValueError, match="must be 1D"):
            get_cwt_ridges(
                sig_2d, wavelet=('gmw', {'beta': 2, 'gamma': 3}),
                fps=20, scmin=10, scmax=50
            )

    def test_ridge_has_required_attributes(self):
        """Ridge objects should have indices, ampls, scales."""
        np.random.seed(42)
        fps = 20.0

        sig = np.zeros(300)
        sig[100:150] = np.exp(-np.arange(50) / 15) * 0.8

        ridges = get_cwt_ridges(
            sig, wavelet=('gmw', {'beta': 2, 'gamma': 3}),
            fps=fps, scmin=5, scmax=30
        )

        if len(ridges) > 0:
            ridge = ridges[0]
            assert hasattr(ridge, 'indices')
            assert hasattr(ridge, 'ampls')  # Not 'amplitudes'
            assert hasattr(ridge, 'scales')


class TestEventsFromTrace:
    """Test single-trace event detection pipeline."""

    def test_validates_1d_trace(self):
        """Should raise error for non-1D trace."""
        trace_2d = np.random.randn(5, 100)
        scales = get_adaptive_wavelet_scales(fps=20, n_scales=30)

        from ssqueezepy import Wavelet
        wavelet = Wavelet(('gmw', {'beta': 2, 'gamma': 3}))
        rel_wvt_times = np.ones(30)

        with pytest.raises(ValueError, match="must be 1D"):
            events_from_trace(
                trace_2d, wavelet, scales, rel_wvt_times, fps=20,
                sigma=1, eps=4, scale_length_thr=10,
                max_scale_thr=100, max_ampl_thr=0.01, max_dur_thr=3.0
            )

    def test_validates_empty_trace(self):
        """Should raise error for empty trace."""
        trace_empty = np.array([])
        scales = get_adaptive_wavelet_scales(fps=20, n_scales=30)

        from ssqueezepy import Wavelet
        wavelet = Wavelet(('gmw', {'beta': 2, 'gamma': 3}))
        rel_wvt_times = np.ones(30)

        with pytest.raises(ValueError, match="cannot be empty"):
            events_from_trace(
                trace_empty, wavelet, scales, rel_wvt_times, fps=20,
                sigma=1, eps=4, scale_length_thr=10,
                max_scale_thr=100, max_ampl_thr=0.01, max_dur_thr=3.0
            )

    def test_validates_constant_trace(self):
        """Should raise error for constant (zero-range) trace."""
        trace_const = np.ones(200)
        scales = get_adaptive_wavelet_scales(fps=20, n_scales=30)

        from ssqueezepy import Wavelet
        wavelet = Wavelet(('gmw', {'beta': 2, 'gamma': 3}))
        rel_wvt_times = np.ones(30)

        with pytest.raises(ValueError, match="zero range"):
            events_from_trace(
                trace_const, wavelet, scales, rel_wvt_times, fps=20,
                sigma=1, eps=4, scale_length_thr=10,
                max_scale_thr=100, max_ampl_thr=0.01, max_dur_thr=3.0
            )

    def test_validates_wavelet_type(self):
        """Should raise error for invalid wavelet type."""
        trace = np.random.randn(200)
        scales = get_adaptive_wavelet_scales(fps=20, n_scales=30)
        rel_wvt_times = np.ones(30)

        with pytest.raises(TypeError, match="must be Wavelet instance"):
            events_from_trace(
                trace, "not_a_wavelet", scales, rel_wvt_times, fps=20,
                sigma=1, eps=4, scale_length_thr=10,
                max_scale_thr=100, max_ampl_thr=0.01, max_dur_thr=3.0
            )

    def test_validates_scales_times_length_match(self):
        """Should raise error when scales and times have different lengths."""
        trace = np.random.randn(200)
        scales = get_adaptive_wavelet_scales(fps=20, n_scales=30)
        rel_wvt_times = np.ones(20)  # Wrong length

        from ssqueezepy import Wavelet
        wavelet = Wavelet(('gmw', {'beta': 2, 'gamma': 3}))

        with pytest.raises(ValueError, match="same length"):
            events_from_trace(
                trace, wavelet, scales, rel_wvt_times, fps=20,
                sigma=1, eps=4, scale_length_thr=10,
                max_scale_thr=100, max_ampl_thr=0.01, max_dur_thr=3.0
            )


class TestEventsToTsArrayNumba:
    """Test the internal Numba event array function via events_to_ts_array."""

    def test_handles_multiple_neurons_correctly(self):
        """Should correctly process multiple neurons with different event counts."""
        length = 200
        fps = 20.0

        # Neuron 1: 2 events, Neuron 2: 1 event, Neuron 3: 3 events
        st_ev_inds = [[20, 80], [50], [30, 100, 150]]
        end_ev_inds = [[40, 100], [70], [50, 120, 170]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        assert result.shape == (3, length)
        # Check each neuron has events
        assert np.sum(result[0, :]) > 0
        assert np.sum(result[1, :]) > 0
        assert np.sum(result[2, :]) > 0

    def test_duration_between_min_max(self):
        """Events within duration limits should be preserved exactly."""
        length = 200
        fps = 20.0
        min_dur_frames = int(MIN_EVENT_DUR * fps)
        max_dur_frames = int(MAX_EVENT_DUR * fps)

        # Create event with duration exactly in the valid range
        valid_duration = (min_dur_frames + max_dur_frames) // 2
        st_ev_inds = [[50]]
        end_ev_inds = [[50 + valid_duration]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Duration should be preserved
        actual_duration = np.sum(result[0, :])
        assert actual_duration == valid_duration

    def test_events_at_boundaries(self):
        """Events at signal start and end should be handled correctly."""
        length = 100
        fps = 20.0

        # Event at very start
        st_ev_inds = [[0, 80]]
        end_ev_inds = [[15, 95]]

        result = events_to_ts_array(length, st_ev_inds, end_ev_inds, fps)

        # Should mark events without going out of bounds
        assert result.shape == (1, length)
        assert np.sum(result[0, :15]) > 0  # Start event
        assert np.sum(result[0, 80:]) > 0  # End event

class TestWaveletBatchOptimization:
    """Test batch processing optimization with pre-computed wavelet objects."""

    def test_extract_wvt_events_with_precomputed_wavelet(self):
        """Pre-computed wavelet should produce identical results to default."""
        from ssqueezepy.wavelets import Wavelet, time_resolution
        
        # Create synthetic calcium trace with clear transient
        fps = 20
        t = np.linspace(0, 10, 200)
        trace = np.zeros(200)
        # Add calcium transient (exponential rise and decay)
        transient_start = 50
        transient_end = 80
        trace[transient_start:transient_end] = np.exp(-np.arange(30) / 10) * 0.5
        traces = trace.reshape(1, -1)
        
        wvt_kwargs = {'fps': fps}
        
        # Run without pre-computed wavelet (default path)
        st_ev_default, end_ev_default, ridges_default = extract_wvt_events(
            traces, wvt_kwargs, show_progress=False
        )
        
        # Pre-compute wavelet and time_resolution
        from driada.experiment.wavelet_event_detection import get_adaptive_wavelet_scales
        gamma = wvt_kwargs.get('gamma', 3)
        beta = wvt_kwargs.get('beta', 2)
        wavelet_precomputed = Wavelet(
            ("gmw", {"gamma": gamma, "beta": beta, "centered_scale": True}), N=8196
        )
        manual_scales = get_adaptive_wavelet_scales(fps)
        rel_wvt_times_precomputed = [
            time_resolution(wavelet_precomputed, scale=sc, nondim=False, min_decay=200)
            for sc in manual_scales
        ]
        
        # Run with pre-computed wavelet
        st_ev_optimized, end_ev_optimized, ridges_optimized = extract_wvt_events(
            traces, wvt_kwargs, show_progress=False,
            wavelet=wavelet_precomputed, rel_wvt_times=rel_wvt_times_precomputed
        )
        
        # Results should be identical
        assert len(st_ev_default) == len(st_ev_optimized)
        assert len(end_ev_default) == len(end_ev_optimized)
        if len(st_ev_default[0]) > 0:
            assert np.array_equal(st_ev_default[0], st_ev_optimized[0])
            assert np.array_equal(end_ev_default[0], end_ev_optimized[0])

    def test_reconstruct_spikes_with_precomputed_wavelet(self):
        """Neuron.reconstruct_spikes should work with pre-computed objects."""
        from driada.experiment.neuron import Neuron
        from ssqueezepy.wavelets import Wavelet, time_resolution
        
        # Create synthetic calcium trace
        fps = 20
        trace = np.zeros(200)
        trace[50:80] = np.exp(-np.arange(30) / 10) * 0.5
        
        neuron = Neuron(cell_id="test", ca=trace, sp=None, fps=fps)
        
        # Reconstruct without optimization (default)
        neuron.reconstruct_spikes(method='wavelet', fps=fps, iterative=False)
        asp_default = neuron.asp.data.copy()
        
        # Pre-compute wavelet objects
        from driada.experiment.wavelet_event_detection import get_adaptive_wavelet_scales
        wavelet_precomputed = Wavelet(
            ("gmw", {"gamma": 3, "beta": 2, "centered_scale": True}), N=8196
        )
        manual_scales = get_adaptive_wavelet_scales(fps)
        rel_wvt_times_precomputed = [
            time_resolution(wavelet_precomputed, scale=sc, nondim=False, min_decay=200)
            for sc in manual_scales
        ]
        
        # Create new neuron and reconstruct with optimization
        neuron2 = Neuron(cell_id="test2", ca=trace, sp=None, fps=fps)
        neuron2.reconstruct_spikes(
            method='wavelet', fps=fps, iterative=False,
            wavelet=wavelet_precomputed, rel_wvt_times=rel_wvt_times_precomputed
        )
        asp_optimized = neuron2.asp.data
        
        # Results should be identical
        assert np.array_equal(asp_default, asp_optimized)

    def test_iterative_reconstruction_with_precomputed(self):
        """Iterative reconstruction should work with pre-computed objects."""
        from driada.experiment.neuron import Neuron
        from ssqueezepy.wavelets import Wavelet, time_resolution
        
        # Create synthetic trace with overlapping events
        fps = 20
        trace = np.zeros(300)
        trace[50:100] = np.exp(-np.arange(50) / 15) * 0.6
        trace[90:130] = trace[90:130] + np.exp(-np.arange(40) / 12) * 0.4
        
        neuron = Neuron(cell_id="test", ca=trace, sp=None, fps=fps)
        
        # Pre-compute wavelet objects
        from driada.experiment.wavelet_event_detection import get_adaptive_wavelet_scales
        wavelet_precomputed = Wavelet(
            ("gmw", {"gamma": 3, "beta": 2, "centered_scale": True}), N=8196
        )
        manual_scales = get_adaptive_wavelet_scales(fps)
        rel_wvt_times_precomputed = [
            time_resolution(wavelet_precomputed, scale=sc, nondim=False, min_decay=200)
            for sc in manual_scales
        ]
        
        # Iterative reconstruction with optimization
        neuron.reconstruct_spikes(
            method='wavelet', fps=fps, iterative=True, n_iter=3,
            wavelet=wavelet_precomputed, rel_wvt_times=rel_wvt_times_precomputed
        )
        
        # Should successfully complete and produce events
        assert neuron.asp is not None
        assert len(neuron.asp.data) == len(trace)

    def test_backward_compatibility_none_defaults(self):
        """Passing None should behave identically to omitting parameters."""
        # Create synthetic trace
        fps = 20
        trace = np.zeros(100)
        trace[30:50] = 0.5
        traces = trace.reshape(1, -1)
        
        wvt_kwargs = {'fps': fps}
        
        # Call without explicit None
        st_ev_1, end_ev_1, _ = extract_wvt_events(traces, wvt_kwargs, show_progress=False)
        
        # Call with explicit None
        st_ev_2, end_ev_2, _ = extract_wvt_events(
            traces, wvt_kwargs, show_progress=False,
            wavelet=None, rel_wvt_times=None
        )
        
        # Should produce identical results
        assert len(st_ev_1) == len(st_ev_2)
        if len(st_ev_1[0]) > 0:
            assert np.array_equal(st_ev_1[0], st_ev_2[0])
            assert np.array_equal(end_ev_1[0], end_ev_2[0])

    def test_precomputed_reusable_across_multiple_calls(self):
        """Pre-computed objects should be reusable for multiple neurons."""
        from ssqueezepy.wavelets import Wavelet, time_resolution
        
        fps = 20
        n_neurons = 5
        
        # Create multiple synthetic traces
        traces = []
        for i in range(n_neurons):
            trace = np.zeros(150)
            start = 30 + i * 10
            trace[start:start+20] = 0.4 + i * 0.1
            traces.append(trace)
        traces = np.array(traces)
        
        # Pre-compute once
        from driada.experiment.wavelet_event_detection import get_adaptive_wavelet_scales
        wavelet_shared = Wavelet(
            ("gmw", {"gamma": 3, "beta": 2, "centered_scale": True}), N=8196
        )
        manual_scales = get_adaptive_wavelet_scales(fps)
        rel_wvt_times_shared = [
            time_resolution(wavelet_shared, scale=sc, nondim=False, min_decay=200)
            for sc in manual_scales
        ]
        
        wvt_kwargs = {'fps': fps}
        
        # Use same pre-computed objects for all neurons
        st_ev, end_ev, _ = extract_wvt_events(
            traces, wvt_kwargs, show_progress=False,
            wavelet=wavelet_shared, rel_wvt_times=rel_wvt_times_shared
        )
        
        # Should successfully process all neurons
        assert len(st_ev) == n_neurons
        assert len(end_ev) == n_neurons
