"""Tests for threshold-based event detection in iterative mode.

These tests verify that the fixed threshold fix prevents finding spurious
events in pure noise signals.
"""

import pytest
import numpy as np
from scipy.stats import median_abs_deviation

from driada.experiment.neuron import Neuron


class TestThresholdDetectionOnNoise:
    """Test that threshold detection doesn't find escalating events in pure noise."""

    def test_iterative_threshold_on_pure_noise_bounded_events(self):
        """
        Verify that iterative threshold detection on realistic calcium-like noise
        finds a bounded number of events (not escalating across iterations).

        This tests the fix where thresholds are computed from the ORIGINAL
        signal, not from residuals.

        Note: We use non-negative noise (baseline + Gaussian) to simulate
        realistic calcium signal without events, since Neuron preprocessing
        clips negative values to 0.
        """
        np.random.seed(42)

        # Create realistic calcium-like noise (non-negative baseline + noise)
        # This simulates a calcium signal with no real events, just baseline fluorescence
        n_frames = 3000
        fps = 20.0
        baseline = 1.0  # Baseline fluorescence level
        noise_std = 0.1  # 10% noise relative to baseline
        calcium_noise = baseline + np.random.randn(n_frames) * noise_std
        calcium_noise = np.maximum(calcium_noise, 0)  # Ensure non-negative

        # Create neuron with noise as calcium signal
        neuron = Neuron(cell_id=0, ca=calcium_noise, sp=None, fps=fps)

        # Run iterative threshold detection with default parameters
        # Using n_mad=4 means threshold = median + 4*MAD ≈ baseline + 0.4
        # Very few points should exceed this in pure noise
        neuron.reconstruct_spikes(
            method='threshold',
            n_iter=5,
            n_mad=4.0,
            adaptive_thresholds=False,  # Use fixed threshold for clarity
            use_scaled=False,
        )

        # Get events found (threshold_events contains SimpleEvent objects)
        n_events = len(neuron.threshold_events) if neuron.threshold_events is not None else 0

        # With 4 MAD threshold on Gaussian noise around baseline:
        # - P(x > μ + 4σ) ≈ 0.003% per tail
        # - For 3000 frames: ~0.1 expected exceedances
        # - With min_duration_frames=3, isolated exceedances are filtered out
        # - Should find very few (if any) spurious events

        # Allow some false positives but verify it's bounded
        assert n_events < 10, (
            f"Found {n_events} events in calcium-like noise - threshold may be "
            "recalibrating to residuals instead of using fixed threshold"
        )

    def test_iterative_threshold_not_escalating(self):
        """
        Verify that the number of events found doesn't escalate with more iterations.

        The old bug: each iteration recalculates MAD from residual, finds more
        "events" in the tails, creating a vicious cycle.

        The fix: use fixed threshold from original signal, so additional iterations
        only find events in residual that exceed the ORIGINAL noise floor.
        """
        np.random.seed(123)

        # Create pure noise
        n_frames = 2000
        fps = 20.0
        pure_noise = np.random.randn(n_frames)

        # Run with different iteration counts
        events_by_n_iter = []
        for n_iter in [1, 3, 5, 10]:
            neuron = Neuron(cell_id=0, ca=pure_noise.copy(), sp=None, fps=fps)
            neuron.reconstruct_spikes(
                method='threshold',
                n_iter=n_iter,
                n_mad=4.0,
                adaptive_thresholds=False,  # Fixed threshold across iterations
                use_scaled=False,
            )
            n_events = len(neuron.threshold_events) if neuron.threshold_events is not None else 0
            events_by_n_iter.append((n_iter, n_events))

        # Verify events don't scale linearly with iterations
        # With fixed threshold, more iterations shouldn't find proportionally more events
        events_1_iter = events_by_n_iter[0][1]
        events_10_iter = events_by_n_iter[3][1]

        # If recalibrating to residuals, we'd expect events_10_iter >> events_1_iter
        # With fixed threshold, they should be similar (within 2x)
        if events_1_iter > 0:
            ratio = events_10_iter / events_1_iter
            assert ratio < 3, (
                f"Events escalated from {events_1_iter} (1 iter) to {events_10_iter} "
                f"(10 iter), ratio={ratio:.1f}. This suggests threshold is being "
                "recalibrated to residuals."
            )

    def test_threshold_computed_from_original_signal(self):
        """
        Verify that threshold is computed from original signal statistics,
        not from residuals.
        """
        np.random.seed(456)

        # Create signal with known statistics
        n_frames = 1000
        fps = 20.0
        signal = np.random.randn(n_frames) * 2.0 + 1.0  # mean=1, std≈2

        # Expected threshold at 4 MAD
        expected_median = np.median(signal)
        expected_mad = median_abs_deviation(signal, scale='normal')
        expected_threshold = expected_median + 4.0 * expected_mad

        neuron = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)

        # Run detection - if threshold is computed correctly from original,
        # only points above expected_threshold should trigger events
        neuron.reconstruct_spikes(
            method='threshold',
            n_iter=1,
            n_mad=4.0,
            use_scaled=False,
        )

        # Check that events only occur where signal exceeds expected threshold
        if neuron.threshold_events is not None and len(neuron.threshold_events) > 0:
            for event in neuron.threshold_events:
                st, end = int(event.start), int(event.end)
                # At least some point in the event window should exceed threshold
                event_max = np.max(signal[st:end+1])
                assert event_max >= expected_threshold * 0.95, (
                    f"Event at [{st}:{end}] has max={event_max:.2f} but "
                    f"threshold={expected_threshold:.2f}. Threshold may not be "
                    "computed from original signal."
                )

    def test_explicit_threshold_used_for_all_iterations(self):
        """
        Verify that when user provides explicit threshold, it's used for all iterations.
        """
        np.random.seed(789)

        # Create noise with some high values
        n_frames = 1000
        fps = 20.0
        signal = np.random.randn(n_frames)

        # Set explicit threshold that would find ~10 points in Gaussian noise
        # At 2.3σ, about 2% of points exceed threshold
        explicit_threshold = 2.3

        neuron = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron.reconstruct_spikes(
            method='threshold',
            n_iter=3,
            threshold=explicit_threshold,  # Explicit threshold
            use_scaled=False,
        )

        # Verify events found are consistent with explicit threshold
        if neuron.threshold_events is not None and len(neuron.threshold_events) > 0:
            for event in neuron.threshold_events:
                st, end = int(event.start), int(event.end)
                event_max = np.max(signal[st:end+1])
                # Events should only occur where signal exceeds explicit threshold
                assert event_max >= explicit_threshold * 0.9, (
                    f"Event found below explicit threshold"
                )


class TestThresholdDetectionWithRealEvents:
    """Test that threshold detection still finds real events correctly."""

    def test_finds_clear_events_in_noisy_signal(self):
        """Verify that real calcium-like events are detected in noisy signal."""
        np.random.seed(42)

        n_frames = 2000
        fps = 20.0

        # Create noisy baseline
        signal = np.random.randn(n_frames) * 0.1

        # Add 5 clear calcium-like events
        event_times = [200, 500, 900, 1300, 1700]
        event_duration = 50  # 2.5 seconds at 20 fps
        event_amplitude = 2.0  # 20x noise std

        for t in event_times:
            # Exponential decay transient
            decay = np.exp(-np.arange(event_duration) / (event_duration / 3))
            signal[t:t+event_duration] += event_amplitude * decay

        neuron = Neuron(cell_id=0, ca=signal, sp=None, fps=fps)
        neuron.reconstruct_spikes(
            method='threshold',
            n_iter=3,
            n_mad=4.0,
            use_scaled=False,
        )

        # Check sp_count (unique spikes in ASP) rather than len(threshold_events).
        # Iterative detection may report multiple overlapping event windows that
        # map to the same spike onset - these are correctly merged in ASP via
        # amplitude summing, so sp_count reflects the true number of detected spikes.
        n_spikes = neuron.sp_count
        assert 3 <= n_spikes <= 8, (
            f"Expected ~5 spikes, found {n_spikes}. "
            "Detection may be too aggressive or too conservative."
        )

        # Check that events are near the true event times
        if neuron.threshold_events is not None:
            detected_starts = [int(e.start) for e in neuron.threshold_events]
            for true_time in event_times:
                # At least one detected event should start within 30 frames of true time
                close_events = [s for s in detected_starts if abs(s - true_time) < 30]
                assert len(close_events) > 0, (
                    f"No event detected near true event at frame {true_time}"
                )

    def test_adaptive_thresholds_find_more_events(self):
        """Verify adaptive thresholds find progressively smaller events."""
        np.random.seed(42)

        n_frames = 2000
        fps = 20.0

        # Create signal with events of varying amplitudes
        signal = np.random.randn(n_frames) * 0.1

        # Large event
        signal[200:250] += 2.0 * np.exp(-np.arange(50) / 15)
        # Medium event
        signal[600:650] += 1.0 * np.exp(-np.arange(50) / 15)
        # Small event
        signal[1000:1050] += 0.5 * np.exp(-np.arange(50) / 15)

        # Without adaptive thresholds
        neuron_fixed = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron_fixed.reconstruct_spikes(
            method='threshold',
            n_iter=5,
            n_mad=4.0,
            adaptive_thresholds=False,
            use_scaled=False,
        )
        n_events_fixed = len(neuron_fixed.threshold_events) if neuron_fixed.threshold_events else 0

        # With adaptive thresholds
        neuron_adaptive = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron_adaptive.reconstruct_spikes(
            method='threshold',
            n_iter=5,
            n_mad=4.0,
            adaptive_thresholds=True,
            use_scaled=False,
        )
        n_events_adaptive = len(neuron_adaptive.threshold_events) if neuron_adaptive.threshold_events else 0

        # Adaptive should find at least as many events
        assert n_events_adaptive >= n_events_fixed, (
            f"Adaptive ({n_events_adaptive}) should find >= fixed ({n_events_fixed})"
        )


class TestIterativeDetectionMergesDuplicates:
    """Test that iterative detection correctly merges duplicate spikes in ASP."""

    def test_well_separated_events_no_extra_spikes_in_asp(self):
        """
        Verify that iterative detection on well-separated events doesn't add
        spurious spikes to ASP.

        When the same event is detected in both iteration 1 (original signal)
        and iteration 2 (residual), both detections should map to the same
        onset time in ASP due to amplitude summing.
        """
        np.random.seed(42)

        n_frames = 2000
        fps = 20.0

        # Create well-separated events
        signal = np.random.randn(n_frames) * 0.1
        event_times = [200, 500, 900, 1300, 1700]
        event_duration = 50
        event_amplitude = 2.0

        for t in event_times:
            decay = np.exp(-np.arange(event_duration) / 16)
            signal[t:t+event_duration] += event_amplitude * decay

        # Run with 1 iteration
        neuron_1iter = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron_1iter.reconstruct_spikes(
            method='threshold',
            n_iter=1,
            n_mad=4.0,
            use_scaled=False,
        )
        sp_count_1iter = neuron_1iter.sp_count

        # Run with 3 iterations
        neuron_3iter = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron_3iter.reconstruct_spikes(
            method='threshold',
            n_iter=3,
            n_mad=4.0,
            use_scaled=False,
        )
        sp_count_3iter = neuron_3iter.sp_count

        # Spike count should be similar - duplicates should merge via amplitude summing
        # Allow small variation due to noise, but not proportional to n_iter
        assert sp_count_3iter <= sp_count_1iter + 2, (
            f"3 iterations ({sp_count_3iter} spikes) should not add many more spikes "
            f"than 1 iteration ({sp_count_1iter} spikes) for well-separated events"
        )

    def test_overlapping_events_finds_individual_spikes(self):
        """
        Verify that iterative detection on overlapping events finds individual
        spikes that were missed in the first iteration.

        When events overlap, iteration 1 may detect only the envelope.
        Subsequent iterations should find individual spikes in the residual,
        improving reconstruction quality.
        """
        np.random.seed(42)

        n_frames = 2000
        fps = 20.0
        tau_decay = 15

        # Create overlapping events
        signal = np.random.randn(n_frames) * 0.05
        event_times = [200, 230, 260,  # Cluster of 3
                       500, 540,        # Cluster of 2
                       800,             # Isolated
                       1000, 1020, 1040, 1060]  # Cluster of 4

        for t in event_times:
            remaining = min(60, n_frames - t)
            decay = np.exp(-np.arange(remaining) / tau_decay)
            signal[t:t+remaining] += 1.5 * decay

        # Run with 1 iteration
        neuron_1iter = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron_1iter.reconstruct_spikes(
            method='threshold',
            n_iter=1,
            n_mad=4.0,
            use_scaled=False,
        )

        # Run with 3 iterations
        neuron_3iter = Neuron(cell_id=0, ca=signal.copy(), sp=None, fps=fps)
        neuron_3iter.reconstruct_spikes(
            method='threshold',
            n_iter=3,
            n_mad=4.0,
            use_scaled=False,
        )

        # With overlapping events, more iterations should find more spikes
        # (individual spikes in clusters that were merged in iteration 1)
        assert neuron_3iter.sp_count >= neuron_1iter.sp_count, (
            f"3 iterations ({neuron_3iter.sp_count} spikes) should find at least "
            f"as many spikes as 1 iteration ({neuron_1iter.sp_count} spikes) "
            "for overlapping events"
        )

        # Reconstruction quality should improve with more iterations
        t_rise = neuron_1iter.default_t_rise
        t_off = neuron_1iter.default_t_off

        recon_1iter = Neuron.get_restored_calcium(
            neuron_1iter.asp.data, t_rise, t_off
        )[:n_frames]
        recon_3iter = Neuron.get_restored_calcium(
            neuron_3iter.asp.data, t_rise, t_off
        )[:n_frames]

        ss_tot = np.sum((signal - np.mean(signal)) ** 2)
        r2_1iter = 1 - np.sum((signal - recon_1iter) ** 2) / ss_tot
        r2_3iter = 1 - np.sum((signal - recon_3iter) ** 2) / ss_tot

        assert r2_3iter >= r2_1iter - 0.05, (
            f"3 iterations (R2={r2_3iter:.3f}) should have similar or better "
            f"reconstruction than 1 iteration (R2={r2_1iter:.3f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
