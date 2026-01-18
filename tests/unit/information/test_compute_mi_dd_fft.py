"""
Tests for compute_mi_dd_fft function (discrete-discrete FFT acceleration).

Following the pattern of test_compute_mi_mts_discrete_fft.py with comprehensive
equivalence tests between FFT and loop implementations.
"""
import pytest
import numpy as np
from sklearn.metrics import mutual_info_score
from driada.information.info_fft import mi_dd_fft, compute_mi_dd_fft


class TestDiscreteDiscreteFFTCorrectness:
    """Test correctness against sklearn mutual_info_score reference implementation."""

    def test_matches_sklearn_at_zero_shift(self):
        """Verify FFT matches sklearn mutual_info_score at shift=0."""
        np.random.seed(42)
        n = 500
        Ym, Yn = 3, 4

        x = np.random.randint(0, Ym, n)
        y = np.random.randint(0, Yn, n)

        # Compute with FFT at shift=0 (no bias correction to match sklearn)
        mi_fft = compute_mi_dd_fft(x, y, np.array([0]), biascorrect=False)[0]

        # Compute with sklearn reference
        mi_sklearn = mutual_info_score(x, y) / np.log(2)  # Convert to bits

        # Should match to machine precision
        np.testing.assert_allclose(mi_fft, mi_sklearn, rtol=1e-10, atol=1e-14)

    def test_matches_sklearn_at_zero_shift_binary(self):
        """Verify FFT matches sklearn for binary variables."""
        np.random.seed(43)
        n = 600

        x = np.random.randint(0, 2, n)
        y = np.random.randint(0, 2, n)

        mi_fft = compute_mi_dd_fft(x, y, np.array([0]), biascorrect=False)[0]
        mi_sklearn = mutual_info_score(x, y) / np.log(2)

        np.testing.assert_allclose(mi_fft, mi_sklearn, rtol=1e-10, atol=1e-14)

    def test_fft_matches_loop_all_shifts_ym3_yn4(self):
        """CRITICAL: Verify FFT matches loop (sklearn with np.roll) at ALL shifts."""
        np.random.seed(99)
        n = 500
        Ym, Yn = 3, 4
        nsh = 50

        x = np.random.randint(0, Ym, n)
        y = np.random.randint(0, Yn, n)
        shifts = np.arange(nsh)

        # Compute with FFT (biascorrect=False to match sklearn)
        mi_fft = compute_mi_dd_fft(x, y, shifts, biascorrect=False)

        # Compute with loop (ground truth)
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            y_shifted = np.roll(y, int(s))
            mi_loop[i] = mutual_info_score(x, y_shifted) / np.log(2)

        # FFT must match loop at ALL shifts (this is the gold standard)
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-10, atol=1e-14)

    def test_fft_matches_loop_all_shifts_binary(self):
        """CRITICAL: Verify FFT matches loop at ALL shifts for binary variables."""
        np.random.seed(100)
        n = 400
        nsh = 40

        x = np.random.randint(0, 2, n)
        y = np.random.randint(0, 2, n)
        shifts = np.arange(nsh)

        mi_fft = compute_mi_dd_fft(x, y, shifts, biascorrect=False)

        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            y_shifted = np.roll(y, int(s))
            mi_loop[i] = mutual_info_score(x, y_shifted) / np.log(2)

        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-10, atol=1e-14)

    def test_fft_matches_loop_all_shifts_ym5_yn6(self):
        """CRITICAL: Verify FFT matches loop at ALL shifts for larger cardinality."""
        np.random.seed(101)
        n = 600
        Ym, Yn = 5, 6
        nsh = 60

        x = np.random.randint(0, Ym, n)
        y = np.random.randint(0, Yn, n)
        shifts = np.arange(nsh)

        mi_fft = compute_mi_dd_fft(x, y, shifts, biascorrect=False)

        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            y_shifted = np.roll(y, int(s))
            mi_loop[i] = mutual_info_score(x, y_shifted) / np.log(2)

        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-10, atol=1e-14)

    def test_fft_matches_loop_negative_shifts(self):
        """Verify FFT matches loop for negative shifts."""
        np.random.seed(102)
        n = 300
        Ym, Yn = 3, 3

        x = np.random.randint(0, Ym, n)
        y = np.random.randint(0, Yn, n)
        shifts = np.arange(-30, 31)  # Include negative shifts

        mi_fft = compute_mi_dd_fft(x, y, shifts, biascorrect=False)

        mi_loop = np.zeros(len(shifts))
        for i, s in enumerate(shifts):
            y_shifted = np.roll(y, int(s))
            mi_loop[i] = mutual_info_score(x, y_shifted) / np.log(2)

        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-10, atol=1e-14)


class TestDiscreteDiscreteFFTWithBiasCorrection:
    """Test bias correction consistency."""

    def test_biascorrect_changes_values(self):
        """Verify biascorrect=True produces different (lower) values than False."""
        np.random.seed(42)
        n = 200
        x = np.random.randint(0, 3, n)
        y = np.random.randint(0, 4, n)
        shifts = np.arange(10)

        mi_nobias = compute_mi_dd_fft(x, y, shifts, biascorrect=False)
        mi_bias = compute_mi_dd_fft(x, y, shifts, biascorrect=True)

        # Bias correction should reduce MI values (Miller-Madow correction is negative)
        assert np.all(mi_bias <= mi_nobias + 1e-10)
        # Values should be different
        assert not np.allclose(mi_nobias, mi_bias)

    def test_biascorrect_fft_matches_fft_internal(self):
        """Verify compute_mi_dd_fft passes biascorrect correctly to mi_dd_fft."""
        np.random.seed(43)
        n = 300
        x = np.random.randint(0, 4, n)
        y = np.random.randint(0, 3, n)
        shifts = np.arange(20)

        # Using wrapper
        mi_wrapper_nobias = compute_mi_dd_fft(x, y, shifts, biascorrect=False)
        mi_wrapper_bias = compute_mi_dd_fft(x, y, shifts, biascorrect=True)

        # Using raw function
        mi_raw_nobias = mi_dd_fft(x, y, shifts, biascorrect=False)
        mi_raw_bias = mi_dd_fft(x, y, shifts, biascorrect=True)

        np.testing.assert_allclose(mi_wrapper_nobias, mi_raw_nobias, rtol=1e-14)
        np.testing.assert_allclose(mi_wrapper_bias, mi_raw_bias, rtol=1e-14)


class TestDiscreteDiscreteFFTEdgeCases:
    """Test edge cases and special inputs."""

    def test_identical_variables(self):
        """MI between identical variables should be entropy of the variable."""
        np.random.seed(42)
        n = 500
        Ym = 4
        x = np.random.randint(0, Ym, n)

        # MI(X;X) at shift=0 should equal H(X)
        mi_fft = compute_mi_dd_fft(x, x, np.array([0]), biascorrect=False)[0]
        mi_sklearn = mutual_info_score(x, x) / np.log(2)

        np.testing.assert_allclose(mi_fft, mi_sklearn, rtol=1e-10)
        # Should be positive (entropy)
        assert mi_fft > 0

    def test_independent_variables(self):
        """MI between independent variables should be close to zero."""
        np.random.seed(42)
        n = 5000  # Large n for better independence approximation
        x = np.random.randint(0, 3, n)
        y = np.random.randint(0, 4, n)

        mi_fft = compute_mi_dd_fft(x, y, np.array([0]), biascorrect=False)[0]

        # Should be close to zero for independent variables
        assert mi_fft < 0.02  # Allow small positive value due to finite sample

    def test_perfectly_correlated(self):
        """MI for perfectly correlated variables should equal marginal entropy."""
        np.random.seed(42)
        n = 500
        x = np.random.randint(0, 4, n)
        y = x.copy()  # Perfect correlation

        mi_fft = compute_mi_dd_fft(x, y, np.array([0]), biascorrect=False)[0]
        mi_sklearn = mutual_info_score(x, y) / np.log(2)

        np.testing.assert_allclose(mi_fft, mi_sklearn, rtol=1e-10)

    def test_single_shift(self):
        """Verify single shift computation works."""
        np.random.seed(42)
        n = 200
        x = np.random.randint(0, 3, n)
        y = np.random.randint(0, 3, n)

        mi_fft = compute_mi_dd_fft(x, y, np.array([0]), biascorrect=False)
        assert mi_fft.shape == (1,)
        assert mi_fft[0] >= 0

    def test_large_number_of_shifts(self):
        """Verify correctness with many shifts (typical shuffle count)."""
        np.random.seed(42)
        n = 1000
        nsh = 500  # Simulates typical shuffle count
        x = np.random.randint(0, 4, n)
        y = np.random.randint(0, 3, n)

        shifts = np.arange(nsh)
        mi_fft = compute_mi_dd_fft(x, y, shifts, biascorrect=False)

        # Spot check a few shifts against loop
        for check_idx in [0, 100, 250, 499]:
            y_shifted = np.roll(y, int(shifts[check_idx]))
            mi_loop = mutual_info_score(x, y_shifted) / np.log(2)
            np.testing.assert_allclose(mi_fft[check_idx], mi_loop, rtol=1e-10, atol=1e-14)


class TestDiscreteDiscreteFFTPerformance:
    """Test that FFT is faster than loop for many shifts."""

    @pytest.mark.slow
    def test_fft_faster_than_loop(self):
        """Verify FFT is significantly faster than loop for many shifts."""
        import time

        np.random.seed(42)
        n = 5000
        nsh = 1000
        Ym, Yn = 4, 5

        x = np.random.randint(0, Ym, n)
        y = np.random.randint(0, Yn, n)
        shifts = np.arange(nsh)

        # Time FFT
        start = time.time()
        mi_fft = compute_mi_dd_fft(x, y, shifts, biascorrect=False)
        time_fft = time.time() - start

        # Time loop
        start = time.time()
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            y_shifted = np.roll(y, int(s))
            mi_loop[i] = mutual_info_score(x, y_shifted) / np.log(2)
        time_loop = time.time() - start

        # Verify correctness
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-10, atol=1e-14)

        # FFT should be faster (typically 10-100x for these parameters)
        speedup = time_loop / time_fft
        print(f"FFT: {time_fft:.4f}s, Loop: {time_loop:.4f}s, Speedup: {speedup:.1f}x")
        assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"
