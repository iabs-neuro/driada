"""
Tests for compute_mi_mts_mts_fft function (MultiTimeSeries vs MultiTimeSeries FFT acceleration).

Tests various dimension combinations (d1+d2 <= 6), edge cases, correctness vs mi_gg reference,
and validation of FFT-based block determinant approach.
"""
import pytest
import numpy as np
from driada.information.info_base import compute_mi_mts_mts_fft
from driada.information.gcmi import mi_gg


class TestMTSMTSFFTCorrectness:
    """Test correctness against mi_gg reference implementation."""

    def test_matches_mi_gg_at_zero_shift_d2_d2(self):
        """Verify FFT matches mi_gg reference at shift=0 for d1=2, d2=2."""
        np.random.seed(42)
        n = 200
        d1, d2 = 2, 2

        # Create random MTS data
        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)

        # Compute with FFT at shift=0
        mi_fft = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)

        # Compute with mi_gg reference
        joint = np.vstack([x1, x2])
        mi_ref = mi_gg(x1, x2, biascorrect=True, demeaned=False)

        # Should match closely (rtol=1e-8 to account for floating point differences in FFT)
        np.testing.assert_allclose(mi_fft[0], mi_ref, rtol=1e-8)

    def test_matches_mi_gg_at_zero_shift_d2_d3(self):
        """Verify FFT matches mi_gg reference at shift=0 for d1=2, d2=3."""
        np.random.seed(43)
        n = 150
        d1, d2 = 2, 3

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)

        mi_fft = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)
        mi_ref = mi_gg(x1, x2, biascorrect=True, demeaned=False)

        np.testing.assert_allclose(mi_fft[0], mi_ref, rtol=1e-8)

    def test_matches_mi_gg_at_zero_shift_d3_d3(self):
        """Verify FFT matches mi_gg reference at shift=0 for d1=3, d2=3 (max case)."""
        np.random.seed(44)
        n = 150
        d1, d2 = 3, 3

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)

        mi_fft = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)
        mi_ref = mi_gg(x1, x2, biascorrect=True, demeaned=False)

        np.testing.assert_allclose(mi_fft[0], mi_ref, rtol=1e-8)

    def test_correctness_multiple_shifts(self):
        """Verify FFT produces valid MI values across multiple shifts."""
        np.random.seed(45)
        n = 200
        d1, d2 = 2, 2
        nsh = 20

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)
        shifts = np.arange(nsh)

        # Compute with FFT
        mi_fft = compute_mi_mts_mts_fft(x1, x2, shifts, biascorrect=True)

        # All MI values should be valid
        assert mi_fft.shape == (nsh,)
        assert np.all(np.isfinite(mi_fft))
        assert np.all(mi_fft >= 0)  # MI is non-negative

        # Verify shift=0 matches mi_gg reference
        mi_ref_0 = mi_gg(x1, x2, biascorrect=True, demeaned=False)
        np.testing.assert_allclose(mi_fft[0], mi_ref_0, rtol=1e-8)

    def test_fft_matches_loop_all_shifts(self):
        """CRITICAL: Verify FFT matches loop (mi_gg with np.roll) at ALL shifts."""
        from driada.information.gcmi import copnorm

        np.random.seed(99)
        n = 150
        d1, d2 = 2, 2
        nsh = 15  # Test multiple non-zero shifts

        # Generate raw data
        x1_raw = np.random.randn(d1, n)
        x2_raw = np.random.randn(d2, n)

        # Copnorm data (matching what INTENSE does)
        x1 = copnorm(x1_raw)
        x2 = copnorm(x2_raw)
        shifts = np.arange(nsh)

        # Compute with FFT
        mi_fft = compute_mi_mts_mts_fft(x1, x2, shifts, biascorrect=True)

        # Compute with loop (ground truth)
        # Must use copnorm'd data and demeaned=True to match FFT behavior
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            x2_shifted = np.roll(x2, int(s), axis=1)
            mi_loop[i] = mi_gg(x1, x2_shifted, biascorrect=True, demeaned=True)

        # FFT must match loop at ALL shifts (this is the gold standard)
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-7, atol=1e-10)

    def test_fft_matches_loop_all_shifts_asymmetric(self):
        """CRITICAL: Verify FFT matches loop for asymmetric dimensions (d1≠d2)."""
        from driada.information.gcmi import copnorm

        np.random.seed(100)
        n = 120
        d1, d2 = 2, 3  # Asymmetric
        nsh = 12

        # Generate raw data
        x1_raw = np.random.randn(d1, n)
        x2_raw = np.random.randn(d2, n)

        # Copnorm data (matching what INTENSE does)
        x1 = copnorm(x1_raw)
        x2 = copnorm(x2_raw)
        shifts = np.arange(nsh)

        # Compute with FFT
        mi_fft = compute_mi_mts_mts_fft(x1, x2, shifts, biascorrect=True)

        # Compute with loop (ground truth)
        # Must use copnorm'd data and demeaned=True to match FFT behavior
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            x2_shifted = np.roll(x2, int(s), axis=1)
            mi_loop[i] = mi_gg(x1, x2_shifted, biascorrect=True, demeaned=True)

        # FFT must match loop at ALL shifts
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-7, atol=1e-10)

    def test_symmetry_x1_x2(self):
        """Verify MI(X1; X2) = MI(X2; X1) at shift=0."""
        np.random.seed(46)
        n = 150
        d1, d2 = 2, 3

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)

        mi_12 = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)
        mi_21 = compute_mi_mts_mts_fft(x2, x1, np.array([0]), biascorrect=True)

        # MI should be symmetric
        np.testing.assert_allclose(mi_12[0], mi_21[0], rtol=1e-10)

    def test_independent_variables_near_zero(self):
        """Independent Gaussian MTS should give MI ≈ 0."""
        np.random.seed(47)
        n = 500  # Larger n for better statistical properties
        d1, d2 = 2, 2

        # Create independent MTS
        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)

        mi = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)

        # Should be near zero for independent variables
        assert mi[0] < 0.05  # Small threshold for statistical fluctuations

    def test_correlated_variables_positive_mi(self):
        """Correlated MTS should give MI > 0."""
        np.random.seed(48)
        n = 300
        d1, d2 = 2, 2

        # Create correlated MTS
        x1 = np.random.randn(d1, n)
        # x2 shares some signal with x1
        x2 = np.zeros((d2, n))
        x2[0] = 0.7 * x1[0] + 0.3 * np.random.randn(n)  # Strong correlation
        x2[1] = np.random.randn(n)  # Independent dimension

        mi = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)

        # Should detect correlation
        assert mi[0] > 0.1  # Non-trivial MI


class TestMTSMTSFFTDimensions:
    """Test various dimension combinations."""

    def test_d1_2_d2_2(self):
        """Most common case: 2D vs 2D."""
        np.random.seed(49)
        n = 100
        x1 = np.random.randn(2, n)
        x2 = np.random.randn(2, n)
        shifts = np.array([0, 1, 5])

        mi = compute_mi_mts_mts_fft(x1, x2, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_d1_2_d2_3(self):
        """Medium case: 2D vs 3D."""
        np.random.seed(50)
        n = 100
        x1 = np.random.randn(2, n)
        x2 = np.random.randn(3, n)
        shifts = np.array([0, 2])

        mi = compute_mi_mts_mts_fft(x1, x2, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_d1_3_d2_3(self):
        """Max case: 3D vs 3D (d1+d2=6)."""
        np.random.seed(51)
        n = 100
        x1 = np.random.randn(3, n)
        x2 = np.random.randn(3, n)
        shifts = np.array([0, 1])

        mi = compute_mi_mts_mts_fft(x1, x2, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_d1_4_d2_3_raises(self):
        """Should raise NotImplementedError for d1+d2=7."""
        np.random.seed(52)
        n = 100
        x1 = np.random.randn(4, n)
        x2 = np.random.randn(3, n)
        shifts = np.array([0])

        with pytest.raises(NotImplementedError, match="d1\\+d2 > 6 is not implemented"):
            compute_mi_mts_mts_fft(x1, x2, shifts)

    def test_delegation_to_mts_1d_when_d1_is_1(self):
        """Should delegate to compute_mi_mts_fft when d1=1."""
        from driada.information.info_base import compute_mi_mts_fft

        np.random.seed(53)
        n = 100
        d1, d2 = 1, 2

        x1 = np.random.randn(d1, n)  # 1D MTS
        x2 = np.random.randn(d2, n)  # 2D MTS
        shifts = np.array([0, 1, 2])

        # Compute with MTS-MTS FFT (should delegate)
        mi_mts_mts = compute_mi_mts_mts_fft(x1, x2, shifts)

        # Compute with MTS-1D FFT directly
        mi_mts = compute_mi_mts_fft(x1[0], x2, shifts)

        # Should match exactly (delegation)
        np.testing.assert_allclose(mi_mts_mts, mi_mts, rtol=1e-12)

    def test_delegation_to_mts_1d_when_d2_is_1(self):
        """Should delegate to compute_mi_mts_fft when d2=1."""
        from driada.information.info_base import compute_mi_mts_fft

        np.random.seed(54)
        n = 100
        d1, d2 = 2, 1

        x1 = np.random.randn(d1, n)  # 2D MTS
        x2 = np.random.randn(d2, n)  # 1D MTS
        shifts = np.array([0, 1, 2])

        # Compute with MTS-MTS FFT (should delegate)
        mi_mts_mts = compute_mi_mts_mts_fft(x1, x2, shifts)

        # Compute with MTS-1D FFT directly
        mi_mts = compute_mi_mts_fft(x2[0], x1, shifts)

        # Should match exactly (delegation)
        np.testing.assert_allclose(mi_mts_mts, mi_mts, rtol=1e-12)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_singular_covariance_raises(self):
        """Should raise ValueError for linearly dependent dimensions."""
        np.random.seed(55)
        n = 100

        # Create singular covariance for x1
        x1_base = np.random.randn(n)
        x1 = np.vstack([x1_base, 2 * x1_base])  # Linearly dependent
        x2 = np.random.randn(2, n)
        shifts = np.array([0])

        # Should raise ValueError
        with pytest.raises(ValueError, match="Singular covariance"):
            compute_mi_mts_mts_fft(x1, x2, shifts)

    def test_transposed_input_raises_x1(self):
        """Should detect and raise on transposed input for x1."""
        np.random.seed(56)
        n = 100
        d1, d2 = 2, 2

        # Transposed x1: shape (n, d1) instead of (d1, n)
        x1_transposed = np.random.randn(n, d1)
        x2 = np.random.randn(d2, n)
        shifts = np.array([0])

        with pytest.raises(ValueError, match="shape looks transposed"):
            compute_mi_mts_mts_fft(x1_transposed, x2, shifts)

    def test_transposed_input_raises_x2(self):
        """Should detect and raise on transposed input for x2."""
        np.random.seed(57)
        n = 100
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n)
        # Transposed x2: shape (n, d2) instead of (d2, n)
        x2_transposed = np.random.randn(n, d2)
        shifts = np.array([0])

        with pytest.raises(ValueError, match="shape looks transposed"):
            compute_mi_mts_mts_fft(x1, x2_transposed, shifts)

    def test_mismatched_sample_sizes_raises(self):
        """Should raise ValueError if n1 != n2."""
        np.random.seed(58)
        n1, n2 = 100, 120
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n1)
        x2 = np.random.randn(d2, n2)
        shifts = np.array([0])

        with pytest.raises(ValueError, match="same number of samples"):
            compute_mi_mts_mts_fft(x1, x2, shifts)

    def test_min_sample_size_n_equals_2(self):
        """Should work with minimum n=2 samples (requires d1=d2=1 to avoid singular covariance)."""
        np.random.seed(59)
        n = 2
        d1, d2 = 1, 1  # Use 1D to avoid singular covariance with n=2

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)
        shifts = np.array([0])

        # Should work with n=2 (delegates to MTS-1D FFT for d=1)
        mi = compute_mi_mts_mts_fft(x1, x2, shifts)

        assert mi.shape == (1,)
        assert np.isfinite(mi[0])

    def test_n_equals_1_raises(self):
        """Should raise ValueError for n=1 (caught by transposed input or sample size check)."""
        n = 1
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)
        shifts = np.array([0])

        # Will be caught by either transpose check or minimum sample size check
        with pytest.raises(ValueError, match="shape looks transposed|at least 2 samples"):
            compute_mi_mts_mts_fft(x1, x2, shifts)

    def test_1d_array_auto_reshape(self):
        """Test with 1D arrays (should auto-reshape to (1, n))."""
        np.random.seed(60)
        n = 100

        x1_1d = np.random.randn(n)
        x2_1d = np.random.randn(n)
        shifts = np.array([0, 1])

        # Should delegate to MTS-1D FFT
        mi = compute_mi_mts_mts_fft(x1_1d, x2_1d, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))


class TestBiasCorrection:
    """Test bias correction correctness."""

    def test_bias_correction_applied(self):
        """Verify bias correction reduces MI estimate."""
        np.random.seed(61)
        n = 100
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)
        shifts = np.array([0])

        mi_corrected = compute_mi_mts_mts_fft(x1, x2, shifts, biascorrect=True)
        mi_uncorrected = compute_mi_mts_mts_fft(x1, x2, shifts, biascorrect=False)

        # Bias correction should reduce estimate (for independent data)
        # Note: For small n, uncorrected MI has upward bias
        assert mi_corrected[0] <= mi_uncorrected[0]

    def test_matches_mi_gg_bias_correction(self):
        """Bias correction should match mi_gg reference."""
        np.random.seed(62)
        n = 150
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)

        # With bias correction
        mi_fft = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=True)
        mi_ref = mi_gg(x1, x2, biascorrect=True, demeaned=False)
        np.testing.assert_allclose(mi_fft[0], mi_ref, rtol=1e-9)

        # Without bias correction
        mi_fft_no = compute_mi_mts_mts_fft(x1, x2, np.array([0]), biascorrect=False)
        mi_ref_no = mi_gg(x1, x2, biascorrect=False, demeaned=False)
        np.testing.assert_allclose(mi_fft_no[0], mi_ref_no, rtol=1e-9)


class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    def test_very_small_values(self):
        """Test with very small values (near machine precision) - may raise singular covariance error."""
        np.random.seed(63)
        n = 100
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n) * 1e-10
        x2 = np.random.randn(d2, n) * 1e-10
        shifts = np.array([0, 1])

        # With very small values, covariance may become singular due to numerical precision
        # This is expected behavior - the function should either compute MI or raise informative error
        try:
            mi = compute_mi_mts_mts_fft(x1, x2, shifts)
            # If computation succeeds, should not have NaN
            assert np.all(np.isfinite(mi))
        except ValueError as e:
            # Singular covariance is acceptable for numerically challenging inputs
            assert "Singular covariance" in str(e)

    def test_very_large_values(self):
        """Test with very large values."""
        np.random.seed(64)
        n = 100
        d1, d2 = 2, 2

        x1 = np.random.randn(d1, n) * 1e5
        x2 = np.random.randn(d2, n) * 1e5
        shifts = np.array([0, 1])

        # Should handle large values without overflow
        mi = compute_mi_mts_mts_fft(x1, x2, shifts)

        assert np.all(np.isfinite(mi))

    def test_many_shifts(self):
        """Test with many shifts (stress test)."""
        np.random.seed(65)
        n = 200
        d1, d2 = 2, 2
        nsh = 100

        x1 = np.random.randn(d1, n)
        x2 = np.random.randn(d2, n)
        shifts = np.arange(nsh)

        mi = compute_mi_mts_mts_fft(x1, x2, shifts)

        assert mi.shape == (nsh,)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)
