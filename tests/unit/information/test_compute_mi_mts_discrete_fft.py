"""
Tests for compute_mi_mts_discrete_fft function (MTS vs discrete FFT acceleration).

Following the pattern of test_compute_mi_mts_mts_fft.py with comprehensive
equivalence tests between FFT and loop implementations.
"""
import pytest
import numpy as np
from driada.information.info_base import compute_mi_mts_discrete_fft
from driada.information.gcmi import mi_model_gd, copnorm


class TestMTSDiscreteFFTCorrectness:
    """Test correctness against mi_model_gd reference implementation."""

    def test_matches_mi_model_gd_at_zero_shift_d2_ym3(self):
        """Verify FFT matches mi_model_gd reference at shift=0 for d=2, Ym=3."""
        np.random.seed(42)
        n, d, Ym = 150, 2, 3

        # Use copnorm'd data
        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        # Compute with FFT at shift=0
        mi_fft = compute_mi_mts_discrete_fft(
            mts, discrete, np.array([0]), biascorrect=True
        )[0]

        # Compute with mi_model_gd reference (demeaned=True to match FFT)
        mi_ref = mi_model_gd(mts, discrete, Ym, biascorrect=True, demeaned=True)

        # Should match closely
        np.testing.assert_allclose(mi_fft, mi_ref, rtol=1e-8)

    def test_matches_mi_model_gd_at_zero_shift_d3_ym2(self):
        """Verify FFT matches mi_model_gd reference at shift=0 for d=3, Ym=2."""
        np.random.seed(43)
        n, d, Ym = 120, 3, 2

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]), True)[0]
        mi_ref = mi_model_gd(mts, discrete, Ym, biascorrect=True, demeaned=True)

        np.testing.assert_allclose(mi_fft, mi_ref, rtol=1e-8)

    def test_fft_matches_loop_all_shifts_d2_ym3(self):
        """CRITICAL: Verify FFT matches loop (mi_model_gd with np.roll) at ALL shifts (d=2, Ym=3)."""
        np.random.seed(99)
        n, d, Ym = 150, 2, 3
        nsh = 20  # Test multiple shifts

        # Generate copnorm'd data
        mts_raw = np.random.randn(d, n)
        mts = copnorm(mts_raw)
        discrete = np.random.randint(0, Ym, n)
        shifts = np.arange(nsh)

        # Compute with FFT
        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, shifts, biascorrect=True)

        # Compute with loop (ground truth) - shift the discrete variable
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            discrete_shifted = np.roll(discrete, int(s))
            mi_loop[i] = mi_model_gd(mts, discrete_shifted, Ym, True, True)

        # FFT must match loop at ALL shifts (this is the gold standard)
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-7, atol=1e-10)

    def test_fft_matches_loop_all_shifts_d3_ym2(self):
        """CRITICAL: Verify FFT matches loop at ALL shifts (d=3, Ym=2)."""
        np.random.seed(100)
        n, d, Ym = 120, 3, 2
        nsh = 15

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.arange(nsh)

        # FFT
        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, shifts, True)

        # Loop
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            discrete_shifted = np.roll(discrete, int(s))
            mi_loop[i] = mi_model_gd(mts, discrete_shifted, Ym, True, True)

        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-7, atol=1e-10)

    def test_fft_matches_loop_all_shifts_d1_ym5(self):
        """CRITICAL: Verify FFT matches loop at ALL shifts (d=1, Ym=5)."""
        np.random.seed(101)
        n, d, Ym = 200, 1, 5
        nsh = 25

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.arange(nsh)

        # FFT
        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, shifts, True)

        # Loop
        mi_loop = np.zeros(nsh)
        for i, s in enumerate(shifts):
            discrete_shifted = np.roll(discrete, int(s))
            mi_loop[i] = mi_model_gd(mts, discrete_shifted, Ym, True, True)

        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-7, atol=1e-10)

    def test_correctness_multiple_shifts(self):
        """Verify FFT produces valid MI values across multiple shifts."""
        np.random.seed(44)
        n, d, Ym = 200, 2, 3
        nsh = 30

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.arange(nsh)

        # Compute with FFT
        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, shifts, True)

        # All MI values should be valid
        assert mi_fft.shape == (nsh,)
        assert np.all(np.isfinite(mi_fft))
        assert np.all(mi_fft >= 0)  # MI is non-negative

        # Verify shift=0 matches reference
        mi_ref_0 = mi_model_gd(mts, discrete, Ym, True, True)
        np.testing.assert_allclose(mi_fft[0], mi_ref_0, rtol=1e-8)

    def test_independent_variables_near_zero(self):
        """Independent MTS and discrete should give MI ≈ 0."""
        np.random.seed(45)
        n, d, Ym = 500, 2, 3

        # Create independent MTS and discrete
        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        mi = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]), True)

        # Should be near zero for independent variables
        assert mi[0] < 0.05  # Small threshold for statistical fluctuations

    def test_correlated_variables_positive_mi(self):
        """Correlated MTS and discrete should give MI > 0."""
        np.random.seed(46)
        n, d, Ym = 300, 2, 2

        # Create discrete variable
        discrete = np.random.randint(0, Ym, n)

        # Create MTS that depends on discrete
        mts = np.zeros((d, n))
        for i in range(Ym):
            mask = discrete == i
            # Each class has different mean
            mts[0, mask] = np.random.randn(np.sum(mask)) + i * 2
            mts[1, mask] = np.random.randn(np.sum(mask))
        mts = copnorm(mts)

        mi = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]), True)

        # Should detect correlation
        assert mi[0] > 0.1  # Non-trivial MI


class TestMTSDiscreteFFTDimensions:
    """Test various MTS dimensionalities."""

    def test_d1_ym2(self):
        """d=1 (univariate MTS), Ym=2."""
        np.random.seed(47)
        n, d, Ym = 100, 1, 2

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.array([0, 1, 5])

        mi = compute_mi_mts_discrete_fft(mts, discrete, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_d2_ym3(self):
        """d=2 (2D position), Ym=3 - most common case."""
        np.random.seed(48)
        n, d, Ym = 100, 2, 3

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.array([0, 2])

        mi = compute_mi_mts_discrete_fft(mts, discrete, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_d3_ym5(self):
        """d=3 (3D position), Ym=5 - edge of limit."""
        np.random.seed(49)
        n, d, Ym = 100, 3, 5

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.array([0, 1])

        mi = compute_mi_mts_discrete_fft(mts, discrete, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_d4_raises(self):
        """d=4 should raise NotImplementedError."""
        np.random.seed(50)
        n, d, Ym = 100, 4, 2

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        with pytest.raises(NotImplementedError, match="d > 3 is not implemented"):
            compute_mi_mts_discrete_fft(mts, discrete, np.array([0]))


class TestClassCounts:
    """Test various discrete class counts."""

    def test_binary_ym2(self):
        """Binary discrete (Ym=2)."""
        np.random.seed(51)
        n, d, Ym = 100, 2, 2

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        mi = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]))

        assert np.isfinite(mi[0])
        assert mi[0] >= 0

    def test_multiclass_ym10(self):
        """Many classes (Ym=10)."""
        np.random.seed(52)
        n, d, Ym = 500, 2, 10  # Need more samples for 10 classes

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        mi = compute_mi_mts_discrete_fft(mts, discrete, np.array([0, 1]))

        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_rare_classes_with_warning(self):
        """Some classes have < d+1 samples (should skip + warn)."""
        np.random.seed(53)
        n, d, Ym = 100, 2, 5

        # Create imbalanced: class 4 has only 1 sample
        discrete = np.concatenate([
            np.full(25, 0), np.full(25, 1), np.full(25, 2),
            np.full(24, 3), np.array([4])
        ])
        mts = copnorm(np.random.randn(d, n))

        with pytest.warns(RuntimeWarning, match="Class 4.*1 samples"):
            mi = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]))
            assert mi[0] >= 0  # Should complete successfully


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_transposed_input_raises(self):
        """Input (n, d) instead of (d, n) should raise."""
        np.random.seed(54)
        n, d, Ym = 100, 2, 3

        mts_transposed = np.random.randn(n, d)  # Wrong orientation
        discrete = np.random.randint(0, Ym, n)

        with pytest.raises(ValueError, match="shape looks transposed"):
            compute_mi_mts_discrete_fft(mts_transposed, discrete, np.array([0]))

    def test_singular_covariance_handled(self):
        """Linearly dependent MTS dimensions use regularization."""
        np.random.seed(55)
        n, Ym = 100, 2

        # Create near-singular covariance
        x = np.random.randn(n)
        mts_raw = np.vstack([x, x + 1e-8*np.random.randn(n)])
        mts = copnorm(mts_raw)
        discrete = np.random.randint(0, Ym, n)

        # Should complete with regularization (from regularized_cholesky)
        mi = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]))
        assert np.isfinite(mi[0])
        assert mi[0] >= 0

    def test_1d_array_auto_reshape(self):
        """Test with 1D array (should auto-reshape to (1, n))."""
        np.random.seed(56)
        n, Ym = 100, 3

        mts_1d = copnorm(np.random.randn(n))  # 1D array
        discrete = np.random.randint(0, Ym, n)
        shifts = np.array([0, 1])

        mi = compute_mi_mts_discrete_fft(mts_1d, discrete, shifts)

        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))


class TestBiasCorrection:
    """Test bias correction correctness."""

    def test_bias_correction_applied(self):
        """Verify bias correction reduces MI estimate."""
        np.random.seed(57)
        n, d, Ym = 100, 2, 3

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.array([0])

        mi_corrected = compute_mi_mts_discrete_fft(mts, discrete, shifts, True)
        mi_uncorrected = compute_mi_mts_discrete_fft(mts, discrete, shifts, False)

        # Bias correction should reduce estimate
        assert mi_corrected[0] <= mi_uncorrected[0]

    def test_matches_mi_model_gd_bias_correction(self):
        """Bias correction should match mi_model_gd reference."""
        np.random.seed(58)
        n, d, Ym = 150, 2, 3

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)

        # With bias correction
        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]), True)
        mi_ref = mi_model_gd(mts, discrete, Ym, True, True)
        np.testing.assert_allclose(mi_fft[0], mi_ref, rtol=1e-9)

        # Without bias correction
        mi_fft_no = compute_mi_mts_discrete_fft(mts, discrete, np.array([0]), False)
        mi_ref_no = mi_model_gd(mts, discrete, Ym, False, True)
        np.testing.assert_allclose(mi_fft_no[0], mi_ref_no, rtol=1e-9)


class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    def test_very_small_values(self):
        """Test with very small values."""
        np.random.seed(59)
        n, d, Ym = 100, 2, 3

        mts = copnorm(np.random.randn(d, n) * 1e-10)
        discrete = np.random.randint(0, Ym, n)
        shifts = np.array([0, 1])

        # Should handle small values
        mi = compute_mi_mts_discrete_fft(mts, discrete, shifts)
        assert np.all(np.isfinite(mi))

    def test_many_shifts(self):
        """Test with many shifts (stress test)."""
        np.random.seed(60)
        n, d, Ym = 200, 2, 3
        nsh = 100

        mts = copnorm(np.random.randn(d, n))
        discrete = np.random.randint(0, Ym, n)
        shifts = np.arange(nsh)

        mi = compute_mi_mts_discrete_fft(mts, discrete, shifts)

        assert mi.shape == (nsh,)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)
