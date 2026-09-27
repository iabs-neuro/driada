"""
Tests for compute_mi_mts_fft function (MultiTimeSeries FFT acceleration).

Tests all supported dimensions (d=1..5), edge cases, and validation.
"""
import pytest
import numpy as np
from driada.information.info_base import compute_mi_mts_fft
from driada.information.gcmi import mi_gg, copnorm


class TestMultiTimeSeriesFFTDimensions:
    """Test compute_mi_mts_fft with different dimensionalities."""

    def test_mts_fft_d1(self):
        """Test with 1D MultiTimeSeries (should work)."""
        n = 100
        # Create 1D multivariate data (d=1, n=100)
        copnorm_x = np.random.randn(1, n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 1, 2, 5, 10])

        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert mi_values.shape == (len(shifts),)
        assert np.all(np.isfinite(mi_values))
        assert np.all(mi_values >= 0)  # MI is non-negative

    def test_mts_fft_d2(self):
        """Test with 2D MultiTimeSeries."""
        n = 100
        # Create 2D multivariate data (d=2, n=100)
        copnorm_x = np.random.randn(2, n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 1, 5])

        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert mi_values.shape == (len(shifts),)
        assert np.all(np.isfinite(mi_values))
        assert np.all(mi_values >= 0)

    def test_mts_fft_d3(self):
        """Test with 3D MultiTimeSeries."""
        n = 100
        # Create 3D multivariate data (d=3, n=100)
        copnorm_x = np.random.randn(3, n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 2])

        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert mi_values.shape == (len(shifts),)
        assert np.all(np.isfinite(mi_values))
        assert np.all(mi_values >= 0)

    @pytest.mark.parametrize("d", [4, 5])
    def test_mts_fft_high_dim(self, d):
        """Test with 4D/5D MultiTimeSeries (extended FFT path)."""
        n = 100
        copnorm_x = np.random.randn(d, n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 1])

        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert mi_values.shape == (len(shifts),)
        assert np.all(np.isfinite(mi_values))
        assert np.all(mi_values >= 0)

    @pytest.mark.parametrize("d", [4, 5])
    def test_mts_fft_high_dim_matches_mi_gg(self, d):
        """Extended FFT path must match the mi_gg loop reference at all shifts."""
        rng = np.random.RandomState(300 + d)
        n = 800
        z = rng.randn(n)
        x = np.vstack([(0.2 + 0.1 * k) * z
                       + np.sqrt(1 - (0.2 + 0.1 * k) ** 2) * rng.randn(n)
                       for k in range(d)])
        copnorm_z = copnorm(z).ravel()
        copnorm_x = copnorm(x)
        shifts = np.array([0, 13, 77, 400])

        mi_fft = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts, biascorrect=True)
        mi_loop = np.array([
            mi_gg(copnorm_z, np.roll(copnorm_x, int(s), axis=1), biascorrect=True)
            for s in shifts
        ])
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("d", [9, 16, 25, 36])
    def test_mts_fft_very_high_dim_matches_mi_gg(self, d):
        """High-d FFT path must stay finite and match mi_gg at all shifts.

        Determinants of well-conditioned high-d covariances underflow any
        absolute threshold, so both the singularity guard and the
        regularization floor must scale with dimension.
        """
        rng = np.random.RandomState(500 + d)
        n = 1500
        z = rng.randn(n)
        coeffs = 0.1 + 0.4 * np.arange(d) / d
        x = np.vstack([a * z + np.sqrt(1 - a ** 2) * rng.randn(n)
                       for a in coeffs])
        copnorm_z = copnorm(z).ravel()
        copnorm_x = copnorm(x)
        shifts = np.array([0, 17, 251, 900])

        mi_fft = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts, biascorrect=True)
        mi_loop = np.array([
            mi_gg(copnorm_z, np.roll(copnorm_x, int(s), axis=1), biascorrect=True)
            for s in shifts
        ])
        np.testing.assert_allclose(mi_fft, mi_loop, rtol=1e-3, atol=5e-5)

    def test_mts_fft_high_dim_singular_raises(self):
        """Linearly dependent dimensions must still be rejected at high d."""
        rng = np.random.RandomState(7)
        n = 400
        x = rng.randn(9, n)
        x[3] = 2 * x[1]
        copnorm_z = rng.randn(n)
        shifts = np.array([0])

        with pytest.raises(ValueError, match="nearly singular"):
            compute_mi_mts_fft(copnorm_z, x, shifts)

    def test_mts_fft_d3_singular_raises(self):
        """The d = 3 closed form must reject linearly dependent dimensions.

        Without the guard the 4x4 determinant collapses for every shift and a
        constant, finite MI is returned instead of an error.
        """
        rng = np.random.RandomState(11)
        n = 400
        x = rng.randn(3, n)
        x[2] = -0.5 * x[0]
        copnorm_z = rng.randn(n)
        shifts = np.arange(0, 40, 10)

        with pytest.raises(ValueError, match="nearly singular"):
            compute_mi_mts_fft(copnorm_z, x, shifts)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_variance_z(self):
        """Test with zero-variance z (should return zeros)."""
        n = 100
        copnorm_x = np.random.randn(2, n)
        copnorm_z = np.ones(n)  # Constant (zero variance)
        shifts = np.array([0, 1, 2])

        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        # Should return zeros for zero-variance input
        assert np.all(mi_values == 0)

    def test_n_equals_2_minimum(self):
        """Test with n=2 (minimum for bias correction)."""
        n = 2
        # Use 1D to avoid singular covariance with n=2
        copnorm_x = np.random.randn(n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0])

        # Should work with n=2 (d=1)
        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert mi_values.shape == (1,)
        assert np.isfinite(mi_values[0])

    def test_n_equals_1_raises(self):
        """Test with n=1 (should raise ValueError)."""
        n = 1
        copnorm_x = np.random.randn(2, n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0])

        # Shape validation catches this first (d=2 > n=1 looks transposed)
        with pytest.raises(ValueError, match="shape looks transposed|at least 2 samples"):
            compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

    def test_transposed_input_raises(self):
        """Test with transposed data (n, d) instead of (d, n) - should raise."""
        n = 100
        d = 2
        # Create transposed data: shape (n, d) instead of (d, n)
        copnorm_x_transposed = np.random.randn(n, d)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 1])

        # Should raise ValueError due to shape validation
        with pytest.raises(ValueError, match="shape looks transposed"):
            compute_mi_mts_fft(copnorm_z, copnorm_x_transposed, shifts)

    def test_1d_array_input(self):
        """Test with 1D array input for copnorm_x (should be reshaped to (1, n))."""
        n = 100
        copnorm_x_1d = np.random.randn(n)  # 1D array
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 1])

        # Should work - auto-reshapes to (1, n)
        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x_1d, shifts)

        assert mi_values.shape == (len(shifts),)
        assert np.all(np.isfinite(mi_values))

    def test_singular_covariance_raises(self):
        """Test with linearly dependent MTS dimensions (singular covariance)."""
        n = 100
        # Create perfectly correlated dimensions (linearly dependent)
        x1 = np.random.randn(n)
        x2 = 2 * x1  # Perfect linear dependence
        copnorm_x = np.vstack([x1, x2])
        copnorm_z = np.random.randn(n)
        shifts = np.array([0])

        # Should raise ValueError due to singular covariance
        with pytest.raises(ValueError, match="nearly singular"):
            compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)


class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    def test_very_small_values(self):
        """Test with very small values (near machine precision)."""
        n = 100
        copnorm_x = np.random.randn(2, n) * 1e-10
        copnorm_z = np.random.randn(n) * 1e-10
        shifts = np.array([0, 1])

        # May have very small MI, but should not crash or produce NaN
        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert np.all(np.isfinite(mi_values))

    def test_very_large_values(self):
        """Test with very large values."""
        n = 100
        copnorm_x = np.random.randn(2, n) * 1e5
        copnorm_z = np.random.randn(n) * 1e5
        shifts = np.array([0, 1])

        # Should handle large values without overflow
        mi_values = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        assert np.all(np.isfinite(mi_values))


class TestBiasCorrection:
    """Test bias correction correctness."""

    def test_bias_correction_d1_matches_univariate(self):
        """For d=1, MTS FFT should match univariate FFT results."""
        from driada.information.info_base import compute_mi_batch_fft

        n = 200
        copnorm_x = np.random.randn(n)
        copnorm_z = np.random.randn(n)
        shifts = np.array([0, 1, 2, 5])

        # Compute with MTS FFT (d=1)
        mi_mts = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)

        # Compute with batch FFT (univariate)
        mi_batch = compute_mi_batch_fft(copnorm_z, copnorm_x, shifts)

        # Should be very close (same bias correction for d=1)
        # Use rtol=1e-9 to account for floating point differences in FFT computation
        np.testing.assert_allclose(mi_mts, mi_batch, rtol=1e-9)

    def test_zero_shift_non_zero_mi(self):
        """Zero shift with correlated data should give non-zero MI."""
        n = 200
        # Create correlated data
        z = np.random.randn(n)
        x1 = z + 0.5 * np.random.randn(n)  # Correlated with z
        x2 = np.random.randn(n)  # Independent
        copnorm_x = np.vstack([x1, x2])
        shifts = np.array([0])

        mi_values = compute_mi_mts_fft(z, copnorm_x, shifts)

        # Should detect correlation at zero shift
        assert mi_values[0] > 0.01  # Non-trivial MI
