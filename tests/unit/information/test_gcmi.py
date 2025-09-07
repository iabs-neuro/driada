"""Consolidated tests for GCMI (Gaussian Copula Mutual Information) functions.

This module combines tests from test_gcmi_functions.py and test_gcmi_jit.py
to avoid duplication and provide comprehensive coverage of GCMI functionality.

ISSUE STATUS SUMMARY (Updated 2025-01-22):
==========================================

1. NEGATIVE JOINT ENTROPY - ✅ NOT A BUG:
   - Differential entropy CAN be negative (unlike discrete entropy)
   - H(X,Y) < max(H(X), H(Y)) is mathematically valid for small noise
   - When X₂ = X₁ + ε with var(ε) < 1/(2πe) ≈ 0.0585, H(ε) < 0
   - Joint entropy H(X₁,X₂) ≈ H(X₁) + H(ε) can be less than H(X₁)
   - Tests updated to reflect this is correct behavior

2. CDC NEGATIVE CMI - ✅ FIXED:
   - Previously produced negative CMI for continuous-discrete-continuous case
   - Fixed by using entropy-based approach: I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
   - All CDC tests now pass with CMI ≥ 0 as required

3. INTERACTION INFORMATION SIGNS - ✅ WORKING:
   - Williams & Beer convention: II < 0 for redundancy, II > 0 for synergy
   - All test cases show correct signs for redundancy and synergy
   - Formula II(X;Y;Z) = I(X;Y|Z) - I(X;Y) verified to match direct calculation

4. NUMERICAL STABILITY - ✅ IMPROVED:
   - Added regularized_cholesky() with adaptive regularization
   - Handles condition numbers up to ~1e17 successfully
   - Extreme cases (ε < 1e-12) still challenging but stable
   - No numerical failures in standard use cases

REMAINING CONSIDERATIONS:
========================
- Extreme correlations with condition numbers > 1e17 need careful handling
- Some edge cases may benefit from alternative entropy estimators
- Documentation should clarify differential vs discrete entropy properties
"""

import numpy as np
import pytest
import time
import warnings
from driada.information.gcmi import (
    demean,
    ent_g,
    mi_model_gd,
    gccmi_ccd,
    ctransform,
    copnorm,
    mi_gg,
    cmi_ggg,
    gcmi_cc,
    _JIT_AVAILABLE,
)
from driada.information.gcmi_jit_utils import (
    ctransform_jit,
    copnorm_jit,
    ctransform_2d_jit,
    copnorm_2d_jit,
    mi_gg_jit,
    cmi_ggg_jit,
    gcmi_cc_jit,
)
from driada.information.info_base import (
    TimeSeries,
    get_mi,
    conditional_mi,
    interaction_information,
)


class TestGCMIDemean:
    """Test class for demean function."""

    def test_demean_single_row(self):
        """Test demeaning a single row."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = demean(data)

        # Check mean is zero
        assert np.abs(np.mean(result)) < 1e-10
        # Check shape preserved
        assert result.shape == data.shape
        # Check specific values
        expected = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_demean_multiple_rows(self):
        """Test demeaning multiple rows independently."""
        data = np.array(
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [-2.0, -1.0, 0.0, 1.0]]
        )
        result = demean(data)

        # Check each row has zero mean
        for i in range(data.shape[0]):
            assert np.abs(np.mean(result[i])) < 1e-10

        # Check shape preserved
        assert result.shape == data.shape

        # Check that rows are demeaned independently
        expected_means = np.array([2.5, 25.0, -0.5])
        for i in range(data.shape[0]):
            np.testing.assert_allclose(
                result[i], data[i] - expected_means[i], rtol=1e-10
            )

    def test_demean_with_zeros(self):
        """Test demeaning data that already has zero mean."""
        data = np.array([[-1.0, 0.0, 1.0]])
        result = demean(data)

        # Should be unchanged
        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_demean_constant_row(self):
        """Test demeaning a constant row."""
        data = np.array([[5.0, 5.0, 5.0, 5.0]])
        result = demean(data)

        # All values should be zero
        np.testing.assert_allclose(result, np.zeros_like(data), rtol=1e-10)

    def test_demean_numerical_stability(self):
        """Test numerical stability with very large values."""
        data = np.array([[1e10, 1e10 + 1, 1e10 + 2]])
        result = demean(data)

        # Check mean is zero (within numerical precision)
        assert np.abs(np.mean(result)) < 1e-4
        # Check relative differences are preserved
        assert np.abs(result[0, 1] - result[0, 0] - 1.0) < 1e-10
        assert np.abs(result[0, 2] - result[0, 0] - 2.0) < 1e-10


class TestGCMIEntropy:
    """Test class for Gaussian entropy calculation."""

    def test_ent_g_univariate_standard_normal(self):
        """Test entropy of standard normal distribution."""
        np.random.seed(42)
        # Standard normal has entropy 0.5 * log2(2*pi*e) ≈ 2.047 bits
        x = np.random.randn(1, 10000)
        h = ent_g(x, biascorrect=True)

        # Theoretical entropy of N(0,1) in bits
        expected = 0.5 * np.log2(2 * np.pi * np.e)
        estimation_tolerance = 0.1  # Allow some estimation error
        assert np.abs(h - expected) < estimation_tolerance

    def test_ent_g_univariate_scaled(self):
        """Test entropy scaling with variance."""
        np.random.seed(42)
        n = 10000

        # H(aX) = H(X) + log2(|a|)
        x1 = np.random.randn(1, n)
        x2 = 2.0 * x1  # Double the scale

        h1 = ent_g(x1, biascorrect=True)
        h2 = ent_g(x2, biascorrect=True)

        # Should differ by log2(2) = 1 bit
        expected_difference = 1.0
        scaling_tolerance = 0.1
        assert np.abs((h2 - h1) - expected_difference) < scaling_tolerance

    def test_ent_g_multivariate_independent(self):
        """Test entropy of independent multivariate Gaussian."""
        np.random.seed(42)
        n = 5000
        d = 3

        # Independent components: H(X,Y,Z) = H(X) + H(Y) + H(Z)
        x = np.random.randn(d, n)
        h_joint = ent_g(x, biascorrect=True)

        h_marginals = sum(ent_g(x[i : i + 1, :], biascorrect=True) for i in range(d))

        # Should be approximately equal for independent components
        independence_tolerance = 0.1
        assert np.abs(h_joint - h_marginals) < independence_tolerance

    def test_ent_g_multivariate_correlated(self):
        """Test entropy reduction with correlation."""
        np.random.seed(42)
        n = 5000

        # Create correlated data
        x1 = np.random.randn(1, n)
        x2 = 0.9 * x1 + 0.1 * np.random.randn(1, n)
        x_corr = np.vstack([x1, x2])

        # Independent data
        x_indep = np.random.randn(2, n)

        h_corr = ent_g(x_corr, biascorrect=True)
        h_indep = ent_g(x_indep, biascorrect=True)

        # Correlated data should have lower entropy
        assert h_corr < h_indep

    def test_ent_g_bias_correction(self):
        """Test effect of bias correction."""
        np.random.seed(42)
        # Small sample size where bias matters
        x = np.random.randn(2, 50)

        h_corrected = ent_g(x, biascorrect=True)
        h_uncorrected = ent_g(x, biascorrect=False)

        # Bias correction changes the estimate - direction depends on sample size
        assert np.abs(h_corrected - h_uncorrected) > 0.001

    def test_ent_g_deterministic(self):
        """Test entropy of nearly deterministic relationship.

        For highly correlated variables X₁, X₂ where X₂ = X₁ + small_noise,
        the joint entropy H(X₁,X₂) ≈ H(X₁) + H(noise) can be less than H(X₁)
        when the noise variance is very small (< 1/(2πe) ≈ 0.058).

        This is mathematically correct for differential entropy, unlike discrete entropy
        where H(X,Y) ≥ max(H(X), H(Y)) always holds.
        """
        np.random.seed(42)
        n = 1000

        # Create data where second component is almost determined by first
        x1 = np.random.randn(1, n)
        noise_level = 0.01  # Very small noise variance = 0.0001
        x2 = x1 + noise_level * np.random.randn(1, n)
        x = np.vstack([x1, x2])

        h_joint = ent_g(x, biascorrect=True)
        h_single = ent_g(x1, biascorrect=True)

        # For differential entropy with small noise variance < 1/(2πe) ≈ 0.058,
        # joint entropy can be less than marginal entropy
        theoretical_noise_entropy = 0.5 * np.log2(2 * np.pi * np.e * noise_level**2)
        expected_joint = h_single + theoretical_noise_entropy

        # Check that GCMI result matches theoretical expectation
        tolerance = 0.1  # Allow small numerical differences
        assert (
            abs(h_joint - expected_joint) < tolerance
        ), f"Joint entropy {h_joint:.3f} should be close to {expected_joint:.3f}"

        # Verify the mathematical relationship holds for this specific case
        # where noise variance << 1/(2πe)
        critical_variance = 1 / (2 * np.pi * np.e)
        assert (
            noise_level**2 < critical_variance
        ), "Test assumes noise variance < 1/(2πe) for negative differential entropy"

    def test_ent_g_input_validation(self):
        """Test input validation."""
        # Test 1D input gets converted to 2D
        n_samples = 100
        x_1d = np.random.randn(n_samples)
        h = ent_g(x_1d)
        assert isinstance(h, float)

        # Test with 3D input (should raise error or numba typing error)
        n_dims1, n_dims2, n_timepoints = 2, 3, 100
        x_3d = np.random.randn(n_dims1, n_dims2, n_timepoints)
        with pytest.raises(
            (ValueError, Exception)
        ):  # Could be ValueError or numba.TypingError
            ent_g(x_3d)


class TestGCMIMutualInformation:
    """Test class for MI calculation functions."""

    def test_mi_model_gd_perfect_dependence(self):
        """Test MI when discrete variable perfectly determines continuous."""
        np.random.seed(42)
        n = 300

        # Create discrete variable
        y = np.array([0, 1, 2] * (n // 3))

        # Continuous variable with different means for each class
        x = np.zeros(n)
        x[y == 0] = np.random.randn(np.sum(y == 0)) + 0.0
        x[y == 1] = np.random.randn(np.sum(y == 1)) + 5.0
        x[y == 2] = np.random.randn(np.sum(y == 2)) + 10.0

        # Reshape for mi_model_gd (samples last axis)
        x_2d = x.reshape(1, -1)

        mi = mi_model_gd(x_2d, y, 3, biascorrect=True, demeaned=False)

        # Should have high MI (close to H(Y) ≈ log2(3) ≈ 1.58 bits)
        assert mi > 1.0

    def test_mi_model_gd_independence(self):
        """Test MI when variables are independent."""
        np.random.seed(42)
        n = 300

        # Independent variables
        y = np.array([0, 1, 2] * (n // 3))
        x = np.random.randn(1, n)

        mi = mi_model_gd(x, y, 3, biascorrect=True, demeaned=False)

        # Should be close to zero
        assert mi < 0.1

    def test_mi_model_gd_multivariate(self):
        """Test MI with multivariate Gaussian."""
        np.random.seed(42)
        n = 400

        # Discrete variable
        y = np.array([0, 1] * (n // 2))

        # 2D Gaussian with different parameters for each class
        x = np.zeros((2, n))
        x[:, y == 0] = np.random.multivariate_normal(
            [0, 0], [[1, 0], [0, 1]], size=np.sum(y == 0)
        ).T
        x[:, y == 1] = np.random.multivariate_normal(
            [3, 3], [[1, 0.5], [0.5, 1]], size=np.sum(y == 1)
        ).T

        mi = mi_model_gd(x, y, 2, biascorrect=True, demeaned=False)

        # Should have substantial MI
        assert mi > 0.5

    def test_mi_model_gd_different_class_sizes(self):
        """Test MI with imbalanced classes."""
        np.random.seed(42)

        # Imbalanced classes
        y = np.array([0] * 100 + [1] * 300)
        x = np.zeros((1, 400))
        x[0, y == 0] = np.random.randn(100) + 0.0
        x[0, y == 1] = np.random.randn(300) + 2.0

        mi = mi_model_gd(x, y, 2, biascorrect=True, demeaned=False)

        # Should still detect dependence
        assert mi > 0.3

    def test_mi_model_gd_many_classes(self):
        """Test MI with many discrete classes."""
        np.random.seed(42)
        n_classes = 5
        n_per_class = 100
        n = n_classes * n_per_class

        # Create discrete variable
        y = np.repeat(np.arange(n_classes), n_per_class)

        # Different mean for each class
        x = np.zeros((1, n))
        for i in range(n_classes):
            x[0, y == i] = np.random.randn(n_per_class) + i * 2.0

        mi = mi_model_gd(x, y, n_classes, biascorrect=True, demeaned=False)

        # Should have high MI
        assert mi > 1.5

    def test_mi_model_gd_demeaned(self):
        """Test MI with pre-demeaned data."""
        np.random.seed(42)
        n = 300

        y = np.array([0, 1] * (n // 2))
        x = np.zeros((1, n))
        x[0, y == 0] = np.random.randn(n // 2) - 1.0
        x[0, y == 1] = np.random.randn(n // 2) + 1.0

        # Demean the data
        x_demeaned = x - np.mean(x)

        mi_normal = mi_model_gd(x, y, 2, biascorrect=True, demeaned=False)
        mi_demeaned = mi_model_gd(x_demeaned, y, 2, biascorrect=True, demeaned=True)

        # Results should be similar (global demean doesn't remove class differences)
        assert np.abs(mi_normal - mi_demeaned) < 0.2

    def test_mi_model_gd_input_validation(self):
        """Test input validation."""
        x = np.random.randn(1, 100)
        y = np.array([0, 1] * 50)

        # Test with wrong y dimension
        y_2d = y.reshape(-1, 1)
        with pytest.raises(ValueError, match="only univariate discrete"):
            mi_model_gd(x, y_2d, 2)

        # Test with non-integer Ym (may raise ValueError or numba TypingError)
        with pytest.raises(
            (ValueError, Exception)
        ):  # Numba can raise various exception types
            mi_model_gd(x, y, 2.5)

        # Test with mismatched sizes
        y_wrong = np.array([0, 1] * 30)  # 60 samples
        with pytest.raises(ValueError, match="number of trials do not match"):
            mi_model_gd(x, y_wrong, 2)


class TestGCMIConditionalMI:
    """Test class for conditional MI calculation."""

    def test_gccmi_ccd_chain_structure(self):
        """Test CMI for chain structure X -> Z -> Y where Z is discrete."""
        np.random.seed(42)
        n = 600

        # Create chain with discrete middle variable
        x = np.random.randn(1, n)

        # Z depends on X (discretized)
        z = np.zeros(n, dtype=int)
        z[x.ravel() < -0.5] = 0
        z[(x.ravel() >= -0.5) & (x.ravel() < 0.5)] = 1
        z[x.ravel() >= 0.5] = 2

        # Y depends on Z
        y = np.zeros((1, n))
        y[0, z == 0] = np.random.randn(np.sum(z == 0)) - 2.0
        y[0, z == 1] = np.random.randn(np.sum(z == 1))
        y[0, z == 2] = np.random.randn(np.sum(z == 2)) + 2.0

        cmi = gccmi_ccd(x, y, z, 3)

        # CMI should be low (X and Y conditionally independent given Z)
        assert cmi < 0.1

    def test_gccmi_ccd_common_cause(self):
        """Test CMI for common cause structure X <- Z -> Y."""
        np.random.seed(42)
        n = 600

        # Z is discrete common cause
        z = np.array([0, 1, 2] * (n // 3))

        # X and Y both depend on Z
        x = np.zeros((1, n))
        y = np.zeros((1, n))

        for zi in range(3):
            mask = z == zi
            n_zi = np.sum(mask)
            # Different correlations for different Z values
            if zi == 0:
                # Independent when Z=0
                x[0, mask] = np.random.randn(n_zi)
                y[0, mask] = np.random.randn(n_zi)
            elif zi == 1:
                # Moderate correlation when Z=1
                x[0, mask] = np.random.randn(n_zi)
                y[0, mask] = 0.5 * x[0, mask] + np.sqrt(0.75) * np.random.randn(n_zi)
            else:
                # Strong correlation when Z=2
                x[0, mask] = np.random.randn(n_zi)
                y[0, mask] = 0.9 * x[0, mask] + np.sqrt(0.19) * np.random.randn(n_zi)

        cmi = gccmi_ccd(x, y, z, 3)

        # Should detect conditional dependence
        assert cmi > 0.2

    def test_gccmi_ccd_multivariate(self):
        """Test CMI with multivariate continuous variables."""
        np.random.seed(42)
        n = 600

        # Discrete conditioning variable
        z = np.array([0, 1] * (n // 2))

        # Multivariate X and Y
        x = np.random.randn(2, n)
        y = np.zeros((2, n))

        # Different relationships for different z
        y[:, z == 0] = 0.7 * x[:, z == 0] + 0.3 * np.random.randn(2, np.sum(z == 0))
        y[:, z == 1] = -0.5 * x[:, z == 1] + 0.5 * np.random.randn(2, np.sum(z == 1))

        cmi = gccmi_ccd(x, y, z, 2)

        # Should detect conditional dependence
        assert cmi > 0.5

    def test_gccmi_ccd_independence_given_z(self):
        """Test when X and Y are independent given Z."""
        np.random.seed(42)
        n = 600

        z = np.array([0, 1, 2] * (n // 3))

        # X and Y are independent within each Z group but have different distributions
        x = np.zeros((1, n))
        y = np.zeros((1, n))

        for zi in range(3):
            mask = z == zi
            n_zi = np.sum(mask)
            # Independent but with different means
            x[0, mask] = np.random.randn(n_zi) + zi
            y[0, mask] = np.random.randn(n_zi) + zi * 2

        cmi = gccmi_ccd(x, y, z, 3)

        # Should be close to zero
        assert cmi < 0.05

    def test_gccmi_ccd_perfect_dependence(self):
        """Test when X determines Y perfectly within each Z group."""
        np.random.seed(42)
        n = 600

        z = np.array([0, 1] * (n // 2))
        x = np.random.randn(1, n)
        y = np.zeros((1, n))

        # Perfect dependence with different relationships
        y[0, z == 0] = x[0, z == 0] + 0.01 * np.random.randn(np.sum(z == 0))
        y[0, z == 1] = -x[0, z == 1] + 0.01 * np.random.randn(np.sum(z == 1))

        cmi = gccmi_ccd(x, y, z, 2)

        # Should be very high
        assert cmi > 2.0

    def test_gccmi_ccd_warning_repeated_values(self):
        """Test warning for repeated values."""
        np.random.seed(42)
        n = 100

        # X has many repeated values
        x = np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0] * (n // 6) + [1.0, 1.0, 1.0, 1.0]])
        y = np.random.randn(1, n)
        z = np.array([0, 1] * (n // 2))

        with pytest.warns(UserWarning, match="more than 10% repeated values"):
            cmi = gccmi_ccd(x, y, z, 2)

    def test_gccmi_ccd_input_validation(self):
        """Test input validation."""
        x = np.random.randn(1, 100)
        y = np.random.randn(1, 100)
        z = np.array([0, 1] * 50)

        # Test with 3D input
        x_3d = np.random.randn(2, 3, 100)
        with pytest.raises(ValueError, match="x and y must be at most 2d"):
            gccmi_ccd(x_3d, y, z, 2)

        # Test with non-integer z
        z_float = np.array([0.0, 1.0] * 50)
        with pytest.raises(ValueError, match="z should be an integer array"):
            gccmi_ccd(x, y, z_float, 2)

        # Test with out of bounds z
        z_bad = np.array([0, 1, 2] * 33 + [3])  # Contains 3 but Zm=2
        with pytest.raises(
            ValueError, match="values of discrete variable z are out of bounds"
        ):
            gccmi_ccd(x, y, z_bad, 2)

        # Test with mismatched sizes
        y_short = np.random.randn(1, 80)
        with pytest.raises(ValueError, match="number of trials do not match"):
            gccmi_ccd(x, y_short, z, 2)


class TestGCMIConsistency:
    """Test consistency between different GCMI implementations."""

    def test_conditional_mi_ccd_consistency(self):
        """Test that conditional_mi gives consistent results with gccmi_ccd."""
        np.random.seed(42)
        n = 600

        # Create test data
        x = np.random.randn(n)
        z = np.array([0, 1, 2] * (n // 3))
        y = np.zeros(n)

        # Y depends on both X and Z
        for zi in range(3):
            mask = z == zi
            y[mask] = (zi + 1) * x[mask] + np.random.randn(np.sum(mask)) * 0.5

        # Create TimeSeries objects
        ts_x = TimeSeries(x, discrete=False)
        ts_y = TimeSeries(y, discrete=False)
        ts_z = TimeSeries(z, discrete=True)

        # Calculate CMI using conditional_mi
        cmi_info_base = conditional_mi(ts_x, ts_y, ts_z)

        # Calculate directly using gccmi_ccd
        cmi_direct = gccmi_ccd(x.reshape(1, -1), y.reshape(1, -1), z, 3)

        # Should be very close
        assert np.abs(cmi_info_base - cmi_direct) < 0.01

    def test_mi_model_gd_consistency(self):
        """Test that get_mi gives consistent results with mi_model_gd."""
        np.random.seed(42)
        n = 400
        n_classes = 4

        # Create test data
        y = np.array([0, 1, 2, 3] * (n // n_classes))
        x = np.zeros(n)
        class_separation = 1.5

        # X depends on Y
        for yi in range(n_classes):
            x[y == yi] = np.random.randn(np.sum(y == yi)) + yi * class_separation

        # Create TimeSeries objects
        ts_x = TimeSeries(x, discrete=False)
        ts_y = TimeSeries(y, discrete=True)

        # Calculate MI using get_mi
        mi_info_base = get_mi(ts_x, ts_y)

        # Calculate directly using mi_model_gd (note: x needs copula normalization)
        from driada.information.gcmi import copnorm

        x_norm = copnorm(x.reshape(1, -1))
        mi_direct = mi_model_gd(x_norm, y, n_classes, biascorrect=True, demeaned=True)

        # Should be reasonably close (allow for copula transform differences)
        # The tolerance depends on the estimation differences between methods
        max_tolerance = max(
            0.3, 0.3 * mi_info_base
        )  # 30% or 0.3 bits, whichever is larger
        assert np.abs(mi_info_base - mi_direct) < max_tolerance


class TestGCMIStability:
    """Test numerical stability and edge cases."""

    def test_numerical_stability(self):
        """Test numerical stability of GCMI functions."""
        np.random.seed(42)

        # Test with very small variance
        x_small_var = np.ones((1, 100)) + np.random.randn(1, 100) * 1e-8
        h = ent_g(x_small_var)
        assert np.isfinite(h)

        # Test with large values
        x_large = np.random.randn(2, 100) * 1e6
        h = ent_g(x_large)
        assert np.isfinite(h)

        # Test MI with near-constant data
        y = np.array([0, 1] * 50)
        x_const = np.ones((1, 100)) + np.random.randn(1, 100) * 1e-10
        mi = mi_model_gd(x_const, y, 2)
        assert np.isfinite(mi)
        assert mi >= 0  # MI should be non-negative

    def test_edge_cases(self):
        """Test edge cases for GCMI functions."""
        np.random.seed(42)

        # Test with minimum sample size
        min_samples = 10
        x_min = np.random.randn(1, min_samples)
        y_min = np.array([0, 1] * (min_samples // 2))

        # Should still work but with less accuracy
        mi = mi_model_gd(x_min, y_min, 2)
        assert np.isfinite(mi)
        assert mi >= 0

        # Test with single class (edge case for mi_model_gd)
        n_samples = 100
        y_single = np.zeros(n_samples, dtype=int)
        x_single = np.random.randn(1, n_samples)

        # Should give near-zero MI (no information in constant y)
        mi = mi_model_gd(x_single, y_single, 1)
        mi_tolerance = 0.05  # Allow small numerical errors
        assert np.abs(mi) < mi_tolerance

        # Test gccmi_ccd with single z value
        z_single = np.zeros(n_samples, dtype=int)
        x = np.random.randn(1, n_samples)
        correlation = 0.5
        noise_var = 1 - correlation**2
        y = correlation * x + np.sqrt(noise_var) * np.random.randn(1, n_samples)

        cmi = gccmi_ccd(x, y, z_single, 1)
        # Should equal unconditional MI
        from driada.information.gcmi import gcmi_cc

        mi_uncond = gcmi_cc(x, y)
        mi_comparison_tolerance = 0.05  # Allow estimation differences
        assert np.abs(cmi - mi_uncond) < mi_comparison_tolerance


@pytest.mark.skipif(not _JIT_AVAILABLE, reason="JIT utilities not available")
class TestGCMIJIT:
    """Test JIT-compiled GCMI functions."""

    def test_ctransform_correctness_1d(self):
        """Test that JIT ctransform produces same results as original."""
        np.random.seed(42)

        for size in [10, 100, 1000]:
            data = np.random.randn(size)

            # Compare results
            result_orig = ctransform(data).ravel()
            result_jit = ctransform_jit(data)

            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)

    def test_ctransform_correctness_2d(self):
        """Test that JIT ctransform works correctly for 2D arrays."""
        np.random.seed(42)

        for shape in [(2, 100), (5, 50), (10, 200)]:
            data = np.random.randn(*shape)

            # Compare results
            result_orig = ctransform(data)
            result_jit = ctransform_2d_jit(data)

            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)

    def test_copnorm_correctness_1d(self):
        """Test that JIT copnorm produces similar results as original."""
        np.random.seed(42)

        for size in [10, 100, 1000]:
            data = np.random.randn(size)

            # Compare results - copnorm uses approximations so allow more tolerance
            result_orig = copnorm(data).ravel()
            result_jit = copnorm_jit(data)

            # Check correlation is very high (approximation is good)
            correlation = np.corrcoef(result_orig, result_jit)[0, 1]
            assert correlation > 0.999, f"Correlation {correlation} too low"

            # Check values are close
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-3, atol=1e-3)

    def test_copnorm_correctness_2d(self):
        """Test that JIT copnorm works correctly for 2D arrays."""
        np.random.seed(42)

        for shape in [(2, 100), (5, 50)]:
            data = np.random.randn(*shape)

            # Compare results
            result_orig = copnorm(data)
            result_jit = copnorm_2d_jit(data)

            # Check correlation is very high for each row
            for i in range(shape[0]):
                correlation = np.corrcoef(result_orig[i], result_jit[i])[0, 1]
                assert correlation > 0.999, f"Row {i} correlation {correlation} too low"

    def test_edge_cases(self):
        """Test edge cases for JIT functions."""
        # Test with constant array
        const_data = np.ones(10)
        result_ct = ctransform_jit(const_data)
        # For constant array, the copula transform breaks ties by index
        # So we get evenly spaced values between 0 and 1
        assert np.all(result_ct > 0) and np.all(result_ct < 1)

        # Test with sorted array
        sorted_data = np.arange(10).astype(float)
        result_ct = ctransform_jit(sorted_data)
        expected = (np.arange(10) + 1) / 11.0
        np.testing.assert_allclose(result_ct, expected)

    def test_performance_improvement(self):
        """Benchmark JIT vs regular implementations."""
        np.random.seed(42)
        sizes = [100, 1000]

        for size in sizes:
            data = np.random.randn(size)

            # Warm up JIT
            _ = ctransform_jit(data)
            _ = copnorm_jit(data)

            # Time ctransform
            n_iter = 100
            start = time.time()
            for _ in range(n_iter):
                _ = ctransform(data)
            time_regular = time.time() - start

            start = time.time()
            for _ in range(n_iter):
                _ = ctransform_jit(data)
            time_jit = time.time() - start

            # JIT version should not be significantly slower
            # (relaxed constraint since performance varies)
            assert time_jit < time_regular * 2.0, f"JIT too slow for size {size}"

    def test_integration_with_gcmi(self):
        """Test that gcmi functions automatically use JIT versions."""
        np.random.seed(42)

        # Test 1D input
        data_1d = np.random.randn(100)
        result = ctransform(data_1d)
        assert result.shape == (1, 100)  # Should be 2D

        # Test 2D input
        data_2d = np.random.randn(3, 100)
        result = ctransform(data_2d)
        assert result.shape == (3, 100)

        # Test copnorm
        result = copnorm(data_1d)
        assert result.shape == (1, 100)

        result = copnorm(data_2d)
        assert result.shape == (3, 100)

    def test_mi_gg_jit_correctness(self):
        """Test that JIT mi_gg produces same results as original."""
        np.random.seed(42)

        for shape in [(1, 100), (2, 100), (3, 50)]:
            x = np.random.randn(*shape)
            y = np.random.randn(*shape)

            # Test with and without bias correction
            for biascorrect in [True, False]:
                result_orig = mi_gg(x, y, biascorrect=biascorrect)
                result_jit = mi_gg_jit(x, y, biascorrect=biascorrect)

                np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9)

    def test_cmi_ggg_jit_correctness(self):
        """Test that JIT cmi_ggg produces same results as original."""
        np.random.seed(42)

        for shape in [(1, 100), (2, 50)]:
            x = np.random.randn(*shape)
            y = np.random.randn(*shape)
            z = np.random.randn(*shape)

            # Test CMI computation
            result_orig = cmi_ggg(x, y, z, biascorrect=True)
            result_jit = cmi_ggg_jit(x, y, z, biascorrect=True)

            # CMI can be negative, so check absolute difference
            np.testing.assert_allclose(result_orig, result_jit, rtol=1e-9, atol=1e-9)

    def test_gcmi_cc_jit_correctness(self):
        """Test that JIT gcmi_cc produces correct results."""
        np.random.seed(42)

        for shape in [(1, 100), (2, 100)]:
            x = np.random.randn(*shape)
            y = np.random.randn(*shape)

            # Add correlation
            if shape[0] == 2:
                y[0] = 0.7 * x[0] + 0.3 * np.random.randn(shape[1])
            else:
                y = 0.7 * x + 0.3 * np.random.randn(*shape)

            result_orig = gcmi_cc(x, y)
            result_jit = gcmi_cc_jit(x, y)

            # GCMI uses approximations, so allow some tolerance
            assert (
                abs(result_orig - result_jit) < 0.01
            ), f"Difference {abs(result_orig - result_jit)} too large"

    def test_interaction_information_jit_consistency(self):
        """Test that interaction information produces consistent results with JIT functions."""
        # Temporarily disable JIT to get reference value
        import src.driada.information.gcmi as gcmi_module

        original_jit = gcmi_module._JIT_AVAILABLE

        np.random.seed(42)

        # Create test data
        n = 200
        x = np.random.randn(n)
        y = 0.8 * x + 0.2 * np.random.randn(n)
        z = 0.5 * x + 0.5 * y + 0.3 * np.random.randn(n)

        ts_x = TimeSeries(x, discrete=False)
        ts_y = TimeSeries(y, discrete=False)
        ts_z = TimeSeries(z, discrete=False)

        # Get reference without JIT
        gcmi_module._JIT_AVAILABLE = False
        ii_ref = interaction_information(ts_x, ts_y, ts_z)
        mi_xy_ref = get_mi(ts_x, ts_y)
        cmi_ref = conditional_mi(ts_x, ts_y, ts_z)

        # Get results with JIT
        gcmi_module._JIT_AVAILABLE = True
        ii_jit = interaction_information(ts_x, ts_y, ts_z)
        mi_xy_jit = get_mi(ts_x, ts_y)
        cmi_jit = conditional_mi(ts_x, ts_y, ts_z)

        # Restore original state
        gcmi_module._JIT_AVAILABLE = original_jit

        # Check consistency
        np.testing.assert_allclose(ii_ref, ii_jit, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(mi_xy_ref, mi_xy_jit, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(cmi_ref, cmi_jit, rtol=1e-9, atol=1e-9)

    def test_conditional_mi_integration(self):
        """Test that conditional MI works with JIT functions."""
        np.random.seed(42)

        # Create chain: X -> Y -> Z
        n = 200
        x = np.random.randn(n)
        y = 0.8 * x + 0.2 * np.random.randn(n)
        z = 0.8 * y + 0.2 * np.random.randn(n)

        ts_x = TimeSeries(x, discrete=False)
        ts_y = TimeSeries(y, discrete=False)
        ts_z = TimeSeries(z, discrete=False)

        # I(X;Z|Y) should be close to 0 for a chain
        cmi = conditional_mi(ts_x, ts_z, ts_y)
        assert cmi < 0.1, f"CMI too high for chain structure: {cmi}"


class TestGCMIEdgeCases:
    """Test edge cases for GCMI implementation."""
    
    def test_single_sample_class(self):
        """Test that GCMI handles classes with only 1 sample correctly."""
        # Create data where one class has only 1 sample
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        y = np.array([0, 0, 1, 1, 2, 2, 3])  # Class 3 has only 1 sample
        
        # Should issue warning but not crash
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mi = mi_model_gd(x, y)
            assert mi >= 0  # MI should be non-negative
            assert len(w) == 1  # Should have warning
            assert "Class 3 has only 1 sample" in str(w[0].message)
    
    def test_empty_class(self):
        """Test that GCMI handles empty classes correctly."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]])  
        y = np.array([0, 0, 1, 1])
        
        # Specify Ym=3 but only have classes 0 and 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mi = mi_model_gd(x, y, Ym=3)
            assert mi >= 0
            assert len(w) == 1
            assert "Class 2 has no samples" in str(w[0].message)
    
    def test_multiple_single_sample_classes(self):
        """Test GCMI with multiple classes having single samples."""
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y = np.array([0, 1, 2, 3, 4])  # Each class has only 1 sample
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mi = mi_model_gd(x, y)
            assert mi >= 0
            # Should have warnings for all 5 classes
            assert len(w) == 5
            for i in range(5):
                assert f"Class {i} has only 1 sample" in str(w[i].message)
    
    def test_bias_correction_with_small_samples(self):
        """Test bias correction doesn't crash with small sample sizes."""
        # Minimum viable case: 2 samples per class
        x = np.array([[1.0, 1.2, 2.0, 2.2]])
        y = np.array([0, 0, 1, 1])
        
        # Both with and without bias correction should work
        mi_with_bias = mi_model_gd(x, y, biascorrect=True)
        mi_without_bias = mi_model_gd(x, y, biascorrect=False)
        
        assert mi_with_bias >= 0
        assert mi_without_bias >= 0


def test_jit_availability():
    """Test that JIT imports are working."""
    from driada.information import gcmi

    assert hasattr(gcmi, "_JIT_AVAILABLE")
    assert gcmi._JIT_AVAILABLE == True  # Should be available in test environment
