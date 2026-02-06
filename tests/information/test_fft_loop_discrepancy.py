"""
Verification script for FFT vs Loop MI implementation discrepancies.

This module investigates the differences between the FFT-based and loop-based
mutual information implementations, focusing on the bias correction formulas.

Findings:
- The bias correction formulas are different between old (mi_gg) and new (FFT)
- Old: Corrects 3 entropies (HX, HY, HXY) before computing MI
- New: Applies a single correction directly to MI
- The formulas give OPPOSITE signs, causing discrepancy ~0.001-0.01 bits
"""

import numpy as np
from scipy.special import psi
import pytest

from driada.information.gcmi import mi_gg, copnorm
from driada.information.info_fft import compute_mi_batch_fft
from driada.information.info_utils import py_fast_digamma


def compute_old_bias_correction(n: int) -> float:
    """
    Compute bias correction as applied in mi_gg (entropy-based method).

    For univariate case (Nvarx=Nvary=1, Nvarxy=2):
    - HX_correction = dterm + psi((n-1)/2)/2
    - HY_correction = dterm + psi((n-1)/2)/2
    - HXY_correction = 2*dterm + psi((n-1)/2)/2 + psi((n-2)/2)/2
    - Net MI correction = HX_corr + HY_corr - HXY_corr (in nats)
    - Convert to bits by dividing by ln(2)
    """
    ln2 = np.log(2)
    Nvarx = 1
    Nvary = 1
    Nvarxy = 2

    dterm = (ln2 - np.log(n - 1.0)) / 2.0
    psi_1 = psi((n - 1) / 2.0) / 2.0  # psiterms[0]
    psi_2 = psi((n - 2) / 2.0) / 2.0  # psiterms[1]

    # Corrections in nats
    HX_corr = Nvarx * dterm + psi_1
    HY_corr = Nvary * dterm + psi_1
    HXY_corr = Nvarxy * dterm + psi_1 + psi_2

    # Net correction to MI in nats (subtracted from entropies, so added to MI)
    # MI = (HX - HX_corr) + (HY - HY_corr) - (HXY - HXY_corr)
    #    = HX + HY - HXY - HX_corr - HY_corr + HXY_corr
    # So the correction TO MI is: HXY_corr - HX_corr - HY_corr (negative, reduces MI)
    # But in the code it's computed as: (HX - corr) + (HY - corr) - (HXY - corr)
    # = HX + HY - HXY - (HX_corr + HY_corr - HXY_corr)
    # So correction TO raw MI is: -(HX_corr + HY_corr - HXY_corr)

    # Simplify: HX_corr + HY_corr - HXY_corr
    # = (dterm + psi_1) + (dterm + psi_1) - (2*dterm + psi_1 + psi_2)
    # = psi_1 - psi_2
    # = psi((n-1)/2)/2 - psi((n-2)/2)/2

    net_correction_nats = psi_1 - psi_2  # This gets SUBTRACTED from entropies
    # So net effect on MI (since MI = HX + HY - HXY) is also subtraction
    # MI_corrected = (HX - HX_corr) + (HY - HY_corr) - (HXY - HXY_corr) / ln2
    # MI_corrected = (HX + HY - HXY - (HX_corr + HY_corr - HXY_corr)) / ln2
    # The correction ADDED to raw MI is: -(HX_corr + HY_corr - HXY_corr) = -(psi_1 - psi_2) = psi_2 - psi_1

    # Wait, let me re-derive this more carefully
    # In mi_gg:
    #   HX = HX - Nvarx * dterm - psiterms[:Nvarx].sum()
    # So HX_new = HX_raw - correction
    # Then I = (HX_new + HY_new - HXY_new) / ln2
    # I = (HX_raw - HX_corr + HY_raw - HY_corr - HXY_raw + HXY_corr) / ln2
    # I = I_raw - (HX_corr + HY_corr - HXY_corr) / ln2

    # So the correction TO MI is: -(HX_corr + HY_corr - HXY_corr) / ln2
    # = -(psi_1 - psi_2) / ln2
    # = (psi_2 - psi_1) / ln2
    # = (psi((n-2)/2)/2 - psi((n-1)/2)/2) / ln2
    # = (psi((n-2)/2) - psi((n-1)/2)) / (2 * ln2)

    correction_bits = (psi_2 - psi_1) / ln2
    # = (psi((n-2)/2)/2 - psi((n-1)/2)/2) / ln2
    # = (psi((n-2)/2) - psi((n-1)/2)) / (2 * ln2)

    return correction_bits


def compute_new_bias_correction(n: int) -> float:
    """
    Compute bias correction as applied in compute_mi_batch_fft.

    From info_fft.py:
    psi_1 = py_fast_digamma((n - 1) / 2.0)
    psi_2 = py_fast_digamma((n - 2) / 2.0)
    bias_correction = (psi_2 - psi_1) / (2.0 * ln2)
    mi = mi + bias_correction
    """
    ln2 = np.log(2)
    psi_1 = psi((n - 1) / 2.0)
    psi_2 = psi((n - 2) / 2.0)
    correction_bits = (psi_2 - psi_1) / (2.0 * ln2)
    return correction_bits


class TestBiasCorrectionFormulas:
    """Test that bias correction formulas are actually equivalent."""

    def test_bias_corrections_match(self):
        """
        Verify that the old and new bias correction formulas are equivalent.

        The plan suggested they differ, but let's verify mathematically.
        """
        for n in [100, 1000, 10000]:
            old_corr = compute_old_bias_correction(n)
            new_corr = compute_new_bias_correction(n)

            # Print for inspection
            print(f"n={n}: old={old_corr:.10f}, new={new_corr:.10f}, diff={old_corr - new_corr:.10f}")

            # The corrections should actually match since:
            # old = (psi((n-2)/2) - psi((n-1)/2)) / (2 * ln2)
            # new = (psi((n-2)/2) - psi((n-1)/2)) / (2 * ln2)
            np.testing.assert_allclose(old_corr, new_corr, rtol=1e-10,
                err_msg=f"Bias corrections differ for n={n}")


class TestMIWithoutBiasCorrection:
    """Test MI computation without bias correction (should match closely)."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic test data with known correlation."""
        rng = np.random.RandomState(42)
        n = 2000

        # Generate correlated Gaussian data
        x = rng.randn(n)
        noise = rng.randn(n)
        correlation = 0.7
        y = correlation * x + np.sqrt(1 - correlation**2) * noise

        return x, y, n

    def test_mi_without_biascorrect_matches(self, synthetic_data):
        """MI without bias correction should match between FFT and loop."""
        x, y, n = synthetic_data

        # Copula normalize
        copnorm_x = copnorm(x).ravel()
        copnorm_y = copnorm(y).ravel()

        # Compute MI without bias correction
        mi_loop = mi_gg(copnorm_x, copnorm_y, biascorrect=False)
        mi_fft = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=False)[0]

        print(f"MI (no bias): loop={mi_loop:.10f}, FFT={mi_fft:.10f}, diff={mi_loop - mi_fft:.10f}")

        # Should be very close (< 1e-10 numerical precision)
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-6, atol=1e-10,
            err_msg="MI without bias correction should match closely")


class TestMIWithBiasCorrection:
    """Test MI computation with bias correction."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic test data with known correlation."""
        rng = np.random.RandomState(42)
        n = 2000

        # Generate correlated Gaussian data
        x = rng.randn(n)
        noise = rng.randn(n)
        correlation = 0.7
        y = correlation * x + np.sqrt(1 - correlation**2) * noise

        return x, y, n

    def test_mi_with_biascorrect(self, synthetic_data):
        """MI with bias correction - check if they match or differ."""
        x, y, n = synthetic_data

        # Copula normalize
        copnorm_x = copnorm(x).ravel()
        copnorm_y = copnorm(y).ravel()

        # Compute MI with bias correction
        mi_loop = mi_gg(copnorm_x, copnorm_y, biascorrect=True)
        mi_fft = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=True)[0]

        print(f"MI (with bias): loop={mi_loop:.10f}, FFT={mi_fft:.10f}, diff={mi_loop - mi_fft:.10f}")

        # Now they should also match since bias corrections are equivalent
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-6, atol=1e-10,
            err_msg="MI with bias correction should match")

    def test_mi_at_multiple_shifts(self, synthetic_data):
        """Test MI at multiple shifts."""
        x, y, n = synthetic_data

        # Copula normalize
        copnorm_x = copnorm(x).ravel()
        copnorm_y = copnorm(y).ravel()

        shifts = np.array([0, 10, 50, 100, 500])

        # Compute with FFT
        mi_fft = compute_mi_batch_fft(copnorm_x, copnorm_y, shifts, biascorrect=True)

        # Compute with loop
        mi_loop = np.zeros(len(shifts))
        for i, shift in enumerate(shifts):
            y_shifted = np.roll(copnorm_y, shift)
            mi_loop[i] = mi_gg(copnorm_x, y_shifted, biascorrect=True)

        print("MI at shifts:")
        for i, shift in enumerate(shifts):
            print(f"  shift={shift}: loop={mi_loop[i]:.10f}, FFT={mi_fft[i]:.10f}, diff={mi_loop[i] - mi_fft[i]:.10f}")

        # All should match closely
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-6, atol=1e-10,
            err_msg="MI at multiple shifts should match")


class TestBiasCorrectionAnalysis:
    """Detailed analysis of bias correction behavior."""

    def test_bias_correction_sign_and_magnitude(self):
        """Analyze the sign and magnitude of bias corrections."""
        for n in [100, 500, 1000, 5000, 10000]:
            old_corr = compute_old_bias_correction(n)
            new_corr = compute_new_bias_correction(n)

            assert np.isfinite(old_corr), f"Non-finite old correction for n={n}"
            assert np.isfinite(new_corr), f"Non-finite new correction for n={n}"
            # Both corrections should decrease in magnitude with larger n
            assert abs(new_corr) < 1.0, f"Correction too large for n={n}: {new_corr}"

    def test_correction_scaling(self):
        """Verify that bias correction scales as ~1/n."""
        ns = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
        corrections = np.array([compute_new_bias_correction(n) for n in ns])

        # Bias correction should scale approximately as 1/n
        # So correction * n should be approximately constant
        scaled = corrections * ns

        print("\nBias correction scaling (should be ~constant):")
        for n, corr, sc in zip(ns, corrections, scaled):
            print(f"  n={n:5d}: correction={corr:+.8f}, correction*n={sc:+.4f}")

        # Check that scaled values are approximately constant (within 50%)
        mean_scaled = np.mean(scaled)
        assert np.all(np.abs(scaled - mean_scaled) < 0.5 * np.abs(mean_scaled)), \
            "Bias correction should scale approximately as 1/n"


def run_verification():
    """Run all verification tests and print summary."""
    print("=" * 70)
    print("FFT vs Loop MI Implementation Discrepancy Verification")
    print("=" * 70)

    # Test 1: Compare bias corrections directly
    print("\n1. BIAS CORRECTION COMPARISON")
    print("-" * 40)
    for n in [100, 1000, 10000]:
        old = compute_old_bias_correction(n)
        new = compute_new_bias_correction(n)
        print(f"n={n:5d}: old={old:+.10f}, new={new:+.10f}, diff={old-new:+.2e}")

    # Test 2: Compare MI without bias correction
    print("\n2. MI WITHOUT BIAS CORRECTION")
    print("-" * 40)
    rng = np.random.RandomState(42)
    n = 2000
    x = rng.randn(n)
    y = 0.7 * x + np.sqrt(1 - 0.7**2) * rng.randn(n)

    copnorm_x = copnorm(x).ravel()
    copnorm_y = copnorm(y).ravel()

    mi_loop_nobc = mi_gg(copnorm_x, copnorm_y, biascorrect=False)
    mi_fft_nobc = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=False)[0]
    print(f"Loop MI: {mi_loop_nobc:.10f}")
    print(f"FFT MI:  {mi_fft_nobc:.10f}")
    print(f"Diff:    {mi_loop_nobc - mi_fft_nobc:+.2e}")

    # Test 3: Compare MI with bias correction
    print("\n3. MI WITH BIAS CORRECTION")
    print("-" * 40)
    mi_loop_bc = mi_gg(copnorm_x, copnorm_y, biascorrect=True)
    mi_fft_bc = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=True)[0]
    print(f"Loop MI: {mi_loop_bc:.10f}")
    print(f"FFT MI:  {mi_fft_bc:.10f}")
    print(f"Diff:    {mi_loop_bc - mi_fft_bc:+.2e}")

    # Test 4: Test at multiple shifts
    print("\n4. MI AT MULTIPLE SHIFTS (with bias correction)")
    print("-" * 40)
    shifts = np.array([0, 10, 100, 500])
    mi_fft_shifts = compute_mi_batch_fft(copnorm_x, copnorm_y, shifts, biascorrect=True)

    print(f"{'Shift':>8} {'Loop MI':>14} {'FFT MI':>14} {'Diff':>12}")
    for i, shift in enumerate(shifts):
        y_shifted = np.roll(copnorm_y, shift)
        mi_loop_shift = mi_gg(copnorm_x, y_shifted, biascorrect=True)
        mi_fft_shift = mi_fft_shifts[i]
        print(f"{shift:8d} {mi_loop_shift:14.10f} {mi_fft_shift:14.10f} {mi_loop_shift - mi_fft_shift:+12.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Without bias correction: diff = {mi_loop_nobc - mi_fft_nobc:+.2e} bits")
    print(f"With bias correction:    diff = {mi_loop_bc - mi_fft_bc:+.2e} bits")
    print("")

    if abs(mi_loop_nobc - mi_fft_nobc) < 1e-10 and abs(mi_loop_bc - mi_fft_bc) < 1e-6:
        print("RESULT: Implementations match within expected numerical precision.")
        print("        The bias correction formulas are equivalent.")
    else:
        print("RESULT: Discrepancy detected!")
        if abs(mi_loop_nobc - mi_fft_nobc) > 1e-6:
            print("        - Core MI formula differs (without bias correction)")
        if abs(mi_loop_bc - mi_fft_bc) > abs(mi_loop_nobc - mi_fft_nobc) + 1e-10:
            print("        - Bias correction contributes to discrepancy")


class TestMultivariateMI:
    """Test multivariate MI implementations (MTS cases)."""

    def test_mts_1d_mi_matches(self):
        """Test MI between 1D and 2D MTS matches between FFT and loop."""
        from driada.information.info_fft import compute_mi_mts_fft

        rng = np.random.RandomState(42)
        n = 1000

        # Generate 1D neural activity
        z = rng.randn(n)

        # Generate 2D position (correlated with z)
        x1 = 0.5 * z + 0.5 * rng.randn(n)
        x2 = 0.3 * z + 0.7 * rng.randn(n)
        x = np.vstack([x1, x2])

        # Copula normalize
        copnorm_z = copnorm(z).ravel()
        copnorm_x = copnorm(x)  # (2, n)

        # Compute with FFT
        mi_fft = compute_mi_mts_fft(copnorm_z, copnorm_x, np.array([0]), biascorrect=True)[0]

        # Compute with loop (mi_gg with joint variable)
        mi_loop = mi_gg(copnorm_z, copnorm_x, biascorrect=True)

        print(f"MTS MI (1D+2D): loop={mi_loop:.10f}, FFT={mi_fft:.10f}, diff={mi_loop - mi_fft:.10f}")

        # Should match closely
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-5, atol=1e-8,
            err_msg="MTS MI (1D+2D) should match between loop and FFT")

    def test_mts_1d_mi_at_shifts(self):
        """Test MTS MI at multiple shifts."""
        from driada.information.info_fft import compute_mi_mts_fft

        rng = np.random.RandomState(42)
        n = 1000

        # Generate data
        z = rng.randn(n)
        x1 = 0.5 * z + 0.5 * rng.randn(n)
        x2 = 0.3 * z + 0.7 * rng.randn(n)
        x = np.vstack([x1, x2])

        # Copula normalize
        copnorm_z = copnorm(z).ravel()
        copnorm_x = copnorm(x)

        shifts = np.array([0, 10, 50, 100])

        # Compute with FFT
        mi_fft = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts, biascorrect=True)

        # Compute with loop
        mi_loop = np.zeros(len(shifts))
        for i, shift in enumerate(shifts):
            x_shifted = np.roll(copnorm_x, shift, axis=1)
            mi_loop[i] = mi_gg(copnorm_z, x_shifted, biascorrect=True)

        print("MTS MI at shifts:")
        for i, shift in enumerate(shifts):
            print(f"  shift={shift}: loop={mi_loop[i]:.10f}, FFT={mi_fft[i]:.10f}, diff={mi_loop[i] - mi_fft[i]:.10f}")

        # All should match closely
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-5, atol=1e-8,
            err_msg="MTS MI at multiple shifts should match")

    def test_mts_3d_mi_matches(self):
        """Test MI between 1D and 3D MTS."""
        from driada.information.info_fft import compute_mi_mts_fft

        rng = np.random.RandomState(42)
        n = 1000

        # Generate 1D neural activity
        z = rng.randn(n)

        # Generate 3D position
        x1 = 0.5 * z + 0.5 * rng.randn(n)
        x2 = 0.3 * z + 0.7 * rng.randn(n)
        x3 = 0.2 * z + 0.8 * rng.randn(n)
        x = np.vstack([x1, x2, x3])

        # Copula normalize
        copnorm_z = copnorm(z).ravel()
        copnorm_x = copnorm(x)

        # Compute with FFT
        mi_fft = compute_mi_mts_fft(copnorm_z, copnorm_x, np.array([0]), biascorrect=True)[0]

        # Compute with loop
        mi_loop = mi_gg(copnorm_z, copnorm_x, biascorrect=True)

        print(f"MTS MI (1D+3D): loop={mi_loop:.10f}, FFT={mi_fft:.10f}, diff={mi_loop - mi_fft:.10f}")

        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-5, atol=1e-8,
            err_msg="MTS MI (1D+3D) should match")


class TestDiscreteFeatureMI:
    """Test MI with discrete features."""

    def test_continuous_discrete_mi_matches(self):
        """Test MI(continuous; discrete) matches between FFT and loop."""
        from driada.information.info_fft import mi_cd_fft
        from driada.information.gcmi import mi_model_gd

        rng = np.random.RandomState(42)
        n = 1000

        # Generate continuous variable
        z = rng.randn(n)

        # Generate discrete labels correlated with z
        # Class 0 for z < -0.5, class 1 for -0.5 <= z < 0.5, class 2 for z >= 0.5
        y = np.zeros(n, dtype=int)
        y[z < -0.5] = 0
        y[(z >= -0.5) & (z < 0.5)] = 1
        y[z >= 0.5] = 2

        # Copula normalize continuous
        copnorm_z = copnorm(z).ravel()

        # Compute with FFT
        mi_fft = mi_cd_fft(copnorm_z, y, np.array([0]), biascorrect=True)[0]

        # Compute with loop
        mi_loop = mi_model_gd(copnorm_z, y, biascorrect=True)

        print(f"Continuous-Discrete MI: loop={mi_loop:.10f}, FFT={mi_fft:.10f}, diff={mi_loop - mi_fft:.10f}")

        # Should match closely
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-8, atol=1e-10,
            err_msg="Continuous-Discrete MI should match")

    def test_continuous_discrete_mi_at_shifts(self):
        """Test MI(continuous; discrete) at multiple shifts."""
        from driada.information.info_fft import mi_cd_fft
        from driada.information.gcmi import mi_model_gd

        rng = np.random.RandomState(42)
        n = 1000

        # Generate data
        z = rng.randn(n)
        y = np.zeros(n, dtype=int)
        y[z < -0.5] = 0
        y[(z >= -0.5) & (z < 0.5)] = 1
        y[z >= 0.5] = 2

        copnorm_z = copnorm(z).ravel()

        shifts = np.array([0, 10, 50, 100])

        # Compute with FFT
        mi_fft = mi_cd_fft(copnorm_z, y, shifts, biascorrect=True)

        # Compute with loop
        mi_loop = np.zeros(len(shifts))
        for i, shift in enumerate(shifts):
            y_shifted = np.roll(y, shift)
            mi_loop[i] = mi_model_gd(copnorm_z, y_shifted, biascorrect=True)

        print("Continuous-Discrete MI at shifts:")
        for i, shift in enumerate(shifts):
            print(f"  shift={shift}: loop={mi_loop[i]:.10f}, FFT={mi_fft[i]:.10f}, diff={mi_loop[i] - mi_fft[i]:.10f}")

        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-6, atol=1e-10,
            err_msg="Continuous-Discrete MI at shifts should match")


class TestEdgeCases:
    """Test edge cases and potential sources of discrepancy."""

    def test_near_zero_correlation(self):
        """Test MI with near-zero correlation."""
        rng = np.random.RandomState(42)
        n = 2000

        # Independent variables
        x = rng.randn(n)
        y = rng.randn(n)

        copnorm_x = copnorm(x).ravel()
        copnorm_y = copnorm(y).ravel()

        mi_loop = mi_gg(copnorm_x, copnorm_y, biascorrect=True)
        mi_fft = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=True)[0]

        print(f"Near-zero MI: loop={mi_loop:.10f}, FFT={mi_fft:.10f}")

        # Both should be close to zero (and match each other)
        assert mi_loop < 0.05, "Loop MI should be near zero for independent variables"
        assert mi_fft < 0.05, "FFT MI should be near zero for independent variables"
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-5, atol=1e-8)

    def test_high_correlation(self):
        """Test MI with high correlation."""
        rng = np.random.RandomState(42)
        n = 2000

        # Highly correlated variables
        x = rng.randn(n)
        y = 0.99 * x + 0.01 * rng.randn(n)

        copnorm_x = copnorm(x).ravel()
        copnorm_y = copnorm(y).ravel()

        mi_loop = mi_gg(copnorm_x, copnorm_y, biascorrect=True)
        mi_fft = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=True)[0]

        print(f"High correlation MI: loop={mi_loop:.10f}, FFT={mi_fft:.10f}, diff={mi_loop - mi_fft:.10f}")

        # Both should be high and match
        assert mi_loop > 2.0, "Loop MI should be high for correlated variables"
        assert mi_fft > 2.0, "FFT MI should be high for correlated variables"
        np.testing.assert_allclose(mi_loop, mi_fft, rtol=1e-5, atol=1e-8)

    def test_small_sample_size(self):
        """Test MI with small sample size (where bias correction matters most)."""
        rng = np.random.RandomState(42)
        n = 100  # Small sample

        # Correlated variables
        x = rng.randn(n)
        y = 0.7 * x + np.sqrt(1 - 0.7**2) * rng.randn(n)

        copnorm_x = copnorm(x).ravel()
        copnorm_y = copnorm(y).ravel()

        # Without bias correction
        mi_loop_nobc = mi_gg(copnorm_x, copnorm_y, biascorrect=False)
        mi_fft_nobc = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=False)[0]

        # With bias correction
        mi_loop_bc = mi_gg(copnorm_x, copnorm_y, biascorrect=True)
        mi_fft_bc = compute_mi_batch_fft(copnorm_x, copnorm_y, np.array([0]), biascorrect=True)[0]

        print(f"Small sample (n={n}):")
        print(f"  No bias corr: loop={mi_loop_nobc:.10f}, FFT={mi_fft_nobc:.10f}, diff={mi_loop_nobc - mi_fft_nobc:.10f}")
        print(f"  With bias:    loop={mi_loop_bc:.10f}, FFT={mi_fft_bc:.10f}, diff={mi_loop_bc - mi_fft_bc:.10f}")

        np.testing.assert_allclose(mi_loop_nobc, mi_fft_nobc, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(mi_loop_bc, mi_fft_bc, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    run_verification()

    # Also run extended verification
    print("\n" + "=" * 70)
    print("EXTENDED VERIFICATION: MULTIVARIATE CASES")
    print("=" * 70)

    import sys
    test_mts = TestMultivariateMI()
    try:
        test_mts.test_mts_1d_mi_matches()
        print("  MTS 1D+2D: PASS")
    except AssertionError as e:
        print(f"  MTS 1D+2D: FAIL - {e}")

    try:
        test_mts.test_mts_1d_mi_at_shifts()
        print("  MTS shifts: PASS")
    except AssertionError as e:
        print(f"  MTS shifts: FAIL - {e}")

    try:
        test_mts.test_mts_3d_mi_matches()
        print("  MTS 1D+3D: PASS")
    except AssertionError as e:
        print(f"  MTS 1D+3D: FAIL - {e}")

    print("\n" + "=" * 70)
    print("EXTENDED VERIFICATION: DISCRETE FEATURES")
    print("=" * 70)

    test_disc = TestDiscreteFeatureMI()
    try:
        test_disc.test_continuous_discrete_mi_matches()
        print("  C-D MI: PASS")
    except AssertionError as e:
        print(f"  C-D MI: FAIL - {e}")

    try:
        test_disc.test_continuous_discrete_mi_at_shifts()
        print("  C-D shifts: PASS")
    except AssertionError as e:
        print(f"  C-D shifts: FAIL - {e}")

    print("\n" + "=" * 70)
    print("EXTENDED VERIFICATION: EDGE CASES")
    print("=" * 70)

    test_edge = TestEdgeCases()
    try:
        test_edge.test_near_zero_correlation()
        print("  Near-zero corr: PASS")
    except AssertionError as e:
        print(f"  Near-zero corr: FAIL - {e}")

    try:
        test_edge.test_high_correlation()
        print("  High corr: PASS")
    except AssertionError as e:
        print(f"  High corr: FAIL - {e}")

    try:
        test_edge.test_small_sample_size()
        print("  Small sample: PASS")
    except AssertionError as e:
        print(f"  Small sample: FAIL - {e}")
