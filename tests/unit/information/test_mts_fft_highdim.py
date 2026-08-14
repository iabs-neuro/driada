"""Tests for FFT-accelerated MI with MultiTimeSeries dimensions d = 4 and d = 5.

These exercise the high-dimensional extension of the FFT path, which is needed
for feature representations such as circular second-harmonic encodings
(``[cos, sin, cos2, sin2]``, d = 4) and anisotropic 2D place fields
(``[x, y, x^2, y^2, xy]``, d = 5).

Correctness is checked against the same loop references used for d <= 3:
``mi_gg`` for the continuous case and ``mi_model_gd`` for the discrete case,
evaluated per shift with ``np.roll``.
"""
import numpy as np
import pytest

from driada.information.info_fft import (
    compute_mi_mts_fft,
    compute_mi_mts_discrete_fft,
    _compute_joint_entropy_4x4_mts,
    _compute_joint_entropy_general_mts,
)
from driada.information.gcmi import mi_gg, mi_model_gd, copnorm


class TestContinuousHighDim:
    """compute_mi_mts_fft must match mi_gg per shift for d = 4 and d = 5."""

    @pytest.mark.parametrize("d", [4, 5])
    def test_matches_mi_gg_at_shifts(self, d):
        rng = np.random.RandomState(42 + d)
        n = 1000

        z = rng.randn(n)
        # d feature dimensions, each partially coupled to z (well conditioned)
        rows = []
        for k in range(d):
            w = 0.2 + 0.1 * k
            rows.append(w * z + np.sqrt(1 - w**2) * rng.randn(n))
        x = np.vstack(rows)

        copnorm_z = copnorm(z).ravel()
        copnorm_x = copnorm(x)  # (d, n)

        shifts = np.array([0, 10, 50, 100, 500])

        mi_fft = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts, biascorrect=True)

        mi_loop = np.zeros(len(shifts))
        for i, s in enumerate(shifts):
            x_shifted = np.roll(copnorm_x, int(s), axis=1)
            mi_loop[i] = mi_gg(copnorm_z, x_shifted, biascorrect=True)

        np.testing.assert_allclose(
            mi_fft, mi_loop, rtol=1e-5, atol=1e-8,
            err_msg=f"d={d}: FFT MI should match mi_gg at all shifts",
        )

    @pytest.mark.parametrize("d", [4, 5])
    def test_nonnegative_and_finite(self, d):
        rng = np.random.RandomState(7 + d)
        n = 300
        copnorm_x = rng.randn(d, n)
        copnorm_z = rng.randn(n)
        shifts = np.array([0, 1, 2, 5])
        mi = compute_mi_mts_fft(copnorm_z, copnorm_x, shifts)
        assert mi.shape == (len(shifts),)
        assert np.all(np.isfinite(mi))
        assert np.all(mi >= 0)

    def test_no_longer_raises_for_d4(self):
        """The d > 3 NotImplementedError must be gone for the extended path."""
        rng = np.random.RandomState(1)
        n = 200
        copnorm_x = rng.randn(4, n)
        copnorm_z = rng.randn(n)
        mi = compute_mi_mts_fft(copnorm_z, copnorm_x, np.array([0, 1]))
        assert np.all(np.isfinite(mi))


class TestGeneralDeterminantHelper:
    """The general (d+1) determinant must reproduce the closed-form d = 3 result."""

    def test_general_matches_closed_form_4x4(self):
        rng = np.random.RandomState(0)
        n = 800
        d = 3
        z = rng.randn(n)
        x = rng.randn(d, n)

        # Build the same statistics compute_mi_mts_fft uses internally.
        zc = z - z.mean()
        xc = x - x.mean(axis=1, keepdims=True)
        var_z = np.var(zc, ddof=1)
        cov_xx = np.cov(xc)

        shifts = np.array([0, 3, 17, 128])
        n_ = n
        fft_z = np.fft.rfft(zc)
        cov_zx = np.zeros((d, len(shifts)))
        sidx = shifts % n_
        for i in range(d):
            cc = np.fft.irfft(fft_z * np.conj(np.fft.rfft(xc[i])), n=n_)
            cov_zx[i] = cc[sidx] / (n_ - 1)

        h_closed = _compute_joint_entropy_4x4_mts(var_z, cov_xx, cov_zx)
        h_general = _compute_joint_entropy_general_mts(var_z, cov_xx, cov_zx)

        np.testing.assert_allclose(h_general, h_closed, rtol=1e-12, atol=1e-12)


class TestDiscreteHighDim:
    """compute_mi_mts_discrete_fft must match mi_model_gd per shift for d = 4, 5."""

    @pytest.mark.parametrize("d,Ym", [(4, 3), (5, 2)])
    def test_matches_mi_model_gd_at_shifts(self, d, Ym):
        rng = np.random.RandomState(100 + d)
        n = 1000
        mts = copnorm(rng.randn(d, n))
        discrete = rng.randint(0, Ym, n).astype(float)

        shifts = np.array([0, 7, 41, 200])
        mi_fft = compute_mi_mts_discrete_fft(mts, discrete, shifts, biascorrect=True)

        mi_loop = np.zeros(len(shifts))
        for i, s in enumerate(shifts):
            discrete_shifted = np.roll(discrete, int(s))
            mi_loop[i] = mi_model_gd(mts, discrete_shifted, Ym, True, True)

        np.testing.assert_allclose(
            mi_fft, mi_loop, rtol=1e-7, atol=1e-10,
            err_msg=f"d={d}: MTS-discrete FFT should match mi_model_gd at all shifts",
        )
