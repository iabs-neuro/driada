"""Tests for higher-order information measures (TC, DTC, O-information)."""
import numpy as np
import pytest

from driada.information.gcmi import tc_gg, dtc_gg, o_info_gg, copnorm


def _gen_gaussian(n_vars, n_samples, cov=None, seed=0):
    """Generate zero-mean Gaussian samples with given covariance."""
    rng = np.random.default_rng(seed)
    if cov is None:
        cov = np.eye(n_vars)
    return rng.multivariate_normal(np.zeros(n_vars), cov, size=n_samples).T


class TestTCgg:
    def test_independent_gaussians_near_zero(self):
        """TC of independent Gaussian variables should be near zero."""
        x = _gen_gaussian(n_vars=5, n_samples=5000, seed=42)
        x_cn = copnorm(x)
        tc = tc_gg(x_cn)
        assert tc >= 0
        assert tc < 0.1  # should be near zero for independent variables

    def test_duplicated_signal_gives_large_tc(self):
        """TC of duplicated signals + noise should be large."""
        rng = np.random.default_rng(1)
        latent = rng.standard_normal(5000)
        x = np.vstack([latent + 0.01 * rng.standard_normal(5000) for _ in range(4)])
        x_cn = copnorm(x)
        tc = tc_gg(x_cn)
        assert tc > 5.0  # strong redundancy -> large TC

    def test_requires_at_least_2_variables(self):
        x = _gen_gaussian(n_vars=1, n_samples=1000)
        x_cn = copnorm(x)
        with pytest.raises(ValueError, match="at least 2 variables"):
            tc_gg(x_cn)

    def test_rejects_non_2d_input(self):
        x = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="2D"):
            tc_gg(x)


class TestDTCgg:
    def test_independent_gaussians_near_zero(self):
        """DTC of independent Gaussian variables should be near zero."""
        x = _gen_gaussian(n_vars=5, n_samples=5000, seed=42)
        x_cn = copnorm(x)
        dtc = dtc_gg(x_cn)
        assert dtc >= 0
        assert dtc < 0.1

    def test_tc_equals_dtc_for_n_eq_2(self):
        """For n=2, TC == DTC == I(X1; X2)."""
        rng = np.random.default_rng(7)
        x1 = rng.standard_normal(5000)
        x2 = 0.7 * x1 + 0.3 * rng.standard_normal(5000)
        x = np.vstack([x1, x2])
        x_cn = copnorm(x)
        tc = tc_gg(x_cn)
        dtc = dtc_gg(x_cn)
        assert abs(tc - dtc) < 1e-10

    def test_requires_at_least_2_variables(self):
        x = _gen_gaussian(n_vars=1, n_samples=1000)
        x_cn = copnorm(x)
        with pytest.raises(ValueError, match="at least 2 variables"):
            dtc_gg(x_cn)

    def test_rejects_non_2d_input(self):
        x = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="2D"):
            dtc_gg(x)
