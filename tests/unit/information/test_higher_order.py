"""Tests for higher-order information measures (TC, DTC, O-information)."""
import numpy as np
import pytest

from driada.information.gcmi import tc_gg, dtc_gg, o_info_gg, copnorm
from driada.information.higher_order import (
    total_correlation,
    dual_total_correlation,
    o_information,
)
from driada.information.info_base import TimeSeries, MultiTimeSeries


def _gen_gaussian(n_vars, n_samples, cov=None, seed=0):
    """Generate zero-mean Gaussian samples with given covariance."""
    rng = np.random.default_rng(seed)
    if cov is None:
        cov = np.eye(n_vars)
    return rng.multivariate_normal(np.zeros(n_vars), cov, size=n_samples).T


def _make_mts(n_vars=4, n_samples=3000, seed=0, cov=None):
    """Helper: build a MultiTimeSeries from n continuous Gaussian signals."""
    rng = np.random.default_rng(seed)
    if cov is None:
        data = rng.standard_normal((n_vars, n_samples))
    else:
        data = rng.multivariate_normal(
            np.zeros(n_vars), cov, size=n_samples
        ).T
    ts_list = [
        TimeSeries(data[i], discrete=False, name=f"var_{i}")
        for i in range(n_vars)
    ]
    return MultiTimeSeries(ts_list)


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


class TestOInfoGg:
    def test_redundancy_positive(self):
        """Three variables sharing a latent should give positive Omega."""
        rng = np.random.default_rng(11)
        n_samples = 5000
        latent = rng.standard_normal(n_samples)
        x = np.vstack([
            latent + 0.3 * rng.standard_normal(n_samples),
            latent + 0.3 * rng.standard_normal(n_samples),
            latent + 0.3 * rng.standard_normal(n_samples),
        ])
        x_cn = copnorm(x)
        omega = o_info_gg(x_cn)
        assert omega > 0.5  # clearly redundancy-dominated

    def test_synergy_negative(self):
        """X3 = X1 + X2 + small noise should give negative Omega."""
        rng = np.random.default_rng(13)
        n_samples = 5000
        x1 = rng.standard_normal(n_samples)
        x2 = rng.standard_normal(n_samples)
        x3 = x1 + x2 + 0.05 * rng.standard_normal(n_samples)
        x = np.vstack([x1, x2, x3])
        x_cn = copnorm(x)
        omega = o_info_gg(x_cn)
        assert omega < -0.2  # synergy-dominated

    def test_return_components(self):
        """return_components=True returns (omega, tc, dtc) and omega == tc - dtc."""
        rng = np.random.default_rng(17)
        x = rng.standard_normal((4, 3000))
        x_cn = copnorm(x)
        omega, tc, dtc = o_info_gg(x_cn, return_components=True)
        assert isinstance(omega, float)
        assert isinstance(tc, float)
        assert isinstance(dtc, float)
        assert abs(omega - (tc - dtc)) < 1e-12

    def test_n_lt_3_raises(self):
        """O-info is identically 0 for n=2, so function refuses."""
        x = _gen_gaussian(n_vars=2, n_samples=1000)
        x_cn = copnorm(x)
        with pytest.raises(ValueError, match="n >= 3"):
            o_info_gg(x_cn)

    def test_rejects_non_2d_input(self):
        x = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="2D"):
            o_info_gg(x)


class TestOInfoClosedForm:
    def test_3d_gaussian_matches_analytic(self):
        """O-info on 3D Gaussian matches closed-form from covariance determinants.

        For a multivariate Gaussian with covariance Sigma,
        H(X) = 0.5 * log2((2*pi*e)^d * det(Sigma))
        Omega = (n-2)*H(X) + sum_i [H(X_i) - H(X_{-i})]
        """
        rng = np.random.default_rng(23)
        n_samples = 20000

        # Construct a 3D Gaussian with known structure
        sigma = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        x = rng.multivariate_normal(np.zeros(3), sigma, size=n_samples).T
        x_cn = copnorm(x)

        # Estimator output
        omega_est = o_info_gg(x_cn, biascorrect=False)

        # Analytic Omega from the SAMPLE covariance of copula-normalized data
        # (not from the true sigma, since copnorm changes marginals).
        # For Gaussian copula data, copnorm is approximately identity in
        # distribution -> sample cov of x_cn should be close to the correlation
        # matrix of sigma. We compute the analytic Omega using the sample cov
        # of the copula-normalized data for direct comparison.
        S = np.cov(x_cn, bias=False)
        two_pi_e = 2 * np.pi * np.e

        def gauss_h(cov):
            if np.ndim(cov) == 0:
                d = 1
                det = float(cov)
            else:
                d = cov.shape[0]
                det = np.linalg.det(cov)
            return 0.5 * np.log2((two_pi_e ** d) * det)

        h_joint = gauss_h(S)
        h_marg = sum(gauss_h(S[i, i]) for i in range(3))
        h_loo = 0.0
        for i in range(3):
            mask = np.ones(3, dtype=bool)
            mask[i] = False
            h_loo += gauss_h(S[np.ix_(mask, mask)])

        tc_analytic = h_marg - h_joint
        dtc_analytic = h_loo - 2 * h_joint
        omega_analytic = tc_analytic - dtc_analytic

        assert abs(omega_est - omega_analytic) < 1e-3


class TestTotalCorrelation:
    def test_returns_float(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=1)
        tc = total_correlation(mts)
        assert isinstance(tc, float)

    def test_matches_low_level(self):
        """High-level result equals low-level result on same data."""
        mts = _make_mts(n_vars=4, n_samples=3000, seed=2)
        tc_high = total_correlation(mts)
        tc_low = tc_gg(mts.copula_normal_data)
        assert abs(tc_high - tc_low) < 1e-12

    def test_rejects_non_mts(self):
        with pytest.raises(TypeError, match="MultiTimeSeries"):
            total_correlation(np.zeros((3, 100)))

    def test_ksg_not_implemented(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=3)
        with pytest.raises(NotImplementedError, match="curse of dimensionality"):
            total_correlation(mts, estimator="ksg")

    def test_unknown_estimator(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=4)
        with pytest.raises(ValueError, match="Unknown estimator"):
            total_correlation(mts, estimator="xyz")


class TestDualTotalCorrelation:
    def test_returns_float(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=5)
        dtc = dual_total_correlation(mts)
        assert isinstance(dtc, float)

    def test_matches_low_level(self):
        mts = _make_mts(n_vars=4, n_samples=3000, seed=6)
        dtc_high = dual_total_correlation(mts)
        dtc_low = dtc_gg(mts.copula_normal_data)
        assert abs(dtc_high - dtc_low) < 1e-12

    def test_rejects_non_mts(self):
        with pytest.raises(TypeError, match="MultiTimeSeries"):
            dual_total_correlation(np.zeros((3, 100)))

    def test_ksg_not_implemented(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=7)
        with pytest.raises(NotImplementedError, match="curse of dimensionality"):
            dual_total_correlation(mts, estimator="ksg")


class TestOInformation:
    def test_returns_float_by_default(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=8)
        omega = o_information(mts)
        assert isinstance(omega, float)

    def test_return_components_returns_dict(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=9)
        result = o_information(mts, return_components=True)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"omega", "tc", "dtc"}
        assert abs(result["omega"] - (result["tc"] - result["dtc"])) < 1e-12

    def test_matches_low_level(self):
        mts = _make_mts(n_vars=4, n_samples=3000, seed=10)
        omega_high = o_information(mts)
        omega_low = o_info_gg(mts.copula_normal_data)
        assert abs(omega_high - omega_low) < 1e-12

    def test_rejects_non_mts(self):
        with pytest.raises(TypeError, match="MultiTimeSeries"):
            o_information(np.zeros((3, 100)))

    def test_ksg_not_implemented(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=11)
        with pytest.raises(NotImplementedError, match="curse of dimensionality"):
            o_information(mts, estimator="ksg")

    def test_unknown_estimator(self):
        mts = _make_mts(n_vars=4, n_samples=2000, seed=12)
        with pytest.raises(ValueError, match="Unknown estimator"):
            o_information(mts, estimator="xyz")

    def test_n_lt_3_mts_raises(self):
        """MTS with only 2 components -> o_info_gg raises n>=3 error."""
        mts = _make_mts(n_vars=2, n_samples=2000, seed=13)
        with pytest.raises(ValueError, match="n >= 3"):
            o_information(mts)


class TestCopulaDataMissing:
    def test_discrete_mts_raises(self):
        """MTS built from discrete TS has no copula_normal_data."""
        rng = np.random.default_rng(14)
        # Discrete binary signals
        data = (rng.random((3, 2000)) > 0.5).astype(int)
        ts_list = [
            TimeSeries(data[i], discrete=True, name=f"d{i}")
            for i in range(3)
        ]
        mts = MultiTimeSeries(ts_list, allow_zero_columns=True)
        # Sanity: discrete MTS should have copula_normal_data == None
        assert mts.copula_normal_data is None

        with pytest.raises(ValueError, match="copula_normal_data"):
            o_information(mts)

    def test_after_clear_caches_raises(self):
        """clear_caches() sets copula_normal_data to None."""
        mts = _make_mts(n_vars=4, n_samples=1500, seed=15)
        # Sanity: continuous MTS has copula_normal_data before clearing
        assert mts.copula_normal_data is not None

        mts.clear_caches()
        assert mts.copula_normal_data is None

        with pytest.raises(ValueError, match="clear_caches"):
            total_correlation(mts)
